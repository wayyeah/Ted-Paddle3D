import numpy as np
import paddle
import paddle.nn as nn
from paddle3d.apis import manager
from paddle3d.transforms.transform import X_TRANS
__all__ = ['BEVPool','BaseBEVBackbone']

@manager.NECKS.add_component
class BaseBEVBackbone(nn.Layer):
    def __init__(self, input_channels=256,**kwargs):
        super().__init__()
        
        '''layer_nums=[4,4]
        layer_strides=[1,2]
        num_filters=[64,128]
        num_upsample_filters =  [128,128]
        upsample_strides = [1,2]'''
        
        self.model_cfg = kwargs
       
        
        if self.model_cfg.get('layer_nums',None) is not None:
            assert len(self.model_cfg.get('layer_nums')) == len(self.model_cfg.get('layer_strides'))==len(self.model_cfg.get('num_filters'))
            layer_nums = self.model_cfg.get('layer_nums')
            layer_strides = self.model_cfg.get('layer_strides')
            num_filters = self.model_cfg.get('num_filters')
        else:
            layer_nums=layer_strides=num_filters=[]
        
        if self.model_cfg.get('upsample_strides', None) is not None:
            assert len(self.model_cfg.get('upsample_strides')) == len(self.model_cfg.get('num_upsample_filters'))
            num_upsample_filters = self.model_cfg.get('num_upsample_filters')
            upsample_strides = self.model_cfg.get('upsample_strides')
        else:
            upsample_strides = num_upsample_filters = []
        

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]

        self.blocks=nn.LayerList()
        self.deblocks=nn.LayerList()
        for idx in range(num_levels):
            cur_layers = [
               nn.ZeroPad2D(1),
                nn.Conv2D(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias_attr=False
                ),
                nn.BatchNorm2D(num_filters[idx], epsilon=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2D(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias_attr=False),
                    nn.BatchNorm2D(num_filters[idx], epsilon=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2DTranspose(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias_attr=False
                        ),
                        nn.BatchNorm2D(num_upsample_filters[idx], epsilon=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2D(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias_attr=False
                        ),
                        nn.BatchNorm2D(num_upsample_filters[idx], epsilon=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.Conv2DTranspose(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias_attr=False),
                nn.BatchNorm2D(c_in, epsilon=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        #for child in self.children():
        #    for param in child.parameters():
        #        param.requires_grad = False
        self.num_bev_features_post = c_in


    def forward(self,data_dict):
        spatial_features = data_dict['spatial_features']
        ups = []
        x = spatial_features
        
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            stride = int(spatial_features.shape[2] / x.shape[2])
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = paddle.concat(ups, axis=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x #may question



        return data_dict
def bilinear_interpolate_paddle(im,x,y):
    x0=paddle.floor(x)
    x1=x0+1
    
    y0=paddle.floor(y)
    y1=y0+1

    x0=paddle.clip(x0,0,im.shape[1]-1).astype('int32')
    x1=paddle.clip(x1,0,im.shape[1]-1).astype('int32')
    y0=paddle.clip(y0,0,im.shape[0]-1).astype('int32')
    y1=paddle.clip(y1,0,im.shape[0]-1).astype('int32')

    
    Ia=im[y0,x0]
    Ib=im[y1,x0]
    Ic=im[y0,x1]
    Id=im[y1,x1]

    wa = (x1.astype(x.dtype) - x) * (y1.astype(y.dtype) - y)
    wb = (x1.astype(x.dtype) - x) * (y - y0.astype(y.dtype))
    wc = (x - x0.astype(x.dtype)) * (y1.astype(y.dtype) - y)
    wd = (x - x0.astype(x.dtype)) * (y - y0.astype(y.dtype))
    ans = (paddle.t(paddle.t(Ia) * wa) +
       paddle.t(paddle.t(Ib) * wb) +
       paddle.t(paddle.t(Ic) * wc) +
       paddle.t(paddle.t(Id) * wd))
    return ans
@manager.NECKS.add_component
class BEVPool(nn.Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.model_cfg=kwargs
        self.num_bev_features =self.model_cfg['num_bev_features']
        self.RANGE =self.model_cfg['point_cloud_range']
        self.x_trans = X_TRANS()
        self.point_cloud_range = self.model_cfg['point_cloud_range']
        self.voxel_size = self.model_cfg['voxel_size']
    def get_pseudo_points(self, pts_range=[0, -40, -3, 70.4, 40, 1], voxel_size=[0.05, 0.05, 0.05], stride=8):
        x_stride = voxel_size[0] * stride
        y_stride = voxel_size[1] * stride

        min_x = pts_range[0] + x_stride / 2
        max_x = pts_range[3]  # + x_stride / 2
        min_y = pts_range[1] + y_stride / 2
        max_y = pts_range[4] + y_stride / 2

        x = np.arange(min_x, max_x, x_stride)
        y = np.arange(min_y, max_y, y_stride)

        x, y = np.meshgrid(x, y)
        zeo = np.zeros(shape=x.shape)

        grids =paddle.transpose(paddle.to_tensor(np.stack([x, y, zeo]).astype(np.float32)),[1,2,0]) 

        return grids
    def interpolate_from_bev_features(self, points, bev_features, bev_stride):

        cur_batch_points = points

        x_idxs = (cur_batch_points[:, 0] - self.point_cloud_range[0]) / self.voxel_size[0]
        y_idxs = (cur_batch_points[:, 1] - self.point_cloud_range[1]) / self.voxel_size[1]
        cur_x_idxs = x_idxs / bev_stride
        cur_y_idxs = y_idxs / bev_stride

        cur_bev_features =paddle.transpose(bev_features, [1, 2, 0]) 
        
        point_bev_features = bilinear_interpolate_paddle(cur_bev_features, cur_x_idxs, cur_y_idxs)

        return point_bev_features
    def bev_align(self, bev_feat, transform_param, stride, stage_i):
        batch_size = len(bev_feat)
        w, h = bev_feat.shape[-2], bev_feat.shape[-1]

        all_feat = []
        for bt_i in range(batch_size):
            cur_bev_feat = bev_feat[bt_i]
            grid_pts = self.get_pseudo_points(self.point_cloud_range, self.voxel_size, stride)

            grid_pts =paddle.reshape(grid_pts,[-1,3])
            bt_transform_param = transform_param[bt_i]
            previous_stage_param = bt_transform_param[0]
            current_stage_param = bt_transform_param[stage_i]
            
            trans_dict = self.x_trans.forward_with_param({'points': grid_pts,
                                                        'transform_param': current_stage_param})
            trans_dict = self.x_trans.backward_with_param({'points': trans_dict['points'],
                                                        'transform_param': previous_stage_param})

            aligned_feat =paddle.reshape(self.interpolate_from_bev_features(trans_dict['points'], cur_bev_feat, stride),[w, h, -1])
            aligned_feat = paddle.transpose(aligned_feat, [2, 0, 1])
            
            all_feat.append(aligned_feat)

        return paddle.stack(all_feat)
    def forward(self, batch_dict):
        
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            rot_num = trans_param.shape[1]
        else:
            rot_num = 1

        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']

        all_feat = []

        for i in range(rot_num):
            if i == 0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)

            encoded_spconv_tensor = batch_dict['encoded_spconv_tensor' + rot_num_id] #差异大
            
            spatial_features = encoded_spconv_tensor.to_dense()
            
            N, D, H, W, C = spatial_features.shape #spatial_features N*D*H*W*C
            spatial_features=spatial_features.transpose([0, 4, 1, 2, 3])
            
            spatial_features = paddle.reshape(spatial_features,[N, C * D, H, W])

            batch_dict['spatial_features' + rot_num_id] = spatial_features
            
            if i == 0:
                all_feat.append(spatial_features)
            elif 'transform_param' in batch_dict and i > 0:
                aligned_bev_feat = self.bev_align(spatial_features.clone(),
                                                batch_dict['transform_param'],
                                                batch_dict['spatial_features_stride'],
                                                i)
                all_feat.append(aligned_bev_feat)
        
        if 'transform_param' in batch_dict:
            all_feat = paddle.stack(all_feat)#差距大

            if self.model_cfg['align_method'] == 'max':
                final_feat = all_feat.max(0)
                batch_dict['spatial_features'] =final_feat
            elif self.model_cfg['align_method'] == 'mean':
                final_feat = all_feat.mean(0)
                batch_dict['spatial_features'] =final_feat
            else:
                raise NotImplementedError
        
        return batch_dict