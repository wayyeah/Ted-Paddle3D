import paddle
import paddle.nn as nn
from functools import partial
from paddle import sparse
from paddle3d.apis import manager
import numpy as np
import time 
def post_act_block(in_channels, out_channels, kernel_size, stride, padding, conv_type='subm',indice_key=None,norm_fn=None):
    if conv_type == 'subm':
        conv = sparse.nn.SubmConv3D(in_channels, out_channels, kernel_size, stride=stride,padding=padding,  bias_attr=False, key=indice_key)
        relu=sparse.nn.ReLU()
    elif conv_type=='spconv':
        conv = sparse.nn.Conv3D(in_channels, out_channels, kernel_size, stride=stride,padding=padding,bias_attr=False)
        relu=sparse.nn.ReLU()
    elif conv_type=='inverseconv':
        #!!!inverseconv is not supported in paddlepaddle 2.0.0
        #conv = sparse.InverseConv3d(in_channels, out_channels, kernel_size,  bias_attr=False)
        conv = sparse.nn.Conv3D(in_channels, out_channels, kernel_size,  stride=stride,padding=padding,bias_attr=False, key=indice_key)
        relu=sparse.nn.ReLU()
    else:
        raise NotImplementedError
    m=nn.Sequential(
        conv,
        norm_fn(out_channels),
        relu,
    )
    return m


def replace_feature(out, new_features):
    
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        
        indices =out.indices()
        ss=out.shape
        ss[-1]=ss[-1]*2
        out_updated = sparse.sparse_coo_tensor(indices, new_features, ss,stop_gradient=False)
         
        return out_updated
class BasicBlock(nn.Layer):
    def __init__(self, inplanes, planes,  norm_fn=None,stride=2,  padding=1,  indice_key=None):
        super(BasicBlock, self).__init__()
        assert norm_fn is not None
        block = post_act_block
        self.stride = stride
        if stride >1:
            self.down_conv = block(inplanes,
                                    planes,
                                    3,
                                    norm_fn=norm_fn,
                                    stride=2,
                                    padding=padding,
                                    indice_key=('sp' + indice_key),
                                    conv_type='spconv')
        if stride >1:
            conv_in = planes
        else:
            conv_in = inplanes

        self.conv1 = block(conv_in,
                              planes // 2,
                              3,
                              norm_fn=norm_fn,
                              stride=1,
                              padding=1,
                              indice_key=('subm1' + indice_key))
        self.conv2 = block(planes//2,
                              planes // 2,
                              3,
                              stride=1,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm2' + indice_key))

        self.conv3 = block(planes//2,
                              planes // 2,
                              3,
                              norm_fn=norm_fn,
                              stride=1,
                              padding=1,
                              indice_key=('subm3' + indice_key))
        self.conv4 = block(planes//2,
                              planes // 2,
                              3,
                              norm_fn=norm_fn,
                              stride=1,
                              padding=1,
                              indice_key=('subm4' + indice_key))


    def forward(self, x):
        if self.stride>1:
            x = self.down_conv(x)
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        
        out = replace_feature(x2, paddle.concat([x1.values(), x4.values()],-1))
       
        return out

__all__ = ['TeVoxelBackBone8x']
@manager.BACKBONES.add_component
class TeVoxelBackBone8x(nn.Layer):
    def __init__(self,input_channels=4,**kwargs ):
        super(TeVoxelBackBone8x, self).__init__()
        self.return_num_features_as_dict=kwargs['return_num_features_as_dict']
        self.out_features=kwargs['out_featues']
        grid_size = (np.array(kwargs['point_cloud_range'][3:6]) - np.array(kwargs['point_cloud_range'][0:3])) / np.array(kwargs['voxel_size'])
        num_filters = kwargs['num_filters']
        
        
        norm_fn=partial(sparse.nn.BatchNorm, epsilon=1e-3, momentum=1-0.01)
        self.sparse_shape=grid_size[::-1]+[1,0,0]
        self.input_channels=input_channels
        self.conv_input = nn.Sequential(
            sparse.nn.SubmConv3D(input_channels, num_filters[0],3,stride=1,padding=1, bias_attr=False,key='subm1'),
            norm_fn(num_filters[0]), 
            sparse.nn.ReLU()
            
        )
        self.conv1 = nn.Sequential(
            post_act_block(num_filters[0], num_filters[0], 3,norm_fn=norm_fn,stride=1, padding=0,indice_key='conv1'),
        )
       
        self.conv2 = BasicBlock(num_filters[0], num_filters[1],norm_fn=norm_fn,stride=2,padding=[1,1,1], indice_key='conv2')
        
        self.conv3 = BasicBlock(num_filters[1], num_filters[2],norm_fn=norm_fn,stride=2,padding=[1,1,1],  indice_key='conv3')
        
        self.conv4 = BasicBlock(num_filters[2], num_filters[3],norm_fn=norm_fn,stride=2,  padding=(0, 1, 1),indice_key='conv4')
        
        last_pad=0
        #last_pad=self.model_cfg.get('last_pad',last_pad)
        self.conv_out=nn.Sequential(
            sparse.nn.Conv3D(num_filters[3],self.out_features,(3,1,1),stride=(2,1,1),padding=last_pad,bias_attr=False),
            norm_fn(self.out_features),
            sparse.nn.ReLU(),
        )
        
        '''if self.model_cfg.get("MM",False):
            self.conv_input_2=nn.Sequential(
                sparse.nn.SubmConv3D(input_channels,num_filters[0],3,padding=1,bias_attr=False,key='subm1_2'),
                norm_fn(num_filters[0]),
                nn.ReLU(),
            )
            block=post_act_block
            self.conv1_2=nn.Sequential(
                block(num_filters[0],num_filters[0],3,norm_fn=norm_fn,padding=1,indice_key='sbum1_2'),
            )
            self.conv2_2=nn.Sequential(
                block(num_filters[0],num_filters[1],3,norm_fn=norm_fn,stride=2,padding=1,indice_key='spconv2_2',conv_type='spconv'),
                block(num_filters[1],num_filters[1],3,norm_fn=norm_fn,padding=1,indice_key='subm2_2'),
                block(num_filters[1],num_filters[1],3,norm_fn=norm_fn,padding=1,indice_key='subm2_2'),
            )
            self.conv3_2=nn.Sequential(
                block(num_filters[1],num_filters[2],3,norm_fn=norm_fn,stride=2,padding=1,indice_key='spconv3_2',conv_type='spconv'),
                block(num_filters[2],num_filters[2],3,norm_fn=norm_fn,padding=1,indice_key='subm3_2'),
                block(num_filters[2],num_filters[2],3,norm_fn=norm_fn,padding=1,indice_key='subm3_2'),
            )
            self.conv4_2=nn.Sequential(
                block(num_filters[2],num_filters[3],3,norm_fn=norm_fn,stride=2,padding=(0,1,1),indice_key='spconv4_2',conv_type='spconv'),
                block(num_filters[3],num_filters[3],3,norm_fn=norm_fn,padding=1,indice_key='subm4_2'),
                block(num_filters[3],num_filters[3],3,norm_fn=norm_fn,padding=1,indice_key='subm4_2'),
            )'''
        
        self.num_point_features=self.out_features
        if self.return_num_features_as_dict:
            num_point_features = {}
            num_point_features.update({
                'x_conv1': num_filters[0],
                'x_conv2': num_filters[1],
                'x_conv3': num_filters[2],
                'x_conv4': num_filters[3],
            })
            self.num_point_features = num_point_features

    def decompose_tensor(self,tensor,i,batch_size):
        
        input_shape=tensor.shape[3]
        begin_shape_ids=i*(input_shape//4)
        end_shape_ids=(i+1)*(input_shape//4)
        x_conv3_features=tensor.values()
        x_conv3_coords=tensor.indices().transpose((1, 0))
        mask=(begin_shape_ids<x_conv3_coords[:,3])&(x_conv3_coords[:,3]<end_shape_ids)
        this_conv3_feat=x_conv3_features[mask]
        this_conv3_coords=x_conv3_coords[mask]
        this_conv3_coords[:,3]-=i*(input_shape//4)
        this_conv3_coords=this_conv3_coords.astype('int32')
        this_shape=[tensor.shape[0],tensor.shape[1],tensor.shape[2],tensor.shape[3]//4,tensor.shape[4]]
        
        this_conv3_tensor=sparse.sparse_coo_tensor(values=this_conv3_feat,indices=this_conv3_coords.transpose((1,0)),shape=this_shape,stop_gradient=False)
        return this_conv3_tensor

    def forward_test(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            rot_num = trans_param.shape[1]
        else:
            rot_num = 1

        all_lidar_feat = []
        all_lidar_coords = []
        
        new_shape = [batch_dict['batch_size'],int(self.sparse_shape[0]), int(self.sparse_shape[1]), int(self.sparse_shape[2] * 4),self.input_channels]

        for i in range(rot_num):
            if i==0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)

            voxel_features, voxel_coords = batch_dict['voxel_features'+rot_num_id], batch_dict['voxel_coords'+rot_num_id]

            all_lidar_feat.append(voxel_features)
            new_coord = voxel_coords.clone()
            
            
            new_coord[:, 3] += i*int(self.sparse_shape[2])
            all_lidar_coords.append(new_coord)
        batch_size = batch_dict['batch_size']

        all_lidar_feat = paddle.concat(all_lidar_feat, 0)
        all_lidar_coords = paddle.concat(all_lidar_coords)
        all_lidar_coords=all_lidar_coords.astype('int32')
        
        
        input_sp_tensor = sparse.sparse_coo_tensor(
            values=all_lidar_feat,
            indices=all_lidar_coords.transpose((1, 0)),
            shape=new_shape
        )
        
        x = self.conv_input(input_sp_tensor)
        
        x_conv1 = self.conv1(x)
       
        x_conv2 = self.conv2(x_conv1)
        
        x_conv3 = self.conv3(x_conv2)
     
        x_conv4 = self.conv4(x_conv3)
        
        out = self.conv_out(x_conv4)
       
        for i in range(rot_num):
            if i==0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)
            
            this_conv3 = self.decompose_tensor(x_conv3, i, batch_size)
            this_conv4 = self.decompose_tensor(x_conv4, i, batch_size)
            this_out = self.decompose_tensor(out, i, batch_size)
           
            batch_dict.update({
                'encoded_spconv_tensor'+rot_num_id: this_out,
                'encoded_spconv_tensor_stride'+rot_num_id: 8,
            })
            batch_dict.update({
                'multi_scale_3d_features'+rot_num_id: {
                    'x_conv1': None,
                    'x_conv2': None,
                    'x_conv3': this_conv3,
                    'x_conv4': this_conv4,
                },
                'multi_scale_3d_strides'+rot_num_id: {
                    'x_conv1': 1,
                    'x_conv2': 2,
                    'x_conv3': 4,
                    'x_conv4': 8,
                }
            })


        if "MM" in batch_dict:
            all_mm_feat = []
            all_mm_coords = []
            for i in range(rot_num):
                if i == 0:
                    rot_num_id = ''
                else:
                    rot_num_id = str(i)

                newvoxel_features, newvoxel_coords = batch_dict['voxel_features_mm'+rot_num_id], batch_dict['voxel_coords_mm'+rot_num_id]

                all_mm_feat.append(newvoxel_features)
                new_mm_coord = newvoxel_coords.clone()
                new_mm_coord[:, 3] += i * self.sparse_shape[2]
                all_mm_coords.append(new_mm_coord)
            all_mm_feat = paddle.concat(all_mm_feat, 0)
            all_mm_coords = paddle.concat(all_mm_coords)
            all_mm_coords = all_mm_coords.astype('int32')

            newinput_sp_tensor = sparse.sparse_coo_tensor(
                values=all_mm_feat,
                indices=all_mm_coords,
                shape=new_shape,
                
            )

            newx = self.conv_input_2(newinput_sp_tensor)

            newx_conv1 = self.conv1_2(newx)
            newx_conv2 = self.conv2_2(newx_conv1)
            newx_conv3 = self.conv3_2(newx_conv2)
            newx_conv4 = self.conv4_2(newx_conv3)

            for i in range(rot_num):
                if i == 0:
                    rot_num_id = ''
                else:
                    rot_num_id = str(i)

                this_conv3 = self.decompose_tensor(newx_conv3, i, batch_size)
                this_conv4 = self.decompose_tensor(newx_conv4, i, batch_size)
                batch_dict.update({
                    'encoded_spconv_tensor_stride_mm'+rot_num_id: 8
                })
                batch_dict.update({
                    'multi_scale_3d_features_mm'+rot_num_id: {
                        'x_conv1': None,
                        'x_conv2': None,
                        'x_conv3': this_conv3,
                        'x_conv4': this_conv4,
                    },
                    'multi_scale_3d_strides'+rot_num_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })
 
        return batch_dict

    def forward_train(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            rot_num = trans_param.shape[1]
        else:
            rot_num = 1
        
        for i in range(rot_num):
            if i==0:
                rot_num_id = ''
            else:
                rot_num_id = str(i)
            
            voxel_features, voxel_coords = batch_dict['voxel_features'+rot_num_id], batch_dict['voxel_coords'+rot_num_id]
           
            batch_size = batch_dict['batch_size']
            
            shape = [batch_size] + list(self.sparse_shape) + [self.input_channels]
           
            input_sp_tensor = sparse.sparse_coo_tensor(
                values=voxel_features,
                indices=voxel_coords.transpose((1, 0)),
                shape=shape,stop_gradient=False
            )
            
            
            x = self.conv_input(input_sp_tensor)
           
            
            
            x_conv1 = self.conv1(x)
           
           
           
            x_conv2 = self.conv2(x_conv1)
           
           
            start=time.time()
            x_conv3 = self.conv3(x_conv2)
          
            
           
            x_conv4 = self.conv4(x_conv3)
           
            
          
            # [200, 176, 5] -> [200, 176, 2]
            
            out = self.conv_out(x_conv4)
            
            
            batch_dict.update({
                'encoded_spconv_tensor'+rot_num_id: out,
                'encoded_spconv_tensor_stride'+rot_num_id: 8,
            })
        
            batch_dict.update({
                'multi_scale_3d_features'+rot_num_id: {
                    'x_conv1': x_conv1,
                    'x_conv2': x_conv2,
                    'x_conv3': x_conv3,
                    'x_conv4': x_conv4,
                },
                'multi_scale_3d_strides'+rot_num_id: {
                    'x_conv1': 1,
                    'x_conv2': 2,
                    'x_conv3': 4,
                    'x_conv4': 8,
                }
            })
           
            if 'MM' in batch_dict:
                newvoxel_features, newvoxel_coords = batch_dict['voxel_features_mm'+rot_num_id], batch_dict['voxel_coords_mm'+rot_num_id]
                newinput_sp_tensor = sparse.sparse_coo_tensor(
                    values=newvoxel_features,
                    indices=newvoxel_coords,
                    shape=self.sparse_shape,
                    stop_gradient=False
                   
                )
                newx = self.conv_input_2(newinput_sp_tensor)

                newx_conv1 = self.conv1_2(newx)
                newx_conv2 = self.conv2_2(newx_conv1)
                newx_conv3 = self.conv3_2(newx_conv2)
                newx_conv4 = self.conv4_2(newx_conv3)

                # for detection head
                # [200, 176, 5] -> [200, 176, 2]
                #newout = self.conv_out(newx_conv4)

                batch_dict.update({
                    #'encoded_spconv_tensor_mm': newout,
                    'encoded_spconv_tensor_stride_mm'+rot_num_id: 8
                })
                batch_dict.update({
                    'multi_scale_3d_features_mm'+rot_num_id: {
                        'x_conv1': newx_conv1,
                        'x_conv2': newx_conv2,
                        'x_conv3': newx_conv3,
                        'x_conv4': newx_conv4,
                    },
                    'multi_scale_3d_strides'+rot_num_id: {
                        'x_conv1': 1,
                        'x_conv2': 2,
                        'x_conv3': 4,
                        'x_conv4': 8,
                    }
                })
        
        return batch_dict

    def forward(self, batch_dict):
        if 'gt_boxes' not in batch_dict:
            self.training = False
        if self.training and not self.in_export_mode:
            return self.forward_train(batch_dict)
        else:
            return self.forward_test(batch_dict)


