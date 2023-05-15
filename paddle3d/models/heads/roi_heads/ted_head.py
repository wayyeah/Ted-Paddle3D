import paddle
import paddle.nn as nn
from paddle3d.models.heads.roi_heads import RoIHeadBase
from paddle3d.models.common.pointnet2_stack import \
    voxel_pool_modules as voxelpool_stack_modules
from paddle3d.utils import spconv_utils,common_utils
from paddle3d.models.common import box_utils,generate_voxel2pinds
import paddle.nn.functional as F
import numpy as np
from functools import partial
from paddle3d.apis import manager
import pickle
import copy

import time
from paddle3d.models.layers import (constant_init, kaiming_normal_init,
                                    xavier_normal_init)

from paddle3d.transforms.transform import X_TRANS
class PositionalEmbedding(nn.Layer):
    def __init__(self, demb=256):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (paddle.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    # pos_seq =  pos_seq = paddle.arange(seq_len-1, -1, -1.0)
    def forward(self, pos_seq, batch_size=2):
        sinusoid_inp = paddle.outer(pos_seq, self.inv_freq)
        pos_emb = paddle.concat([sinusoid_inp.sin(), sinusoid_inp.cos()], axis=-1)

        if batch_size is not None:
            return pos_emb[:, None, :].expand(-1, batch_size, -1)
        else:
            return pos_emb[:, None, :]
class CrossAttention(nn.Layer):

    def __init__(self, hidden_dim, pos=True, head=4):
        super(CrossAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.pos_dim = 8
        self.pos = pos

        if self.pos:
            self.pos_en = PositionalEmbedding(self.pos_dim)

            self.Q_linear = nn.Linear(hidden_dim + self.pos_dim, hidden_dim, bias_attr=False)
            self.K_linear = nn.Linear(hidden_dim + self.pos_dim, hidden_dim, bias_attr=False)
            self.V_linear = nn.Linear(hidden_dim + self.pos_dim, hidden_dim, bias_attr=False)
        else:
            self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias_attr=False)
            self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias_attr=False)
            self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias_attr=False)

        self.att = nn.MultiHeadAttention(hidden_dim, head)

    def forward(self, inputs, Q_in):  # N, B, C

        batch_size = inputs.shape[1]
        seq_len = inputs.shape[0]

        if self.pos:
            pos_input = paddle.to_tensor(np.arange(seq_len) + 1,stop_gradient=False)
            pos_input = self.pos_en(pos_input, batch_size)
            inputs_pos = paddle.concat([inputs, pos_input], -1)
            pos_Q = paddle.to_tensor(np.array([seq_len]),stop_gradient=False)
            pos_Q = self.pos_en(pos_Q, batch_size)
            Q_in_pos = paddle.concat([Q_in, pos_Q], -1)
        else:
            inputs_pos = inputs
            Q_in_pos = Q_in

        Q = self.Q_linear(Q_in_pos)
        K = self.K_linear(inputs_pos)
        V = self.V_linear(inputs_pos)

        out = self.att(Q, K, V)

        return out[0]
class Attention_Layer(nn.Layer):

    def __init__(self, hidden_dim):
        super(Attention_Layer, self).__init__()

        self.hidden_dim = hidden_dim

        self.Q_linear = nn.Linear(hidden_dim, hidden_dim, bias_attr=False)
        self.K_linear = nn.Linear(hidden_dim, hidden_dim, bias_attr=False)
        self.V_linear = nn.Linear(hidden_dim, hidden_dim, bias_attr=False)

    def forward(self, inputs):  # B, K, N

        Q = self.Q_linear(inputs)
        K = paddle.transpose(self.K_linear(inputs), (0, 2, 1))
        V = self.V_linear(inputs)

        alpha = paddle.matmul(Q, K)

        alpha = F.softmax(alpha, axis=2)

        out = paddle.matmul(alpha, V)

        out = paddle.mean(out, axis=-2)

        return out
def gen_sample_grid(rois, grid_size=7, grid_offsets=(0, 0), spatial_scale=1.):
    faked_features = paddle.ones((grid_size, grid_size))
    N = rois.shape[0]
    dense_idx = paddle.nonzero(faked_features, as_tuple=False)  # (N, 2) [x_idx, y_idx]
    dense_idx = dense_idx.tile((N, 1)).reshape((N, -1, 2)).astype("float32")  # (B, 7 * 7, 2)

    local_roi_size = rois.reshape(N, -1)[:, 3:5]
    local_roi_grid_points = (dense_idx ) / (grid_size-1) * local_roi_size.unsqueeze(dim=1) \
                      - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 7 * 7, 2)

    ones = paddle.ones_like(local_roi_grid_points[..., 0:1])
    local_roi_grid_points = paddle.concat([local_roi_grid_points, ones], -1)

    global_roi_grid_points = box_utils.rotate_points_along_z(
        local_roi_grid_points.clone(), rois[:, 6]
    ).squeeze(dim=1)
    global_center = rois[:, 0:3].clone()
    global_roi_grid_points += global_center.unsqueeze(dim=1)

    x = global_roi_grid_points[..., 0:1]
    y = global_roi_grid_points[..., 1:2]

    x = (x.transpose(1, 2).transpose(0, 2) + grid_offsets[0]) * spatial_scale
    y = (y.transpose(1, 2).transpose(0, 2) + grid_offsets[1]) * spatial_scale

    return x.reshape(grid_size**2, -1), y.reshape(grid_size**2, -1)


def bilinear_interpolate_paddle_gridsample(image, samples_x, samples_y):
    C, H, W = image.shape
    image = image.unsqueeze(1)  # change to:  C x 1 x H x W        C,K,1,2   C,K,1,1

    samples_x = samples_x.unsqueeze(2)
    samples_x = samples_x.unsqueeze(3)  # 49,K,1,1
    samples_y = samples_y.unsqueeze(2)
    samples_y = samples_y.unsqueeze(3)

    samples = paddle.concat([samples_x, samples_y], 3)
    samples[:, :, :, 0] = (samples[:, :, :, 0] / W)  # normalize to between  0 and 1

    samples[:, :, :, 1] = (samples[:, :, :, 1] / H)  # normalize to between  0 and 1
    samples = samples * 2 - 1  # normalize to between -1 and 1  # 49,K,1,2

    #B,C,H,W
    #B,H,W,2
    #B,C,H,W

    return nn.functional.grid_sample(image, samples, align_corners=False)
@manager.HEADS.add_component
class TEDSHead(RoIHeadBase):
    """ def __init__(self,model_cfg,input_channels={'x_conv1': 16, 'x_conv2': 32, 'x_conv3': 64, 'x_conv4': 64},point_cloud_range=[0, -40, -3,70.4 , 40, 1 ]
,voxel_size=[0.05,0.05,0.05],num_class=1,
                 **kwargs):
        
        super().__init__(num_class=num_class,model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg=model_cfg['roi_grid_pool']
        LAYER_cfg=self.pool_cfg['pool_layers']
        self.point_cloud_range=point_cloud_range
        self.voxel_size=voxel_size
        self.rot_num=1
        self.x_trans_train=X_TRANS()
        c_out=0
        self.roi_grid_pool_layers=nn.LayerList()
        for src_name in self.pool_cfg['features_source']:
            mlps = LAYER_cfg[src_name]['mlps']
            for k in range(len(mlps)):
                mlps[k] = [input_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name]['query_ranges'],
                nsamples=LAYER_cfg[src_name]['nsample'],
                radii=LAYER_cfg[src_name]['pool_radius'],
                mlps=mlps,
                pool_method=LAYER_cfg[src_name]['pool_method'],
            )

            self.roi_grid_pool_layers.append(pool_layer)

            c_out += sum([x[-1] for x in mlps])

        
        GRID_SIZE = self.model_cfg['roi_grid_pool']['grid_size']
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out
        shared_fc_list = []
        self.training=self.model_cfg['training']
        for k in range(0, self.model_cfg['shared_fc'].__len__()):
            if self.training!='Flase':
                shared_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg['shared_fc'][k], bias_attr=False),
                    nn.BatchNorm1D(self.model_cfg['shared_fc'][k],momentum=0.1),
                    nn.ReLU()
                ])
            else:
                shared_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg['shared_fc'][k], bias_attr=False),
                    nn.BatchNorm1D(self.model_cfg['shared_fc'][k],momentum=0.1,use_global_stats=True),
                    nn.ReLU()
                ])
            pre_channel = self.model_cfg['shared_fc'][k]

            if k != self.model_cfg['shared_fc'].__len__() - 1 and self.model_cfg['dp_ratio'] > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg['dp_ratio']))
        self.shared_fc_layers=nn.Sequential(*shared_fc_list)


        self.shared_channel = pre_channel

        pre_channel = self.model_cfg['shared_fc'][-1] * 2
        cls_fc_list = []
        for k in range(0, self.model_cfg['cls_fc'].__len__()):
            if self.training:
                cls_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg['cls_fc'][k], bias_attr=False),
                    nn.BatchNorm1D(self.model_cfg['cls_fc'][k],momentum=0.1),
                    nn.ReLU()
                ])
            else:
                cls_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg['cls_fc'][k], bias_attr=False),
                    nn.BatchNorm1D(self.model_cfg['cls_fc'][k],momentum=0.1,use_global_stats=True),
                    nn.ReLU()
                ])
            pre_channel = self.model_cfg['cls_fc'][k]

            if k != self.model_cfg['cls_fc'].__len__() - 1 and self.model_cfg['dp_ratio'] > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg['dp_ratio']))

        cls_fc_list.append(nn.Linear(pre_channel, self.num_class, bias_attr=True))
        self.cls_layers=nn.Sequential(*cls_fc_list)

        pre_channel = self.model_cfg['shared_fc'][-1] * 2
        reg_fc_list = []
        for k in range(0, self.model_cfg['reg_fc'].__len__()):
            if self.training:
                reg_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg['reg_fc'][k], bias_attr=False),
                    nn.BatchNorm1D(self.model_cfg['reg_fc'][k],momentum=0.1),
                    nn.ReLU()
                ])
            else:
                reg_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg['reg_fc'][k], bias_attr=False),
                    nn.BatchNorm1D(self.model_cfg['reg_fc'][k],momentum=0.1,use_global_stats=True),
                    nn.ReLU()
                ])
            pre_channel = self.model_cfg['reg_fc'][k]

            if k != self.model_cfg['reg_fc'].__len__() - 1 and self.model_cfg['dp_ratio'] > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg['dp_ratio']))

        reg_fc_list.append(nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias_attr=True))
        reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_layers=reg_fc_layers

        
        self.cross_attention_layers = Attention_Layer(self.shared_channel)


        self.init_weights()
        self.forward_ret_dict = {}
        self.ious = {0: [], 1: [], 2: [], 3: []}"""
 
    def __init__(self,model_cfg,input_channels={'x_conv1': 16, 'x_conv2': 32, 'x_conv3': 64, 'x_conv4': 64},point_cloud_range=[0, -40, -3,70.4 , 40, 1 ]
,voxel_size=[0.05,0.05,0.05],num_class=1,
                 **kwargs):
        
        super().__init__(num_class=num_class,model_cfg=model_cfg)
        self.model_cfg = model_cfg
        self.pool_cfg=model_cfg['roi_grid_pool']
        LAYER_cfg=self.pool_cfg['pool_layers']
        self.point_cloud_range=point_cloud_range
        self.voxel_size=voxel_size
        self.rot_num=1
        self.x_trans_train=X_TRANS()
        c_out=0
        self.roi_grid_pool_layers=nn.LayerList()
        for src_name in self.pool_cfg['features_source']:
            mlps = LAYER_cfg[src_name]['mlps']
            for k in range(len(mlps)):
                mlps[k] = [input_channels[src_name]] + mlps[k]
            pool_layer = voxelpool_stack_modules.NeighborVoxelSAModuleMSG(
                query_ranges=LAYER_cfg[src_name]['query_ranges'],
                nsamples=LAYER_cfg[src_name]['nsample'],
                radii=LAYER_cfg[src_name]['pool_radius'],
                mlps=mlps,
                pool_method=LAYER_cfg[src_name]['pool_method'],
            )

            self.roi_grid_pool_layers.append(pool_layer)

            c_out += sum([x[-1] for x in mlps])

        
        GRID_SIZE = self.model_cfg['roi_grid_pool']['grid_size']
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out
        shared_fc_list = []
        
        for k in range(0, self.model_cfg['shared_fc'].__len__()):
            
            shared_fc_list.extend([
                    nn.Linear(pre_channel, self.model_cfg['shared_fc'][k], bias_attr=False),
                    nn.BatchNorm1D(self.model_cfg['shared_fc'][k]),
                    nn.ReLU()
                ])
            
            pre_channel = self.model_cfg['shared_fc'][k]

            if k != self.model_cfg['shared_fc'].__len__() - 1 and self.model_cfg['dp_ratio'] > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg['dp_ratio']))
        self.shared_fc_layers=nn.Sequential(*shared_fc_list)


        self.shared_channel = pre_channel

        pre_channel = self.model_cfg['shared_fc'][-1] * 2
        cls_fc_list = []
        for k in range(0, self.model_cfg['cls_fc'].__len__()):
            
            cls_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg['cls_fc'][k], bias_attr=False),
                nn.BatchNorm1D(self.model_cfg['cls_fc'][k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg['cls_fc'][k]

            if k != self.model_cfg['cls_fc'].__len__() - 1 and self.model_cfg['dp_ratio'] > 0:
                cls_fc_list.append(nn.Dropout(self.model_cfg['dp_ratio']))

        cls_fc_list.append(nn.Linear(pre_channel, self.num_class, bias_attr=True))
        self.cls_layers=nn.Sequential(*cls_fc_list)

        pre_channel = self.model_cfg['shared_fc'][-1] * 2
        reg_fc_list = []
        for k in range(0, self.model_cfg['reg_fc'].__len__()):
            
            reg_fc_list.extend([
                nn.Linear(pre_channel, self.model_cfg['reg_fc'][k], bias_attr=False),
                nn.BatchNorm1D(self.model_cfg['reg_fc'][k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg['reg_fc'][k]

            if k != self.model_cfg['reg_fc'].__len__() - 1 and self.model_cfg['dp_ratio'] > 0:
                reg_fc_list.append(nn.Dropout(self.model_cfg['dp_ratio']))

        reg_fc_list.append(nn.Linear(pre_channel, self.box_coder.code_size * self.num_class, bias_attr=True))
        reg_fc_layers = nn.Sequential(*reg_fc_list)
        self.reg_layers=reg_fc_layers

        
        self.cross_attention_layers = Attention_Layer(self.shared_channel)


        self.init_weights()
        self.forward_ret_dict = {}
        self.ious = {0: [], 1: [], 2: [], 3: []}
        
    def init_weights(self, weight_init='xavier'):
        if weight_init not in ['kaiming', 'xavier', 'normal']:
            raise NotImplementedError
        
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D) or isinstance(m, nn.Conv1D):
                if weight_init == 'normal':
                    m.weight.set_value(
                        paddle.normal(mean=0, std=0.001, shape=m.weight.shape))
                elif weight_init == 'kaiming':
                    kaiming_normal_init(
                        m.weight, reverse=isinstance(m, nn.Linear))
                elif weight_init == 'xavier':
                    xavier_normal_init(
                        m.weight, reverse=isinstance(m, nn.Linear))

                if m.bias is not None:
                    constant_init(m.bias, value=0)
            elif isinstance(m, nn.BatchNorm1D):
                constant_init(m.weight, value=1)
                constant_init(m.bias, value=0)
        self.reg_layers[-1].weight.set_value(
            paddle.normal(
                mean=0, std=0.001, shape=self.reg_layers[-1].weight.shape))
    def roi_grid_pool(self, batch_dict, i):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """

        if i==0:
            rot_num_id = ''
        else:
            rot_num_id = str(i)

        rois = batch_dict['rois'].clone()
        
        batch_size = batch_dict['batch_size']
        with_vf_transform = batch_dict.get('with_voxel_feature_transform', False)
        start_time=time.time()
        roi_grid_xyz, _ = self.get_global_grid_points_of_roi(
            rois, grid_size=self.pool_cfg["grid_size"]
        )  # (BxN, 6x6x6, 3)
        
        roi_grid_xyz = paddle.reshape(roi_grid_xyz, shape=[batch_size, -1, 3])
        # compute the voxel coordinates of grid points
        roi_grid_coords_x = paddle.floor((roi_grid_xyz[:, :, 0:1] - self.point_cloud_range[0]) / self.voxel_size[0])
        roi_grid_coords_y = paddle.floor((roi_grid_xyz[:, :, 1:2] - self.point_cloud_range[1]) / self.voxel_size[1])
        roi_grid_coords_z = paddle.floor((roi_grid_xyz[:, :, 2:3] - self.point_cloud_range[2]) / self.voxel_size[2])
        
        # roi_grid_coords: (B, Nx6x6x6, 3)
        roi_grid_coords = paddle.concat([roi_grid_coords_x, roi_grid_coords_y, roi_grid_coords_z], axis=-1)
        batch_idx=paddle.zeros([batch_size, roi_grid_coords.shape[1], 1])
       
        for bs_idx in range(batch_size):
            batch_idx[bs_idx, :, 0] = bs_idx
       
        
        if self.in_export_mode==True:
            roi_grid_batch_cnt = paddle.full([batch_size], roi_grid_coords.shape[1], dtype='int32')
        else:    
            roi_grid_batch_cnt = paddle.zeros([batch_size],dtype='int32')
            roi_grid_batch_cnt.fill_(roi_grid_coords.shape[1])
        
        pooled_features_list = []
        
        for k, src_name in enumerate(self.pool_cfg['features_source']):
            
            pool_layer = self.roi_grid_pool_layers[k]
           
            if src_name in ['x_conv1', 'x_conv2', 'x_conv3', 'x_conv4']:

                cur_stride = batch_dict['multi_scale_3d_strides'][src_name]

                j=i
                while 'multi_scale_3d_features'+rot_num_id not in batch_dict:
                    j-=1
                    rot_num_id = str(j)

                cur_sp_tensors = batch_dict['multi_scale_3d_features'+rot_num_id][src_name]
                
                if with_vf_transform:
                    cur_sp_tensors = batch_dict['multi_scale_3d_features_post'][src_name]
                else:
                    cur_sp_tensors = batch_dict['multi_scale_3d_features'+rot_num_id][src_name]
                
                # compute voxel center xyz and batch_cnt
                cur_coords = cur_sp_tensors.indices()
                #cur_coords=batch_dict['multi_scale_3d_features_indices'+rot_num_id][src_name]
                cur_coords = cur_coords.transpose([1, 0])
                
                start_time=time.time()
                cur_voxel_xyz = common_utils.get_voxel_centers(
                    cur_coords[:, 1:4],
                    downsample_times=cur_stride,
                    voxel_size=self.voxel_size,
                    point_cloud_range=self.point_cloud_range
                )  #
                
                cur_voxel_xyz_batch_cnt =paddle.zeros([batch_size],dtype='int32')
                #cur_voxel_xyz_batch_cnt = cur_voxel_xyz.new_zeros(batch_size).int()
                

                """ for bs_idx in range(batch_size):
                    bs_idx_tensor = paddle.to_tensor(bs_idx, dtype='int32')
                    cur_voxel_xyz_batch_cnt_list.append(paddle.sum(cur_coords[:, 0] == bs_idx_tensor).numpy()[0])

                cur_voxel_xyz_batch_cnt = paddle.to_tensor(cur_voxel_xyz_batch_cnt_list, dtype='int32') """
                for bs_idx in range(batch_size):
                    cur_voxel_xyz_batch_cnt[bs_idx] = int((cur_coords[:, 0] == bs_idx).sum().item())
                                
               
               
                
               
                v2p_ind_tensor = generate_voxel2pinds(cur_sp_tensors.shape,
                                                  cur_coords)
               
                
                v2p_ind_tensor = v2p_ind_tensor.astype('int32')
                
                # compute the grid coordinates in this scale, in [batch_idx, x y z] order
                cur_roi_grid_coords = paddle.floor(roi_grid_coords / cur_stride)
                
                cur_roi_grid_coords = paddle.concat([batch_idx, cur_roi_grid_coords], axis=-1)
                cur_roi_grid_coords =paddle.to_tensor( cur_roi_grid_coords,dtype='int32')
                # voxel neighbor aggregation
                #question
                
                pooled_features = pool_layer(
                    xyz=cur_voxel_xyz,
                    xyz_batch_cnt=cur_voxel_xyz_batch_cnt,
                    new_xyz=paddle.reshape(roi_grid_xyz,[-1,3]),
                    new_xyz_batch_cnt=roi_grid_batch_cnt,
                    new_coords=paddle.reshape(cur_roi_grid_coords,[-1, 4]),
                    features=cur_sp_tensors.values(),
                    voxel2point_indices=v2p_ind_tensor
                )
                
               
                pooled_features=paddle.reshape(pooled_features,shape=[ -1, self.pool_cfg['grid_size'] ** 3,pooled_features.shape[-1]])      # (BxN, 6x6x6, C)
                pooled_features_list.append(pooled_features)
       
        ms_pooled_features = paddle.concat(pooled_features_list, axis=-1)
        
        return ms_pooled_features
    def get_global_grid_points_of_roi(self, rois, grid_size):
        start_time=time.time()
        rois=paddle.reshape(rois,[-1, rois.shape[-1]])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = box_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(axis=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(axis=1)
        #print("get_global_grid time",time.time()-start_time)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features=paddle.ones(shape=[grid_size, grid_size, grid_size])
       
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        
        
        dense_idx=paddle.tile(dense_idx,repeat_times=[batch_size_rcnn, 1, 1])
       
        local_roi_size=paddle.reshape(rois,[batch_size_rcnn, -1])[:, 3:6]
      
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(axis=1) \
                          - (local_roi_size.unsqueeze(axis=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    
    def roi_x_trans(self, rois, rot_num_i, transform_param):

        batch_size = len(rois)
        rois = rois.clone()

        x_transformed_roi = []


        for bt_i in range(batch_size):

            cur_roi = rois[bt_i]
            bt_transform_param = transform_param[bt_i]
            previous_trans_param = bt_transform_param[rot_num_i-1]
            current_trans_param = bt_transform_param[rot_num_i]

            transed_roi = self.x_trans_train.backward_with_param({'boxes': cur_roi,
                                                                  'transform_param': previous_trans_param})
            transed_roi = self.x_trans_train.forward_with_param({'boxes': transed_roi['boxes'],
                                                                  'transform_param': current_trans_param})

            x_transformed_roi.append(transed_roi['boxes'])

        return paddle.stack(x_transformed_roi)

    def pred_x_trans(self, preds, rot_num_i, transform_param):

        batch_size = len(preds)
        preds = preds.clone()

        x_transformed_roi = []
        if self.in_export_mode:
            return preds
        else:
            for bt_i in range(batch_size):

                cur_roi = preds[bt_i]
                bt_transform_param = transform_param[bt_i]
                current_trans_param = bt_transform_param[rot_num_i]

                transed_roi = self.x_trans_train.backward_with_param({'boxes': cur_roi,
                                                                    'transform_param': current_trans_param})

                x_transformed_roi.append(transed_roi['boxes'])

            return paddle.stack(x_transformed_roi)

    def multi_grid_pool_aggregation(self, batch_dict, targets_dict):

        all_preds = []
        all_scores = []

        all_shared_features = []
        t=[]
        for i in range(self.rot_num):
            rot_num_id = str(i)

            if i >= 1 and 'transform_param' in batch_dict:
                batch_dict['rois'] = self.roi_x_trans(batch_dict['rois'], i, batch_dict['transform_param'])

            if self.training:
                start_time=time.time()
                targets_dict = self.assign_targets_ted(batch_dict, i, enable_dif=True)
                #print("assign_targets_ted time:",time.time()-start_time)
                batch_dict['rois'] = targets_dict['rois']

                batch_dict['roi_labels'] = targets_dict['roi_labels']

            if 'transform_param' in batch_dict:
                start_time=time.time()
                pooled_features = self.roi_grid_pool(batch_dict, i)
                #print("roi_grid_pool time:",time.time()-start_time)
            else:
                start_time=time.time()
                pooled_features = self.roi_grid_pool(batch_dict, 0)
                #print("roi_grid_pool time:",time.time()-start_time)
            
            
            pooled_features=paddle.reshape(pooled_features,[pooled_features.shape[0], -1])
        
            shared_features = self.shared_fc_layers(pooled_features)  ##不一样
            shared_features = shared_features.unsqueeze(0)  # 1,B,C  
            all_shared_features.append(shared_features)
            pre_feat = paddle.concat(all_shared_features, 0)
            
            attentive_cur_feat = self.cross_attention_layers(paddle.transpose(pre_feat,[1,0,2])).unsqueeze(0)
            
            attentive_cur_feat = paddle.concat([attentive_cur_feat, shared_features], -1)
            attentive_cur_feat = attentive_cur_feat.squeeze(0)  # B, C*2
            
            rcnn_cls = self.cls_layers(attentive_cur_feat)
            
            rcnn_reg = self.reg_layers(attentive_cur_feat)
        
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )

            if self.training:

                targets_dict['rcnn_cls'] = rcnn_cls
                targets_dict['rcnn_reg'] = rcnn_reg
                
                
                self.forward_ret_dict['targets_dict' + rot_num_id] = targets_dict
                #batch_dict['targets_dict' + rot_num_id]=self.forward_ret_dict['targets_dict' + rot_num_id]
            
            batch_dict['rois'] = batch_box_preds
            batch_dict['roi_scores'] = batch_cls_preds.squeeze(-1)
            
            outs = batch_box_preds.clone()
            
            if 'transform_param' in batch_dict:
                start_time=time.time()
                outs = self.pred_x_trans(outs, i, batch_dict['transform_param'])
                
            all_preds.append(outs)
            all_scores.append(batch_cls_preds)
            #print("all_preds",all_preds)
            #print("all_score",all_scores)
            ##exit()
        
        return paddle.mean(paddle.stack(all_preds), 0), paddle.mean(paddle.stack(all_scores), 0),batch_dict


    def forward(self, batch_dict):
        if 'gt_boxes' not in batch_dict:
            self.training=False
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            self.rot_num = trans_param.shape[1]
        

        
        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg['nms_config']['train' if self.training else 'test']
        )
        
       
                
        #return self.multi_grid_pool_aggregation(batch_dict, targets_dict)
        boxes,scores,batch_dict = self.multi_grid_pool_aggregation(batch_dict, targets_dict)
        
       
        if not self.training:
            batch_dict['batch_box_preds'] = boxes
            batch_dict['batch_cls_preds'] = scores

        return batch_dict
        