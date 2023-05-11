

import numpy as np
import paddle
from paddle import sparse
from paddle.sparse import nn

from paddle3d.apis import manager
from paddle3d.models.layers import param_init

__all__ = ['ToVoxelBackBone8x']


def sparse_conv_bn_relu(in_channels,
                        out_channels,
                        kernel_size,
                        stride=1,
                        padding=0,
                        conv_type='subm'):
    if conv_type == 'subm':
        conv = nn.SubmConv3D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=0,
            bias_attr=False)
    elif conv_type == 'spconv':
        conv = nn.Conv3D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias_attr=False)
    elif conv_type == 'inverseconv':
        raise NotImplementedError
    else:
        raise NotImplementedError

    m = paddle.nn.Sequential(
        conv,
        nn.BatchNorm(out_channels, epsilon=1e-3, momentum=1 - 0.01),
        nn.ReLU(),
    )

    return m


@manager.BACKBONES.add_component
class ToVoxelBackBone8x(paddle.nn.Layer):
    def __init__(self,
                 in_channels=4,
                 **kwargs):
        super(ToVoxelBackBone8x, self).__init__()
        self.return_num_features_as_dict=kwargs['return_num_features_as_dict']
        self.out_features=kwargs['out_featues']
        self.conv_input = paddle.nn.Sequential(
            nn.SubmConv3D(in_channels, 16, 3, padding=1, bias_attr=False),
            nn.BatchNorm(16, epsilon=1e-3, momentum=1 - 0.01), nn.ReLU())

        self.conv1 = paddle.nn.Sequential(
            sparse_conv_bn_relu(16, 16, 3, padding=1), )

        self.conv2 = paddle.nn.Sequential(
            sparse_conv_bn_relu(
                16, 32, 3, stride=2, padding=1, conv_type='spconv'),
            sparse_conv_bn_relu(32, 32, 3, padding=1),
            sparse_conv_bn_relu(32, 32, 3, padding=1))

        self.conv3 = paddle.nn.Sequential(
            sparse_conv_bn_relu(
                32, 64, 3, stride=2, padding=1, conv_type='spconv'),
            sparse_conv_bn_relu(64, 64, 3, padding=1),
            sparse_conv_bn_relu(64, 64, 3, padding=1))

        self.conv4 = paddle.nn.Sequential(
            sparse_conv_bn_relu(
                64, 64, 3, stride=2, padding=(0, 1, 1), conv_type='spconv'),
            sparse_conv_bn_relu(64, 64, 3, padding=1),
            sparse_conv_bn_relu(64, 64, 3, padding=1),
        )

        last_pad = 0
        self.extra_conv = paddle.nn.Sequential(
            nn.Conv3D(
                64,
                64, (3, 1, 1),
                stride=(2, 1, 1),
                padding=last_pad,
                bias_attr=False),  # [200, 150, 5] -> [200, 150, 2]
            nn.BatchNorm(64, epsilon=1e-3, momentum=1 - 0.01),
            nn.ReLU(),
        )
        
        point_cloud_range = np.array(kwargs['point_cloud_range'], dtype=np.float32)
        voxel_size = np.array(kwargs['voxel_size'], dtype=np.float32)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)

        self.sparse_shape = np.array(grid_size[::-1]) + [1, 0, 0]
        self.in_channels = in_channels

        self.num_point_features = 64
        if self.return_num_features_as_dict:
            num_point_features = {}
            num_point_features.update({
                'x_conv1': 16,
                'x_conv2': 32,
                'x_conv3': 64,
                'x_conv4': 64,
            })
            self.num_point_features = num_point_features
        
        self.init_weight()

    def init_weight(self):
        for layer in self.sublayers():
            if isinstance(layer, (nn.Conv3D, nn.SubmConv3D)):
                param_init.reset_parameters(layer)
            if isinstance(layer, nn.BatchNorm):
                param_init.constant_init(layer.weight, value=1)
                param_init.constant_init(layer.bias, value=0)
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
    def forward(self,batch_dict):
        if self.training:
            return self.forward_train(batch_dict)
        else:
            return self.forward_test(batch_dict)
    def forward_train(self,batch_dict):
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

            shape = [batch_dict['batch_size']] + list(self.sparse_shape) + [self.in_channels]
            sp_x = sparse.sparse_coo_tensor(
                voxel_coords.transpose((1, 0)),
                voxel_features,
                shape=shape,
                stop_gradient=False)

            x = self.conv_input(sp_x)

            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)

            out = self.extra_conv(x_conv4)

            #out = out.to_dense()
            """ out = paddle.transpose(out, perm=[0, 4, 1, 2, 3])
            N, C, D, H, W = out.shape
            out = paddle.reshape(out, shape=[N, C * D, H, W]) """

           
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
        return batch_dict
    def forward_test(self, batch_dict):
       
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            rot_num = trans_param.shape[1]
        else:
            rot_num = 1

        all_lidar_feat = []
        all_lidar_coords = []
        
        new_shape = [batch_dict['batch_size'],int(self.sparse_shape[0]), int(self.sparse_shape[1]), int(self.sparse_shape[2] * 4),self.in_channels]

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
        
        out = self.extra_conv(x_conv4)
       
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