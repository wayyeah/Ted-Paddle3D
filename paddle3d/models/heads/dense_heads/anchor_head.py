# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.nn.initializer import Constant, Normal
import time 
from paddle3d.apis import manager
from paddle3d.models.layers import reset_parameters
from paddle3d.models.common.model_nms_utils import class_agnostic_nms
from paddle3d.models.losses import (SigmoidFocalClassificationLoss,
                                    WeightedCrossEntropyLoss,
                                    WeightedSmoothL1Loss)
from paddle3d.utils.box_coder import ResidualCoder,PreviousResidualDecoder

from .target_assigner.anchor_generator import AnchorGenerator
from .target_assigner.axis_aligned_target_assigner import \
    AxisAlignedTargetAssigner

__all__ = ['AnchorHeadSingle']


@manager.HEADS.add_component
class AnchorHeadSingle(nn.Layer):
    def __init__(self, model_cfg, input_channels, class_names, voxel_size,
                 point_cloud_range, 
                 predict_boxes_when_training, anchor_generator_cfg,target_assigner_config,anchor_target_cfg,
                 num_dir_bins, loss_weights):
        super().__init__()
        point_cloud_range=[0, -40, -3, 70.4, 40, 1]
        self.model_cfg = model_cfg
        self.num_class = len(class_names)
        self.class_names = class_names
        self.predict_boxes_when_training = predict_boxes_when_training
        self.anchor_generator_cfg = anchor_generator_cfg
        self.num_dir_bins = num_dir_bins
        self.anchor_target_cfg = anchor_target_cfg
        self.loss_weights = loss_weights
        self.target_assigner_config=target_assigner_config
        if 'use_multihead' in self.model_cfg:
            self.use_multihead = self.model_cfg['use_multihead'] 
        else:
            self.use_multihead = False
        self.box_coder = ResidualCoder(num_dir_bins=num_dir_bins)
      
        point_cloud_range = np.asarray(point_cloud_range)
        grid_size = (point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        grid_size = np.round(grid_size).astype(np.int64)
        self.grid_size=grid_size
        self.voxel_size = (point_cloud_range[3] - point_cloud_range[0]) / grid_size[0]
        self.range=point_cloud_range
        anchors, self.num_anchors_per_location = self.generate_anchors(
            grid_size=grid_size,
            point_cloud_range=point_cloud_range,
            anchor_ndim=self.box_coder.code_size,anchor_generator_cfg=anchor_generator_cfg)
        
        self.anchors_root = [x for x in anchors]
        [self.register_buffer(name=f'buffer_{i}', tensor=x) for i, x in enumerate(anchors)]
        self.num_anchors_per_location = sum(self.num_anchors_per_location)
        
        
        self.conv_cls = nn.Conv2D(
            input_channels,
            self.num_anchors_per_location * self.num_class,
            kernel_size=1)

        self.conv_box = nn.Conv2D(
            input_channels,
            self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1,stride=1)
        self.target_assigner = AxisAlignedTargetAssigner(
            anchor_generator_cfg,
            anchor_target_cfg,
            class_names=self.class_names,
            box_coder=self.box_coder,match_height=anchor_target_cfg['match_height'])
        if self.model_cfg.get('use_direction_classifier', None) is not None:
            self.conv_dir_cls = nn.Conv2D(
                input_channels,
                self.num_anchors_per_location * self.model_cfg['num_dir_bins'],
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.forward_ret_dict = {}
        self.reg_loss_func = WeightedSmoothL1Loss(
            code_weights=loss_weights["code_weights"])
        self.cls_loss_func = SigmoidFocalClassificationLoss(
            alpha=0.25, gamma=2.0)
        self.dir_loss_func = WeightedCrossEntropyLoss()
        self.init_weight()

    def init_weight(self):

        for sublayer in self.sublayers():
            if isinstance(sublayer, nn.Conv2D):
                reset_parameters(sublayer)
        bias_shape = self.conv_cls.bias.shape
        temp_value = paddle.ones(bias_shape) * -paddle.log(
            paddle.to_tensor((1.0 - 0.01) / 0.01))
        self.conv_cls.bias.set_value(temp_value)
        weight_shape = self.conv_box.weight.shape
        self.conv_box.weight.set_value(
            paddle.normal(mean=0.0, std=0.001, shape=weight_shape))
    @staticmethod
    def generate_anchors( grid_size, point_cloud_range, anchor_ndim=7, anchor_generator_cfg=None):
        anchor_generator = AnchorGenerator(
            anchor_range=point_cloud_range,
            anchor_generator_config=anchor_generator_cfg)
        feature_map_size = [
            grid_size[:2] // config['feature_map_stride']
            for config in anchor_generator_cfg
        ]
        anchors_list, num_anchors_per_location_list = anchor_generator.generate_anchors(
            feature_map_size)

        if anchor_ndim != 7:
            for idx, anchors in enumerate(anchors_list):
                pad_zeros=paddle.zeros( [*anchors.shape[0:-1], anchor_ndim - 7])
                new_anchors = paddle.concat((anchors, pad_zeros), axis=-1)
                anchors_list[idx] = new_anchors

        return anchors_list, num_anchors_per_location_list

    def generate_predicted_boxes(self,
                                 batch_size,
                                 cls_preds,
                                 box_preds,
                                 dir_cls_preds=None):
        """
        Args:
            batch_size:
            cls_preds: (N, H, W, C1)
            box_preds: (N, H, W, C2)
            dir_cls_preds: (N, H, W, C3)

        Returns:
            batch_cls_preds: (B, num_boxes, num_classes)
            batch_box_preds: (B, num_boxes, 7+C)

        """
        # anchors = paddle.concat(self.anchors, axis=-3)
        if self.in_export_mode:
            anchors = self.anchors[0]

            
            num_anchors = paddle.shape(
                anchors.reshape([-1, paddle.shape(anchors)[-1]]))[0]
            batch_anchors = anchors.reshape([1, -1, paddle.shape(anchors)[-1]]).tile(
                [batch_size, 1, 1])
            batch_cls_preds = cls_preds.reshape([batch_size, num_anchors, -1]) \
                if not isinstance(cls_preds, list) else cls_preds
            batch_box_preds = box_preds.reshape([batch_size, num_anchors, -1]) if not isinstance(box_preds, list) \
                else paddle.concat(box_preds, axis=1).reshape([batch_size, num_anchors, -1])
            batch_box_preds = self.box_coder.decode_paddle(batch_box_preds,batch_anchors)

        else:

            if isinstance(self.anchors, list):
                if self.in_export_mode:
                    
                    anchors = self.anchors
                    
                else:
                    if self.use_multihead:
                        anchors = paddle.concat([anchor.transpose([3, 4, 0, 1, 2, 5]).reshape([-1, anchor.shape[-1]])
                                            for anchor in self.anchors], axis=0)
                    else:
                        anchors = paddle.concat(self.anchors, axis=-3)
            else:
                if len(self.anchors.shape) == 6:
                    anchors=self.anchors[0]
                else:
                    anchors = self.anchors
        
            num_anchors = anchors.reshape([-1,anchors.shape[-1]]).shape[0]
            batch_anchors = anchors.reshape([1, -1,anchors.shape[-1]]).tile([batch_size, 1, 1])
            
            batch_cls_preds = cls_preds.reshape([batch_size, num_anchors, -1]) \
                if not isinstance(cls_preds, list) else cls_preds
            batch_box_preds = box_preds.reshape([batch_size, num_anchors, -1]) if not isinstance(box_preds, list) \
                else paddle.concat(box_preds, axis=1).reshape([batch_size, num_anchors, -1])
            batch_box_preds = self.box_coder.decode_paddle(batch_box_preds,
                                                        batch_anchors)
        if dir_cls_preds is not None:
            dir_offset = self.model_cfg['dir_offset']
            dir_limit_offset = self.model_cfg['dir_limit_offset']
            dir_cls_preds = dir_cls_preds.reshape([batch_size, num_anchors, -1]) if not isinstance(dir_cls_preds, list) \
                else paddle.concat(dir_cls_preds, axis=1).reshape([batch_size, num_anchors, -1])
            dir_labels = paddle.argmax(dir_cls_preds, axis=-1)

            period = (2 * np.pi / self.num_dir_bins)
            dir_rot = self.limit_period(batch_box_preds[..., 6] - dir_offset,
                                        dir_limit_offset, period)
          
            batch_box_preds[
                ..., 6] = dir_rot + dir_offset + period * dir_labels.cast(
                    batch_box_preds.dtype)
            
        if isinstance(self.box_coder, PreviousResidualDecoder):
            batch_box_preds[..., 6] = self.limit_period(
                -(batch_box_preds[..., 6] + np.pi / 2), offset=0.5, period=np.pi * 2
            )
     

        return batch_cls_preds, batch_box_preds

    def limit_period(self, val, offset=0.5, period=np.pi):
        ans = val - paddle.floor(val / period + offset) * period
        return ans

    def get_anchor_mask(self, data_dict, shape):
        stride = np.round(self.voxel_size*8.*10.)
        minx=self.range[0]
        miny=self.range[1]
        points = data_dict["points"]
        mask = paddle.zeros([shape[-2],shape[-1]])
        mask_large = paddle.zeros([shape[-2]//10,shape[-1]//10])
        in_x = (points[:, 1] - minx) / stride
        in_y = (points[:, 2] - miny) / stride
        
        in_x=in_x.astype("int32")
        in_y=in_y.astype("int32")
        in_x =paddle.clip( in_x,max=shape[-1]//10-1)
        in_y = paddle.clip(in_y,max=shape[-2]//10-1)
        mask_large = mask_large.clone().detach().cpu().numpy()
        mask_large[in_y,in_x] = 1
        
        mask_large_index = np.argwhere( mask_large>0 )
        mask_large_index = mask_large_index*10
        index_list=[]
        for i in np.arange(-10, 10, 1):
            for j in np.arange(-10, 10, 1):
                index_list.append(mask_large_index+[i,j])
        index_list = np.concatenate(index_list,0)
        inds = paddle.to_tensor((index_list),dtype="int64")
        mask=mask.numpy()
        mask[inds[:,0],inds[:,1]]=1
        mask=paddle.to_tensor(mask).astype("bool")
        
        return mask

    def get_anchor_mask_export(self, data_dict, shape):
        stride = np.round(self.voxel_size*8.*10.)
        minx=self.range[0]
        miny=self.range[1]
        points = data_dict["points"]
        
        mask = paddle.zeros([shape[-2],shape[-1]])
        mask_large = paddle.zeros([shape[-2]//10,shape[-1]//10])
        in_x = (points[:, 1] - minx) / stride
        in_y = (points[:, 2] - miny) / stride

        in_x=in_x.astype("int32")
        in_y=in_y.astype("int32")
        in_x = paddle.clip(in_x,max=shape[-1]//10-1)
        in_y = paddle.clip(in_y,max=shape[-2]//10-1)

        indices_large = paddle.stack([in_y, in_x], axis=1)

        unique_indices, inverse_index, counts = paddle.unique(indices_large, return_inverse=True, return_counts=True, axis=0)

        # Create a tensor of ones with shape (num_indices,)
        updates = paddle.ones([unique_indices.shape[0]])

        mask_large = paddle.scatter_nd(unique_indices, updates, [shape[-2]//10,shape[-1]//10]).astype('bool')
        

        mask_large_index = paddle.nonzero(mask_large > 0)
        mask_large_index = mask_large_index * 10

        index_list=[]
        
        for i in paddle.arange(-10, 10, 1):
            for j in paddle.arange(-10, 10, 1):
                index_list.append(mask_large_index + paddle.to_tensor([i,j]).reshape([1,2]))
        index_list = paddle.concat(index_list, 0)
        #存在索引小于0
        negative_values_first_column = index_list[:, 0] < 0
        negative_values_second_column = index_list[:, 1] < 0
        large_values_first_column=index_list[:, 0]>=shape[-2]
        large_values_second_column=index_list[:, 1]>=shape[-1]
        positive_rows = paddle.where((negative_values_first_column == 0) & (negative_values_second_column == 0)&(large_values_first_column==0)&(large_values_second_column==0))[0]
        index_list=index_list[ positive_rows]
        unique_indices, inverse_index, counts = paddle.unique(index_list, return_inverse=True, return_counts=True, axis=0)
        unique_indices=unique_indices.reshape([-1,2])
        updates = paddle.ones([len(unique_indices)], dtype='int32')
        mask = paddle.scatter_nd(unique_indices,  updates, [shape[-2],shape[-1]]).astype('bool')
        
        return mask

    def forward(self, data_dict):
        
        
        batch_size=data_dict["batch_size"]
        if  self.in_export_mode:
            anchor_mask=self.get_anchor_mask_export(data_dict,data_dict['spatial_features_2d'].shape)
        else:
            
            anchor_mask = self.get_anchor_mask(data_dict,data_dict['spatial_features_2d'].shape)
            
            
        #print("anchor_mask",time.time()-start_time)
        new_anchors = []
        #print(self.anchors_root)
        
        if self.in_export_mode:
            
            
            
            anchors=getattr(self, 'buffer_0')
           
            
            anchors_t=anchors.reshape([anchors.shape[1]*anchors.shape[2],anchors.shape[3],anchors.shape[4],anchors.shape[5]])
            
            mask_t=anchor_mask.reshape([anchor_mask.shape[0]*anchor_mask.shape[1]])

            anchors_t=anchors_t[mask_t]
           
            anchors_t=anchors_t.reshape([anchors.shape[0],int((mask_t>0).sum()),anchors.shape[3],anchors.shape[4],anchors.shape[5]])
            new_anchors.append(anchors_t)
            #new_anchors=self.anchors_root


        else:
            for i in range (len(self.anchors_root)):
                anchors_numpy = self.anchors_root[i].numpy()
                anchors_numpy = anchors_numpy[:,anchor_mask.numpy(),...]
                new_anchors.append(paddle.to_tensor(anchors_numpy))
        
           
        self.anchors = new_anchors
       
        spatial_features_2d = data_dict['spatial_features_2d']
        #question
        
        cls_preds = self.conv_cls(spatial_features_2d)
        
        
        box_preds = self.conv_box(spatial_features_2d)
        
        
        
        cls_preds = cls_preds.transpose([0, 2, 3, 1])
        cls_preds=cls_preds.reshape([cls_preds.shape[0],cls_preds.shape[1]*cls_preds.shape[2],cls_preds.shape[3]])
        mask_t=anchor_mask.reshape([-1]) # H W # 1 HW
        cls_preds_temp=paddle.zeros([cls_preds.shape[0],cls_preds[0][mask_t].shape[0],cls_preds.shape[2]])
        for i in range(cls_preds_temp.shape[0]):
            cls_preds_temp[i]=cls_preds[i][mask_t]
        cls_preds=cls_preds_temp
       
      

        box_preds = box_preds.transpose([0, 2, 3, 1])
        box_preds = box_preds.reshape([box_preds.shape[0],box_preds.shape[1]*box_preds.shape[2],box_preds.shape[3]])
        box_preds_temp=paddle.zeros([box_preds.shape[0],box_preds[0][mask_t].shape[0],box_preds.shape[2]])
        for i in range(box_preds_temp.shape[0]):
            box_preds_temp[i]=box_preds[i][mask_t]
        box_preds=box_preds_temp

        self.forward_ret_dict['cls_preds'] = cls_preds  
        self.forward_ret_dict['box_preds'] = box_preds  #may question
        if self.conv_dir_cls is not None:
            
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds=dir_cls_preds.transpose([0, 2, 3, 1])
            dir_cls_preds=dir_cls_preds.reshape([dir_cls_preds.shape[0],dir_cls_preds.shape[1]*dir_cls_preds.shape[2],dir_cls_preds.shape[3]])
            dir_cls_preds_temp=paddle.zeros([dir_cls_preds.shape[0],dir_cls_preds[0][mask_t].shape[0],dir_cls_preds.shape[2]])
            for i in range(dir_cls_preds_temp.shape[0]):
                dir_cls_preds_temp[i]=dir_cls_preds[i][mask_t]
            dir_cls_preds=dir_cls_preds_temp

            self.forward_ret_dict['dir_cls_preds'] =dir_cls_preds
        else:
            dir_cls_preds=None
        if self.training:
            
            
            start_time=time.time()
            #targets_dict['box_cls_labels']
            targets_dict = self.target_assigner.assign_targets(
                self.anchors, data_dict['gt_boxes'])
            #rint("assign_targets",time.time()-start_time)
            self.forward_ret_dict.update(targets_dict)
           
            data_dict['gt_ious']=targets_dict['gt_ious']
            
        if not self.training or self.predict_boxes_when_training:
            if getattr(self, 'in_export_mode', False):
                batch_size = 1
            else:
                batch_size = data_dict['batch_size']
            start_time=time.time()
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_size,
                cls_preds=cls_preds,
                box_preds=box_preds,
                dir_cls_preds=dir_cls_preds)
            
           

            #print("generate_predicted_boxes",time.time()-start_time)
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False
        
        if 'nms_config' in self.model_cfg:
            start_time  = time.time()
            data_dict=self.proposal_layer(
                data_dict, nms_config=self.model_cfg['nms_config']['train' if self.training else 'test']
            )
            #print("proposal_layer",time.time()-start_time)

        return data_dict

    def get_loss(self):
        cls_loss, tb_dict = self.get_cls_layer_loss()
        box_loss, tb_dict_box = self.get_box_reg_layer_loss()
        tb_dict.update(tb_dict_box)
        #print("cls_loss",cls_loss.item(),"box_loss",box_loss.item())
        rpn_loss = cls_loss + box_loss

        tb_dict['rpn_loss'] = rpn_loss.item()
        return rpn_loss, tb_dict

    def get_cls_layer_loss(self):
        cls_preds = self.forward_ret_dict['cls_preds']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(cls_preds.shape[0])
        cared = box_cls_labels >= 0  # [N, num_anchors]
        positives = box_cls_labels > 0
        negatives = box_cls_labels == 0
        negative_cls_weights = negatives * 1.0
        cls_weights = (negative_cls_weights + 1.0 * positives)
        reg_weights = positives.astype("float32")
        if self.num_class == 1:
            # class agnostic
            
            
            box_cls_labels[positives] = 1   

        pos_normalizer = positives.sum(1, keepdim=True).astype("float32")
        
        
        reg_weights /= paddle.clip(pos_normalizer.astype('float32'), min=1.0)
        
        cls_weights /= paddle.clip(pos_normalizer, min=1.0)
        cls_targets = box_cls_labels * cared.cast(box_cls_labels.dtype)
        
        one_hot_targets = []
        for b in range(batch_size):
            one_hot_targets.append(
                F.one_hot(cls_targets[b], num_classes=self.num_class + 1))
        one_hot_targets = paddle.stack(one_hot_targets)
        cls_preds = cls_preds.reshape([batch_size, -1, self.num_class])
        one_hot_targets = one_hot_targets[..., 1:]
        
        cls_loss_src = self.cls_loss_func(
            cls_preds, one_hot_targets, weights=cls_weights)  # [N, M]
        cls_loss = cls_loss_src.sum() / batch_size
        cls_loss = cls_loss * self.loss_weights['cls_weight']
        tb_dict = {'rpn_loss_cls': cls_loss.item()}

        return cls_loss, tb_dict

    def get_box_reg_layer_loss(self):
        box_preds = self.forward_ret_dict['box_preds']
        box_dir_cls_preds = self.forward_ret_dict.get('dir_cls_preds', None)
        box_reg_targets = self.forward_ret_dict['box_reg_targets']
        box_cls_labels = self.forward_ret_dict['box_cls_labels']
        batch_size = int(box_preds.shape[0])

        positives = box_cls_labels > 0
        reg_weights = positives.cast("float32")
        pos_normalizer = positives.sum(1, keepdim=True)
        reg_weights /= paddle.clip(pos_normalizer, min=1.0)

        if isinstance(self.anchors, list):
            if self.use_multihead:
                anchors = paddle.concat(
                    [anchor.transpose([3, 4, 0, 1, 2, 5]).reshape([-1, anchor.shape[-1]]) for anchor in
                     self.anchors], axis=0)
            else:
                anchors = paddle.concat(self.anchors, axis=-3)
        else:
            anchors = self.anchors
        
        anchors = anchors.reshape([1, -1,
                                   anchors.shape[-1]]).tile([batch_size, 1, 1])
        box_preds = box_preds.reshape([
            batch_size, -1, box_preds.shape[-1] // self.num_anchors_per_location
        ])
        box_preds_sin, reg_targets_sin = self.add_sin_difference(
            box_preds, box_reg_targets)
        loc_loss_src = self.reg_loss_func(box_preds_sin, reg_targets_sin, \
                                        weights=reg_weights)  # [N, M]

        loc_loss = loc_loss_src.sum() / batch_size

        loc_loss = loc_loss * self.loss_weights['loc_weight']
        box_loss = loc_loss
        tb_dict = {'rpn_loss_loc': loc_loss.item()}

        if box_dir_cls_preds is not None:
            dir_targets = self.get_direction_target(
                anchors,
                box_reg_targets,
                dir_offset=self.model_cfg['dir_offset'],
                num_bins=self.num_dir_bins)

            dir_logits = box_dir_cls_preds.reshape(
                [batch_size, -1, self.num_dir_bins])
            weights = positives.cast("float32")
            weights /= paddle.clip(weights.sum(-1, keepdim=True), min=1.0)
            
            dir_loss = self.dir_loss_func(
                dir_logits, dir_targets, weights=weights)
            dir_loss = dir_loss.sum() / batch_size
            dir_loss = dir_loss * self.loss_weights['dir_weight']
            box_loss += dir_loss
            tb_dict['rpn_loss_dir'] = dir_loss.item()
        return box_loss, tb_dict

    def add_sin_difference(self, boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = paddle.sin(boxes1[..., dim:dim + 1]) * paddle.cos(
            boxes2[..., dim:dim + 1])
        rad_tg_encoding = paddle.cos(boxes1[..., dim:dim + 1]) * paddle.sin(
            boxes2[..., dim:dim + 1])
        boxes1 = paddle.concat(
            [boxes1[..., :dim], rad_pred_encoding, boxes1[..., dim + 1:]],
            axis=-1)
        boxes2 = paddle.concat(
            [boxes2[..., :dim], rad_tg_encoding, boxes2[..., dim + 1:]],
            axis=-1)
        return boxes1, boxes2

    def get_direction_target(self,
                             anchors,
                             reg_targets,
                             one_hot=True,
                             dir_offset=0,
                             num_bins=2):
        batch_size = reg_targets.shape[0]

        anchors = anchors.reshape([batch_size, -1, anchors.shape[-1]])
        rot_gt = reg_targets[..., 6] + anchors[..., 6]
        offset_rot = self.limit_period(rot_gt - dir_offset, 0, 2 * np.pi)
        dir_cls_targets = paddle.floor(
            offset_rot / (2 * np.pi / num_bins)).cast("int64")
        dir_cls_targets = paddle.clip(dir_cls_targets, min=0, max=num_bins - 1)

        if one_hot:
            dir_targets = []
            for b in range(batch_size):
                dir_targets.append(
                    F.one_hot(dir_cls_targets[b], num_classes=num_bins))
            dir_cls_targets = paddle.stack(dir_targets)
        return dir_cls_targets
    def proposal_layer(self, batch_dict, nms_config):
        if batch_dict.get('rois', None) is not None:
            batch_dict['cls_preds_normalized'] = False
            return batch_dict

        batch_size = batch_dict['batch_size']
        batch_box_preds = batch_dict['batch_box_preds']
        batch_cls_preds = batch_dict['batch_cls_preds']
        rois = paddle.zeros((batch_size, nms_config["nms_post_maxsize"],
                             batch_box_preds.shape[-1]))
        roi_scores = paddle.zeros((batch_size, nms_config["nms_post_maxsize"]))
        roi_labels = paddle.zeros((batch_size, nms_config["nms_post_maxsize"]),
                                  dtype='int64')

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]

            cur_roi_scores = paddle.max(cls_preds, axis=1)
            cur_roi_labels = paddle.argmax(cls_preds, axis=1)

            if nms_config['multi_class_nms']:
                raise NotImplementedError
            else:
                selected_score, selected_label, selected_box = class_agnostic_nms(
                    box_scores=cur_roi_scores,
                    box_preds=box_preds,
                    label_preds=cur_roi_labels,
                    nms_config=nms_config)

            rois[index, :selected_label.shape[0], :] = selected_box
            roi_scores[index, :selected_label.shape[0]] = selected_score
            roi_labels[index, :selected_label.shape[0]] = selected_label
        
        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels + 1
        batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 else False
        batch_dict.pop('batch_index', None)
        return batch_dict