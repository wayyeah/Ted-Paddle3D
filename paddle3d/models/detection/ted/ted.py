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
import pickle
import collections
import os
from typing import Dict, List
import time
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.static import InputSpec
from paddle3d.models.heads.roi_heads.target_assigner import iou3d_nms_utils
from paddle3d.apis import manager
from paddle3d.geometries import BBoxes3D
from paddle3d.models.common.model_nms_utils import class_agnostic_nms,compute_WBF
from paddle3d.sample import Sample, SampleMeta
from paddle3d.utils.logger import logger
from paddle3d.models.layers.param_init import uniform_init

@manager.MODELS.add_component
class Ted(nn.Layer):
    def __init__(self, num_class, voxelizer, voxel_encoder, 
                 backbone, neck,neck1, dense_head, roi_head, post_process_cfg):
        
        
        super(Ted, self).__init__()
        self.num_class = num_class
        self.voxelizer = voxelizer
        self.voxel_encoder = voxel_encoder
        #self.middle_encoder = middle_encoder
        self.backbone = backbone
        self.neck = neck
        self.neck1=neck1
        self.dense_head = dense_head
        self.roi_head = roi_head
        self.post_process_cfg = post_process_cfg
        self.init_weights()
    @staticmethod
    def generate_recall_record(box_preds, recall_dict, batch_index, data_dict=None, thresh_list=None):
        if 'gt_boxes' not in data_dict:
            
            return recall_dict

        rois = data_dict['rois'][batch_index] if 'rois' in data_dict else None
        gt_boxes = data_dict['gt_boxes'][batch_index]

        if recall_dict.__len__() == 0:
            recall_dict = {'gt': 0}
            for cur_thresh in thresh_list:
                recall_dict['roi_%s' % (str(cur_thresh))] = 0
                recall_dict['rcnn_%s' % (str(cur_thresh))] = 0

        cur_gt = gt_boxes
        k = cur_gt.__len__() - 1
        while k > 0 and cur_gt[k].sum() == 0:
            k -= 1
        cur_gt = cur_gt[:k + 1]

        if cur_gt.shape[0] > 0:
            if box_preds.shape[0] > 0:
                iou3d_rcnn = iou3d_nms_utils.boxes_iou3d_gpu(box_preds[:, 0:7], cur_gt[:, 0:7])
            else:
                iou3d_rcnn = paddle.zeros_like([0,cur_gt.shape[0]])
                

            if rois is not None:
                iou3d_roi = iou3d_nms_utils.boxes_iou3d_gpu(rois[:, 0:7], cur_gt[:, 0:7])

            for cur_thresh in thresh_list:
                if iou3d_rcnn.shape[0] == 0:
                    recall_dict['rcnn_%s' % str(cur_thresh)] += 0
                else:
                    rcnn_recalled = (iou3d_rcnn.max(axis=0)[0] > cur_thresh).sum().item()
                    recall_dict['rcnn_%s' % str(cur_thresh)] += rcnn_recalled
                if rois is not None:
                    roi_recalled = (iou3d_roi.max(axis=0)[0] > cur_thresh).sum().item()
                    recall_dict['roi_%s' % str(cur_thresh)] += roi_recalled

            recall_dict['gt'] += cur_gt.shape[0]
        else:
            gt_iou = paddle.zeros([box_preds.shape[0]])
          
        
        print(recall_dict)
        return recall_dict

    def init_weights(self):
        
        self.in_export_mode = False
        self.voxel_encoder.in_export_mode = False
        self.voxelizer.in_export_mode = False
        self.backbone.in_export_mode = False
        self.neck.in_export_mode = False
        self.neck1.in_export_mode = False
        self.dense_head.in_export_mode = False
        self.roi_head.in_export_mode = False
        need_uniform_init_bn_weight_modules = [
                 self.backbone, self.neck,self.neck1,
                self.roi_head.shared_fc_layers, self.roi_head.cls_layers,
                    self.roi_head.reg_layers]

        for module in need_uniform_init_bn_weight_modules:
            for layer in module.sublayers():
                if 'BatchNorm' in layer.__class__.__name__:
                    uniform_init(layer.weight, 0, 1)

    def voxelize(self, points):
       
        voxels, coordinates, num_points_in_voxel = self.voxelizer(points)
        return voxels, coordinates, num_points_in_voxel

    def forward(self, batch_dict, **kwargs):
        if 'gt_boxes' not in batch_dict:
            self.training=False
        
        #batch_dict['points']=[paddle.to_tensor(np.fromfile("/home/yw/Paddle3D/output/points.npy", dtype=np.float32).reshape(-1, 4)[8:,:])]
        #batch_dict['points1']=[paddle.to_tensor(np.fromfile("/home/yw/Paddle3D/output/points1.npy", dtype=np.float32).reshape(-1, 4)[8:,:])]
        #batch_dict['points2']=[paddle.to_tensor(np.fromfile("/home/yw/Paddle3D/output/points2.npy", dtype=np.float32).reshape(-1, 4)[8:,:])]
        #batch_dict['gt_boxes']=paddle.to_tensor([np.fromfile("/home/yw/Paddle3D/output/gt_boxes.npy", dtype=np.float32)[32:].reshape(-1, 7)])
        #batch_dict['gt_boxes1']=paddle.to_tensor([np.fromfile("/home/yw/Paddle3D/output/gt_boxes1.npy", dtype=np.float32).reshape(-1, 8)[4:,:]])
        #batch_dict['gt_boxes2']=paddle.to_tensor([np.fromfile("/home/yw/Paddle3D/output/gt_boxes2.npy", dtype=np.float32).reshape(-1, 8)[4:,:]])
       
        if not getattr(self, "in_export_mode", False):
            if 'transform_param' in batch_dict:
                batch_dict['transform_param']=paddle.to_tensor(batch_dict['transform_param'])
            
        if getattr(self, "in_export_mode", False):
            batch_dict['points']=batch_dict['data']
        if 'transform_param' in batch_dict:
            trans_param = batch_dict['transform_param']
            rot_num = trans_param.shape[1]
        else:
            rot_num = 1
        for i in range(rot_num):
            if i==0:
                frame_id = ''
            else:
                frame_id = str(i)
            """ if getattr(self, "in_export_mode", False) or not self.training:
                batch_dict['points']=batch_dict['data'] """
            
            voxel_features, coordinates, voxel_num_points = self.voxelizer(batch_dict['points'+frame_id])
            batch_dict['voxel_features'+frame_id] = voxel_features
            batch_dict['voxel_num_points'+frame_id] = voxel_num_points
            batch_dict["voxel_coords"+frame_id] = coordinates
         
       
        if not getattr(self, "in_export_mode", False):
            for i in range(rot_num):
                if i==0:
                    frame_id = ''
                else:
                    frame_id = str(i)
                points_pad = []
                for bs_idx, point in enumerate(batch_dict['points'+frame_id]):
                    point_dim = point.shape[-1]
                    point = point.reshape([1, -1, point_dim])
                    point_pad = F.pad(
                        point, [1, 0],
                        value=bs_idx,
                        mode='constant',
                        data_format="NCL")
                    point_pad = point_pad.reshape([-1, point_dim + 1])
                    points_pad.append(point_pad)
                batch_dict['points'+frame_id] = paddle.concat(points_pad, axis=0)
        else:
            point = batch_dict['points']
            batch_dict['batch_size'] = 1
            point = point.unsqueeze(1)
            point_pad = F.pad(
                point, [1, 0], value=0, mode='constant', data_format="NCL")
            
            batch_dict['points'] = point_pad.squeeze(1)
            #print(batch_dict['points'])
           
        file_path = '/home/yw/batch_dict.pkl'
        #batch_dict.pop('calibs', None)
        #save_batch_dict(batch_dict, file_path) 
        start_time1=time.time()
        """ np.save("/home/yw/points.npy",batch_dict['points'][:,1:])
        np.save("/home/yw/gt_boxes.npy",batch_dict['gt_boxes'][0][:,:-1])
        np.save("/home/yw/points1.npy",batch_dict['points1'][:,1:])
        np.save("/home/yw/gt_boxes1.npy",batch_dict['gt_boxes1'][0][:,:-1])
        np.save("/home/yw/points2.npy",batch_dict['points2'][:,1:])
        np.save("/home/yw/gt_boxes2.npy",batch_dict['gt_boxes2'][0][:,:-1]) """
        batch_dict=self.voxel_encoder(batch_dict)
        #print("voxel_encoder time:",time.time()-start_time)
        start_time=time.time()
        
        batch_dict = self.backbone(batch_dict) #question  
        #print("backbone time:",time.time()-start_time)
        start_time=time.time()
        
        batch_dict = self.neck(batch_dict)
        #print("neck time:",time.time()-start_time)
       
        start_time=time.time()
        batch_dict = self.neck1(batch_dict)
        #print("neck1 time:",time.time()-start_time) 
        
        start_time=time.time()
        
        batch_dict = self.dense_head(batch_dict)
        #print("dense_head time:",time.time()-start_time)
        
        start_time=time.time()
        #print(self.roi_head(batch_dict))
        #return self.roi_head(batch_dict)
        batch_dict = self.roi_head(batch_dict)
       
        #print("roi_head time:",time.time()-start_time)
        #print("forward time",time.time()-start_time1)
        #print(batch_dict)
        if self.training:
            #start_time=time.time()
            loss = self.get_training_loss(batch_dict)
            pred_dicts = self.post_processing(batch_dict)
            #print(pred_dicts)
            """ np.save("/home/yw/points.npy",batch_dict['points'][:,1:])
            np.save("/home/yw/pred_boxes.npy",pred_dicts[0]['box3d_lidar'])
            np.save("/home/yw/gt_boxes.npy",batch_dict['gt_boxes'][0][:,:-1])
            np.save("/home/yw/pred_scores.npy",pred_dicts[0]['scores']) """
            #print(batch_dict['gt_boxes'])
            #print("forward time:",time.time()-start_time1)
                
            return loss
        else:
            
            
            #print(pred_dicts)
            """ np.save("/home/yw/points.npy",batch_dict['points'][:,1:])
            np.save("/home/yw/gt_boxes.npy",batch_dict['gt_boxes'][0][:,:-1]) """
            if not getattr(self, "in_export_mode", False):
                pred_dicts = self.post_processing(batch_dict)
                print(batch_dict['points'].sum())
                preds = self._parse_results_to_sample(pred_dicts, batch_dict)
                print(preds[0].bboxes_3d)
                return {'preds': preds}
            else:
                
                pred_dicts = self.post_processing(batch_dict)
                
                return pred_dicts
            
        
    def collate_fn(self, batch: List):
        """
        """
        
        sample_merged = collections.defaultdict(list)
        for sample in batch:
            for k, v in sample.items():
                sample_merged[k].append(v)
        batch_size = len(sample_merged['meta'])
        ret = {}
        for key, elems in sample_merged.items():
            if key in ["meta"]:
                ret[key] = [elem.id for elem in elems]
            elif key in ["path", "modality", "calibs","rot_num"]:
                ret[key] = elems
            elif key in["data","points","points1","points2",'transform_param']:
                ret[key] = [elem for elem in elems]
            elif key in ['gt_boxes','gt_boxes1','gt_boxes2']:
                max_gt = max([len(x) for x in elems])
                batch_gt_boxes3d = np.zeros(
                    (batch_size, max_gt, elems[0].shape[-1]), dtype=np.float32)
                for k in range(batch_size):
                    batch_gt_boxes3d[k, :elems[k].__len__(), :] = elems[k]
                ret[key] = batch_gt_boxes3d
        ret['batch_size'] = batch_size

        return ret

    def get_training_loss(self,batch_dict):
       
        disp_dict = {}
        start_time=time.time()
        loss_rpn, tb_dict = self.dense_head.get_loss()
        #print("loss dense_head time:",time.time()-start_time)
        start_time=time.time()
        loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict,batch_dict)
        #print("loss roi_head time:",time.time()-start_time)
        #print("loss_rpn:",loss_rpn.item()," loss_rcnn:",loss_rcnn.item())
        loss = loss_rpn + loss_rcnn
        
        return {"loss": loss}

    def post_processing(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                                or [(B, num_boxes, num_class1), (B, num_boxes, num_class2) ...]
                multihead_label_mapping: [(num_class1), (num_class2), ...]
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
                has_class_labels: True/False
                roi_labels: (B, num_rois)  1 .. num_classes
                batch_pred_labels: (B, num_boxes, 1)
        Returns:

        """
        if self.in_export_mode:
            batch_size=1
        else:
            batch_size = batch_dict['batch_size']
        recall_dict = {}
        pred_dicts = []
        
        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_dict['batch_box_preds'].shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_box_preds'].shape.__len__() == 3
                batch_mask = index

            box_preds = batch_dict['batch_box_preds'][batch_mask]
            src_box_preds = box_preds
            
            if not isinstance(batch_dict['batch_cls_preds'], list):
                cls_preds = batch_dict['batch_cls_preds'][batch_mask]

                src_cls_preds = cls_preds
                assert cls_preds.shape[1] in [1, self.num_class]

                if not batch_dict['cls_preds_normalized']:
                    cls_preds = F.sigmoid(cls_preds)
            else:
                cls_preds = [
                    x[batch_mask] for x in batch_dict['batch_cls_preds']
                ]
                src_cls_preds = cls_preds
                if not batch_dict['cls_preds_normalized']:
                    cls_preds = [F.sigmoid(x) for x in cls_preds]
            
            if self.post_process_cfg["nms_config"]["multi_classes_nms"]:
                """ if not isinstance(cls_preds, list):
                    cls_preds = [cls_preds]
                    multihead_label_mapping = [paddle.arange(1, self.num_class)]
                else:
                    multihead_label_mapping = batch_dict['multihead_label_mapping']
                cur_start_idx = 0
                pred_scores, pred_labels, pred_boxes = [], [], []
                for cur_cls_preds, cur_label_mapping in zip(cls_preds, multihead_label_mapping):
                    assert cur_cls_preds.shape[1] == len(cur_label_mapping)
                    cur_box_preds = box_preds[cur_start_idx: cur_start_idx + cur_cls_preds.shape[0]]
                    cur_pred_scores, cur_pred_labels, cur_pred_boxes = multi_classes_nms(
                        cls_scores=cur_cls_preds, box_preds=cur_box_preds,
                        nms_config=self.post_process_cfg["nms_config"],
                        score_thresh=self.post_process_cfg["score_thresh"]
                    )
                    cur_pred_labels = cur_label_mapping[cur_pred_labels]
                    pred_scores.append(cur_pred_scores)
                    pred_labels.append(cur_pred_labels)
                    pred_boxes.append(cur_pred_boxes)
                    cur_start_idx += cur_cls_preds.shape[0]

                final_scores = paddle.concat(pred_scores, axis=0)
                final_labels = paddle.concat(pred_labels, axis=0)
                final_boxes = paddle.concat(pred_boxes, axis=0) """
                raise NotImplementedError
            else:
                label_preds = paddle.argmax(cls_preds, axis=-1)
                cls_preds = paddle.max(cls_preds, axis=-1)

                
                if self.in_export_mode:
                    label_preds = label_preds + 1

                else:
                    if 'has_class_labels' not in batch_dict:
                        batch_dict['has_class_labels']=False
                    if batch_dict['has_class_labels']:
                        label_key = 'roi_labels' if 'roi_labels' in batch_dict else 'batch_pred_labels'
                        label_preds = batch_dict[label_key][index]
                        
                    else:
                        label_preds = label_preds + 1
                  
                if 'wbf' not in self.post_process_cfg:
                    self.post_process_cfg['wbf']=True
                if self.post_process_cfg['wbf']:
                    if self.post_process_cfg['output_raw_score']:
                        max_cls_preds=paddle.max(src_cls_preds,dim=-1)
                    if self.in_export_mode:
                        score_mask = cls_preds > self.post_process_cfg['score_thresh']
                        indices = paddle.nonzero(score_mask).squeeze(axis=-1)

                        final_scores = paddle.gather(cls_preds, indices)
                        final_labels = paddle.gather(label_preds, indices)
                        final_boxes = paddle.gather(box_preds, indices)
                    else:
                        score_mask = cls_preds > self.post_process_cfg['score_thresh']
                        final_scores = cls_preds[score_mask]
                        final_labels = label_preds[score_mask]
                        final_boxes = box_preds[score_mask] 
                else:   
                    
                    final_scores, final_labels, final_boxes,selected = class_agnostic_nms(
                        box_scores=cls_preds,
                        box_preds=box_preds,
                        label_preds=label_preds,
                        nms_config=self.post_process_cfg["nms_config"],
                        score_thresh=self.post_process_cfg["score_thresh"])
                    if self.post_process_cfg['output_raw_score']:
                        max_cls_preds, = paddle.max(src_cls_preds, axis=-1)
                        final_scores = max_cls_preds[selected]

            """ recall_dict = self.generate_recall_record(
                box_preds=final_boxes if 'rois' not in batch_dict else src_box_preds,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=self.post_process_cfg['recall_thresh_list']
            ) """
            if not self.in_export_mode:
                final_labels, final_scores,final_boxes=compute_WBF(final_labels, final_scores,final_boxes)
            if not self.training:
                final_labels = paddle.to_tensor(final_labels)
                final_scores = paddle.to_tensor(final_scores)
                final_boxes = paddle.to_tensor(final_boxes)
                if len(final_labels.shape)>1:
                    final_labels = final_labels.reshape([-1])
                if len(final_scores.shape)>1:
                    final_scores = final_scores.reshape([-1])
            #print(final_scores)
            if not getattr(self, "in_export_mode", False):
                record_dict = {
                    'box3d_lidar': final_boxes,
                    'scores': final_scores,
                    'label_preds': final_labels
                }
                if 'wbf' not in self.post_process_cfg:
                    self.post_process_cfg['wbf']=True
                if self.post_process_cfg['wbf']:    
                    record_dict.update({'wbf': True})
                pred_dicts.append(record_dict)
            else:
                pred_dicts.append([final_boxes, final_scores, final_labels])
          
            

           

        return pred_dicts

    def _convert_origin_for_eval(self, sample: dict):
        
        if sample.bboxes_3d.origin != [.5, .5, 0]:
            sample.bboxes_3d[:, :3] += sample.bboxes_3d[:, 3:6] * (
                np.array([.5, .5, 0]) - np.array(sample.bboxes_3d.origin))
            sample.bboxes_3d.origin = [.5, .5, 0]
        return sample

    def _parse_results_to_sample(self, results: dict, sample: dict):
        num_samples = len(results)
        new_results = []
        for i in range(num_samples):
            data = Sample(sample["path"][i], sample["modality"][i])
            bboxes_3d = results[i]["box3d_lidar"].numpy()
            labels = results[i]["label_preds"].numpy() - 1
            confidences = results[i]["scores"].numpy()
            bboxes_3d[..., 3:5] = bboxes_3d[..., [4, 3]]
            bboxes_3d[..., -1] = -(bboxes_3d[..., -1] + np.pi / 2.)
            data.bboxes_3d = BBoxes3D(bboxes_3d)
            data.bboxes_3d.coordmode = 'Lidar'
            data.bboxes_3d.origin = [0.5, 0.5, 0.5]
            data.bboxes_3d.rot_axis = 2
            data.labels = labels
            data.confidences = confidences
            data.meta = SampleMeta(id=sample["meta"][i])
            if "calibs" in sample:
                data.calibs = [calib.numpy() for calib in sample["calibs"][i]]
            data = self._convert_origin_for_eval(data)
            new_results.append(data)
        return new_results

    def export(self, save_dir: str, **kwargs):
       
        self.in_export_mode = True
        self.voxel_encoder.in_export_mode = True
        self.voxelizer.in_export_mode = True
        self.backbone.in_export_mode = True
        self.neck.in_export_mode = True
        self.neck1.in_export_mode = True
        self.dense_head.in_export_mode = True
        self.roi_head.in_export_mode = True
        save_path = os.path.join(save_dir, 'ted')
        points_shape = [-1, self.voxel_encoder.in_channels]

        input_spec = [{
            "data":
            InputSpec(shape=points_shape, name='data', dtype='float32')
        }]
        
        paddle.jit.to_static(self, input_spec=input_spec)
        paddle.jit.save(self, save_path, input_spec=input_spec)

        logger.info("Exported model is saved in {}".format(save_dir))




def save_batch_dict(batch_dict, file_path):
    # 将Paddle Tensor转换为numpy数组，以便pickle可以序列化
    for key in batch_dict:
        if isinstance(batch_dict[key], paddle.Tensor):
            batch_dict[key] = batch_dict[key].numpy()

    with open(file_path, 'wb') as f:
        pickle.dump(batch_dict, f)

def load_batch_dict(file_path):
    with open(file_path, 'rb') as f:
        batch_dict = pickle.load(f)

    # 将numpy数组转换回Paddle Tensor
    for key in batch_dict:
        if isinstance(batch_dict[key], (np.ndarray, np.generic)):
            batch_dict[key] = paddle.to_tensor(batch_dict[key])

    return batch_dict