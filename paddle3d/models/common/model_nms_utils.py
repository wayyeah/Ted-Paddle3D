# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import numpy as np
from paddle3d.ops import iou3d_nms_cuda
def limit(ang):
    ang = ang % (2 * np.pi)

    ang[ang > np.pi] = ang[ang > np.pi] - 2 * np.pi

    ang[ang < -np.pi] = ang[ang < -np.pi] + 2 * np.pi

    return ang


def class_agnostic_nms(box_scores,
                       box_preds,
                       label_preds,
                       nms_config,
                       score_thresh=None):
    def nms(box_scores, box_preds, label_preds, nms_config):
        order = box_scores.argsort(0, descending=True)
        order = order[:nms_config['nms_pre_maxsize']]
        box_preds = paddle.gather(box_preds, index=order)
        box_scores = paddle.gather(box_scores, index=order)
        label_preds = paddle.gather(label_preds, index=order)
        # When order is one-value tensor,
        # boxes[order] loses a dimension, so we add a reshape
        keep, num_out = iou3d_nms_cuda.nms_gpu(box_preds,
                                               nms_config['nms_thresh'])
        selected = keep[0:num_out]
        selected = selected[:nms_config['nms_post_maxsize']]
        selected_score = paddle.gather(box_scores, index=selected)
        selected_box = paddle.gather(box_preds, index=selected)
        selected_label = paddle.gather(label_preds, index=selected)
        return selected_score, selected_label, selected_box,selected

    if score_thresh is not None:
        scores_mask = box_scores >= score_thresh

        def box_empty(box_scores, box_preds, label_preds):
            fake_score = paddle.to_tensor([-1.0], dtype=box_scores.dtype)
            fake_label = paddle.to_tensor([-1.0], dtype=label_preds.dtype)
            fake_box = paddle.to_tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                                        dtype=box_preds.dtype)
            fake_selected = paddle.to_tensor([0], dtype=paddle.int64)
            return fake_score, fake_label, fake_box,fake_selected

        def box_not_empty(scores_mask, box_scores, box_preds, label_preds,
                          nms_config):
            nonzero_index = paddle.nonzero(scores_mask)
            box_scores = paddle.gather(box_scores, index=nonzero_index)
            box_preds = paddle.gather(box_preds, index=nonzero_index)
            label_preds = paddle.gather(label_preds, index=nonzero_index)
            return nms(box_scores, box_preds, label_preds, nms_config)

        return paddle.static.nn.cond(
            paddle.logical_not(scores_mask.any()), lambda: box_empty(
                box_scores, box_preds, label_preds), lambda: box_not_empty(
                    scores_mask, box_scores, box_preds, label_preds, nms_config)
        )
    else:
        return nms(box_scores, box_preds, label_preds, nms_config)
    
def compute_WBF(det_names, det_scores, det_boxes, iou_thresh=0.85, iou_thresh2=0.03, type='mean'):
    if len(det_names) == 0:
        return det_names, det_scores, det_boxes
    cluster_id = -1
    cluster_box_dict = {}
    cluster_score_dict = {}

    cluster_merged_dict = {}
    cluster_name_dict = {}
    '''
    det_boxes[:, 6] = common_utils.limit_period(
        det_boxes[:, 6], offset=0.5, period=2 * np.pi
    )
    '''
    det_boxes[:, 6] = limit(det_boxes[:, 6])

    for i, box in enumerate(det_boxes):

        score = det_scores[i]
        name = det_names[i]
        if i == 0:
            cluster_id += 1
            cluster_box_dict[cluster_id] = [box]
            cluster_score_dict[cluster_id] = [score]
            cluster_merged_dict[cluster_id] = box
            cluster_name_dict[cluster_id] = name
            continue

        valid_clusters = []
        keys = list(cluster_merged_dict)
        keys.sort()
        for key in keys:
            valid_clusters.append(cluster_merged_dict[key])

        valid_clusters = np.array(valid_clusters).reshape((-1, 7))
        boxes_a=paddle.to_tensor(np.array([box[:7]]),dtype='float32',place=paddle.CPUPlace())
        boxes_b=paddle.to_tensor(valid_clusters,dtype='float32',place=paddle.CPUPlace())
        assert boxes_a.shape[1] == 7 and boxes_b.shape[1] == 7
        
        ious = iou3d_nms_cuda.boxes_iou_bev_cpu(boxes_a,boxes_b)
        ious=ious.numpy()
        argmax = np.argmax(ious, -1)[0]
        max_iou = np.max(ious, -1)[0]

        if max_iou >= iou_thresh:
            cluster_box_dict[argmax].append(box)
            cluster_score_dict[argmax].append(score)
        elif iou_thresh2<=max_iou<iou_thresh:
            continue
        else:
            cluster_id += 1
            cluster_box_dict[cluster_id] = [box]
            cluster_score_dict[cluster_id] = [score]
            cluster_merged_dict[cluster_id] = box
            cluster_name_dict[cluster_id] = name

    out_boxes = []
    out_scores = []
    out_name = []
    for i in cluster_merged_dict.keys():
        if type == 'mean':
            score_sum = 0
            box_sum = paddle.zeros([7])

            angles = []

            for j, sub_score in enumerate(cluster_score_dict[i]):
                box_sum += cluster_box_dict[i][j]
                score_sum += sub_score
                angles.append(cluster_box_dict[i][j][6])
            box_sum /= len(cluster_score_dict[i])
            score_sum /= len(cluster_score_dict[i])

            cluster_merged_dict[i][:6] = box_sum[:6]

            angles = np.array(angles)
            angles = limit(angles)
            res = angles - cluster_merged_dict[i][6]
            res = limit(res)
            res = res[paddle.abs(res) < 1.5]
            res = res.mean()
            b = cluster_merged_dict[i][6] + res
            cluster_merged_dict[i][6] = b

            out_scores.append(score_sum)
            out_boxes.append(cluster_merged_dict[i])
            out_name.append(cluster_name_dict[i])
        elif type == 'max':
            out_scores.append(np.max(cluster_score_dict[i]))
            out_boxes.append(cluster_merged_dict[i])
            out_name.append(cluster_name_dict[i])

    out_boxes = np.array(out_boxes)
    out_scores = np.array(out_scores)
    out_names = np.array(out_name)

    return out_names, out_scores, out_boxes
