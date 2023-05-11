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
"""
This code is based on https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/roi_heads/target_assigner/proposal_target_layer.py
Ths copyright of OpenPCDet is as follows:
Apache-2.0 license [see LICENSE for details].
"""

import numpy as np
import paddle
import paddle.nn as nn

from . import iou3d_nms_utils


class ProposalTargetLayer(nn.Layer):
    def __init__(self, roi_sampler_cfg):
        super().__init__()
        self.roi_sampler_cfg = roi_sampler_cfg
    def limit(self,ang):
        ang = ang % (2 * np.pi)

        ang[ang > np.pi] = ang[ang > np.pi] - 2 * np.pi

        ang[ang < -np.pi] = ang[ang < -np.pi] + 2 * np.pi

        return ang

    def ang_weight(self,pred, gt):

        a = paddle.abs(pred - gt)
        b = 2 * np.pi - paddle.abs(pred - gt)

        res = paddle.stack([a, b])

        res = paddle.min(res, 0)[0]

        return 1 - res / np.pi
    def forward(self, batch_dict, ind=''):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:
            batch_dict:
                rois: (B, M, 7 + C)
                gt_of_rois: (B, M, 7 + C)
                gt_iou_of_rois: (B, M)
                roi_scores: (B, M)
                roi_labels: (B, M)
                reg_valid_mask: (B, M)
                rcnn_cls_labels: (B, M)
        """
        batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels = self.sample_rois_for_rcnn(
            batch_dict=batch_dict,ind=ind)
        
        # regression valid mask
        if self.roi_sampler_cfg['cls_score_type'] in ['roi_iou_x', 'roi_ioud_x']:

            reg_valid_mask = paddle.zeros_like(batch_roi_ious, dtype='int64')
            for cls_i in range(len(self.roi_sampler_cfg['reg_fg_thresh'])):
                reg_fg_thresh = self.roi_sampler_cfg['reg_fg_thresh'][cls_i]
                cls_mask = batch_gt_of_rois[...,-1] == (cls_i+1)

                
                if self.roi_sampler_cfg.get('enable_hard_sampling', False):
                    mask_hard = (batch_roi_ious < reg_fg_thresh) & (batch_roi_ious > self.roi_sampler_cfg['hard_sampling_thresh'][cls_i]) & cls_mask

                    mask_prob = paddle.zeros_like(mask_hard, dtype='bool')
                    teval = int(1/self.roi_sampler_cfg['hard_sampling_ratio'][cls_i])
                    ints = range(np.random.randint(0, teval), mask_prob.shape[0], teval)

                    mask_prob[ints] = 1

                    mask_hard2 = mask_hard * mask_prob

                    this_fg_inds1 = ((batch_roi_ious > reg_fg_thresh) & cls_mask)
                    
                    this_reg_valid_mask = paddle.cast(this_fg_inds1, 'int64') + paddle.cast(mask_hard2, 'int64')


                else:
                    this_reg_valid_mask = ((batch_roi_ious > reg_fg_thresh) & cls_mask)
                reg_valid_mask += this_reg_valid_mask
        else:
            reg_valid_mask = (batch_roi_ious >
                            self.roi_sampler_cfg["reg_fg_thresh"]).astype('int64')

        # classif.concation label
        if self.roi_sampler_cfg["cls_score_type"] == 'cls':
            batch_cls_labels = (
                batch_roi_ious >
                self.roi_sampler_cfg["cls_fg_thresh"]).astype('int64')
            ignore_mask = (batch_roi_ious > self.roi_sampler_cfg["cls_bg_thresh"]) & \
                          (batch_roi_ious < self.roi_sampler_cfg["cls_fg_thresh"])
            batch_cls_labels[ignore_mask > 0] = -1
        elif self.roi_sampler_cfg["cls_score_type"] == 'roi_iou':
            iou_bg_thresh = self.roi_sampler_cfg["cls_bg_thresh"]
            iou_fg_thresh = self.roi_sampler_cfg["cls_fg_thresh"]
            fg_mask = batch_roi_ious > iou_fg_thresh
            bg_mask = batch_roi_ious < iou_bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)

            batch_cls_labels = (fg_mask > 0).astype('float32')
            batch_cls_labels[interval_mask] = \
                (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)
        elif self.roi_sampler_cfg["cls_score_type"] == 'roi_ioud':
            iou_bg_thresh = self.roi_sampler_cfg["cls_bg_thresh"]
            iou_fg_thresh = self.roi_sampler_cfg["cls_fg_thresh"]
            fg_mask = batch_roi_ious > iou_fg_thresh
            bg_mask = batch_roi_ious < iou_bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)

            batch_cls_labels = (fg_mask > 0)
            batch_cls_labels[interval_mask] = \
                (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)

            ang_roi = batch_rois[...,6]
            ang_gt = batch_gt_of_rois[...,6]

            ang_roi = self.limit(ang_roi)
            ang_gt = self.limit(ang_gt)

            ang_target = self.ang_weight(ang_roi,ang_gt)
            direction_constraint = self.roi_sampler_cfg['direction_min']
            direction_constraint2 = self.roi_sampler_cfg['direction_max']

            ang_target = (paddle.clamp(ang_target, direction_constraint,
                                      direction_constraint2) - direction_constraint) / (
                                 direction_constraint2 - direction_constraint)

            batch_cls_labels *= ang_target


        elif self.roi_sampler_cfg['cls_score_type']== 'roi_ioud_x':
            all_iou_bg_thresh = self.roi_sampler_cfg['cls_bg_thresh']
            all_iou_fg_thresh = self.roi_sampler_cfg['cls_fg_thresh']
            batch_cls_labels = batch_roi_ious.new_zeros(size = batch_roi_ious.shape)
            for cls_id in range(len(all_iou_bg_thresh)):
                gt_cls = batch_gt_of_rois[..., -1]
                iou_fg_thresh = all_iou_fg_thresh[cls_id]
                iou_bg_thresh = all_iou_bg_thresh[cls_id]

                cls_mask = gt_cls == (cls_id+1)

                fg_mask = batch_roi_ious > iou_fg_thresh
                bg_mask = batch_roi_ious < iou_bg_thresh
                interval_mask = (fg_mask == 0) & (bg_mask == 0)

                cls_labels = (fg_mask > 0)
                cls_labels[interval_mask] = \
                    (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)

                ang_roi = batch_rois[...,6]
                ang_gt = batch_gt_of_rois[...,6]

                ang_roi = self.limit(ang_roi)
                ang_gt = self.limit(ang_gt)

                ang_target = self.ang_weight(ang_roi,ang_gt)
                direction_constraint = self.roi_sampler_cfg['direction_min']
                direction_constraint2 = self.roi_sampler_cfg['direction_max']

                ang_target = (paddle.clip(ang_target, direction_constraint, direction_constraint2 ) - direction_constraint) / (
                            direction_constraint2  - direction_constraint)

                cls_labels*=ang_target
                
                batch_cls_labels[cls_mask] = cls_labels[cls_mask]
 
        elif self.roi_sampler_cfg['cls_score_type'] == 'roi_iou_x':
            all_iou_bg_thresh = self.roi_sampler_cfg['cls_bg_thresh']
            all_iou_fg_thresh = self.roi_sampler_cfg['cls_fg_thresh']
            batch_cls_labels = paddle.zeros_like(batch_roi_ious)
            
            for cls_id in range(len(all_iou_bg_thresh)):
                gt_cls = batch_gt_of_rois[..., -1]
                iou_fg_thresh = all_iou_fg_thresh[cls_id]
                iou_bg_thresh = all_iou_bg_thresh[cls_id]

                cls_mask = gt_cls == (cls_id+1)

                fg_mask = batch_roi_ious > iou_fg_thresh
                bg_mask = batch_roi_ious < iou_bg_thresh # may question cls_labels=0
                interval_mask = (fg_mask == 0) & (bg_mask == 0)
                cls_labels = (fg_mask > 0)
                cls_labels = paddle.cast(cls_labels, dtype='float32')
                cls_labels[interval_mask] = (batch_roi_ious[interval_mask] - iou_bg_thresh) / (iou_fg_thresh - iou_bg_thresh)
                #print("batch_cls_labels[cls_mask]:",batch_cls_labels[cls_mask])
                #print("cls_labels[cls_mask]:",cls_labels[cls_mask])
                batch_cls_labels[cls_mask] = cls_labels[cls_mask].astype('float32')
                
        else:
            raise NotImplementedError

        targets_dict = {
            'rois': batch_rois,
            'gt_of_rois': batch_gt_of_rois,
            'gt_iou_of_rois': batch_roi_ious,
            'roi_scores': batch_roi_scores,
            'roi_labels': batch_roi_labels,
            'reg_valid_mask': reg_valid_mask,
            'rcnn_cls_labels': batch_cls_labels
        }

        return targets_dict

    def sample_rois_for_rcnn(self, batch_dict,ind=''):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                roi_scores: (B, num_rois)
                gt_boxes: (B, N, 7 + C + 1)
                roi_labels: (B, num_rois)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        roi_scores = batch_dict['roi_scores']
        roi_labels = batch_dict['roi_labels']
        gt_boxes = batch_dict['gt_boxes'+ind]

        code_size = rois.shape[-1]
        gt_code_size = gt_boxes.shape[-1]
        batch_rois = paddle.zeros(
            (batch_size, self.roi_sampler_cfg["roi_per_image"], code_size),
            dtype=rois.dtype)
        batch_gt_of_rois = paddle.zeros(
            (batch_size, self.roi_sampler_cfg["roi_per_image"], gt_code_size),
            dtype=gt_boxes.dtype)
        batch_roi_ious = paddle.zeros(
            (batch_size, self.roi_sampler_cfg["roi_per_image"]),
            dtype=rois.dtype)
        batch_roi_scores = paddle.zeros(
            (batch_size, self.roi_sampler_cfg["roi_per_image"]),
            dtype=roi_scores.dtype)
        batch_roi_labels = paddle.zeros(
            (batch_size, self.roi_sampler_cfg["roi_per_image"]),
            dtype=roi_labels.dtype)

        for index in range(batch_size):
            cur_roi, cur_gt, cur_roi_labels, cur_roi_scores = \
                rois[index], gt_boxes[index], roi_labels[index], roi_scores[index]
            k = cur_gt.__len__() - 1
            while k >= 0 and cur_gt[k].sum() == 0:
                k -= 1
            cur_gt = cur_gt[:k + 1]
            cur_gt = paddle.zeros(
                (1, cur_gt.shape[1])) if len(cur_gt) == 0 else cur_gt
#may question
            if self.roi_sampler_cfg.get('sample_roi_by_each_class', False):
                max_overlaps, gt_assignment = self.get_max_iou_with_same_class(
                    rois=cur_roi,
                    roi_labels=cur_roi_labels,
                    gt_boxes=cur_gt[:, 0:7],
                    gt_labels=cur_gt[:, -1].astype('int64'))
            else:
                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(
                    cur_roi, cur_gt[:, 0:7])  # (M, N)
                max_overlaps = paddle.max(iou3d, axis=1)
                gt_assignment = paddle.argmax(iou3d, axis=1)
            if self.roi_sampler_cfg['cls_score_type'] in ['roi_iou_x','roi_ioud_x']:
                sampled_inds = self.subsample_rois(max_overlaps=max_overlaps, gts = cur_gt[gt_assignment])
            else:
                sampled_inds = self.subsample_rois(max_overlaps=max_overlaps)

            batch_rois[index] = cur_roi[sampled_inds]
            batch_roi_labels[index] = cur_roi_labels[sampled_inds]
            batch_roi_ious[index] = max_overlaps[sampled_inds]  #may question
            batch_roi_scores[index] = cur_roi_scores[sampled_inds]
            batch_gt_of_rois[index] = cur_gt[gt_assignment[sampled_inds]]

        return batch_rois, batch_gt_of_rois, batch_roi_ious, batch_roi_scores, batch_roi_labels

    def subsample_rois(self, max_overlaps, gts=None):
        # sample fg, easy_bg, hard_bg  may question
        fg_rois_per_image = int(np.round(self.roi_sampler_cfg['fg_ratio'] * self.roi_sampler_cfg['roi_per_image']))

        if gts is None:
            fg_thresh = min(self.roi_sampler_cfg['reg_fg_thresh'], self.roi_sampler_cfg['cls_fg_thresh'])
            fg_inds = paddle.reshape(paddle.nonzero(max_overlaps >= fg_thresh), shape=(-1,))
        else:
            fg_inds = paddle.zeros_like(max_overlaps, dtype='int64')
            for i in range(len(self.roi_sampler_cfg['cls_fg_thresh'])):
                cls_mask = gts[..., -1] == (i + 1)
                this_fg_thresh = min(self.roi_sampler_cfg['reg_fg_thresh'][i], self.roi_sampler_cfg['cls_fg_thresh'][i])

                this_fg_inds = (max_overlaps >= this_fg_thresh) & cls_mask

                fg_inds += this_fg_inds.astype('int64')
            fg_inds = paddle.nonzero(fg_inds).flatten()

        easy_bg_inds = paddle.reshape(paddle.nonzero(max_overlaps < self.roi_sampler_cfg['cls_bg_thresh_lo']), shape=(-1,))

        if gts is None:
            hard_bg_inds = paddle.reshape(paddle.nonzero((max_overlaps < self.roi_sampler_cfg['reg_fg_thresh']) &
                                                        (max_overlaps >= self.roi_sampler_cfg['cls_bg_thresh_lo'])), shape=(-1,))
        else:
            hard_bg_inds = paddle.zeros_like(max_overlaps, dtype='int64')
            for i in range(len(self.roi_sampler_cfg['reg_fg_thresh'])):
                cls_mask = gts[..., -1] == (i + 1)
                this_hard_bg_inds = (max_overlaps < self.roi_sampler_cfg['reg_fg_thresh'][i]) & \
                                    (max_overlaps >= self.roi_sampler_cfg['cls_bg_thresh_lo']) & cls_mask
                hard_bg_inds += this_hard_bg_inds.astype('int64')
            hard_bg_inds = paddle.nonzero(hard_bg_inds).flatten()
        fg_num_rois = fg_inds.shape[0]
        bg_num_rois = hard_bg_inds.shape[0] + easy_bg_inds.shape[0]

        if fg_num_rois > 0 and bg_num_rois > 0:
            # sampling fg
            fg_rois_per_this_image = min(fg_rois_per_image, fg_num_rois)

            rand_num = np.random.permutation(fg_num_rois).astype('int64')
            rand_num = paddle.to_tensor(rand_num)
            fg_inds = fg_inds[rand_num[:fg_rois_per_this_image]]

            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg['roi_per_image'] - fg_rois_per_this_image
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg['hard_bg_ratio']
            )

        elif fg_num_rois > 0 and bg_num_rois == 0:
            # sampling fg
            rand_num = np.floor(np.random.rand(self.roi_sampler_cfg['roi_per_image']) * fg_num_rois).astype('int64')
            rand_num = paddle.to_tensor(rand_num)
            fg_inds = fg_inds[rand_num]
            bg_inds = []

        elif bg_num_rois > 0 and fg_num_rois == 0:
            # sampling bg
            bg_rois_per_this_image = self.roi_sampler_cfg['roi_per_image']
            bg_inds = self.sample_bg_inds(
                hard_bg_inds, easy_bg_inds, bg_rois_per_this_image, self.roi_sampler_cfg['hard_bg_ratio']
            )
        else:
            print('maxoverlaps:(min=%f, max=%f)' % (paddle.min(max_overlaps).item(), paddle.max(max_overlaps).item()))
            print('ERROR: FG=%d, BG=%d' % (fg_num_rois, bg_num_rois))
            raise NotImplementedError

        sampled_inds = paddle.concat((fg_inds, paddle.to_tensor(bg_inds, dtype='int64')) if bg_inds.numel() > 0 else fg_inds, axis=0)

        return sampled_inds


    @staticmethod
    def sample_bg_inds(hard_bg_inds, easy_bg_inds, bg_rois_per_this_image,
                       hard_bg_ratio):
        if hard_bg_inds.numel() > 0 and easy_bg_inds.numel() > 0:
            hard_bg_inds = hard_bg_inds.reshape([-1])
            easy_bg_inds = easy_bg_inds.reshape([-1])
            hard_bg_rois_num = min(
                int(bg_rois_per_this_image * hard_bg_ratio), len(hard_bg_inds))
            easy_bg_rois_num = bg_rois_per_this_image - hard_bg_rois_num

            # sampling hard bg
            rand_idx = paddle.randint(
                low=0, high=hard_bg_inds.numel(),
                shape=(hard_bg_rois_num, )).astype('int64')
            hard_bg_inds = hard_bg_inds[rand_idx]

            # sampling easy bg
            rand_idx = paddle.randint(
                low=0, high=easy_bg_inds.numel(),
                shape=(easy_bg_rois_num, )).astype('int64')
            easy_bg_inds = easy_bg_inds[rand_idx]

            bg_inds = paddle.concat([hard_bg_inds, easy_bg_inds], axis=0)
        elif hard_bg_inds.numel() > 0 and easy_bg_inds.numel() == 0:
            hard_bg_inds = hard_bg_inds.reshape([-1])
            hard_bg_rois_num = bg_rois_per_this_image
            # sampling hard bg
            rand_idx = paddle.randint(
                low=0, high=hard_bg_inds.numel(),
                shape=(hard_bg_rois_num, )).astype('int64')
            bg_inds = hard_bg_inds[rand_idx]
        elif hard_bg_inds.numel() == 0 and easy_bg_inds.numel() > 0:
            easy_bg_inds = easy_bg_inds.reshape([-1])
            easy_bg_rois_num = bg_rois_per_this_image
            # sampling easy bg
            rand_idx = paddle.randint(
                low=0, high=easy_bg_inds.numel(),
                shape=(easy_bg_rois_num, )).astype('int64')
            bg_inds = easy_bg_inds[rand_idx]
        else:
            raise NotImplementedError

        return bg_inds

    @staticmethod
    def get_max_iou_with_same_class(rois, roi_labels, gt_boxes, gt_labels):
        """
        Args:
            rois: (N, 7)
            roi_labels: (N)
            gt_boxes: (N, )
            gt_labels:

        Returns:

        """
        """
        :param rois: (N, 7)
        :param roi_labels: (N)
        :param gt_boxes: (N, 8)
        :return:
        """
        max_overlaps = paddle.zeros(shape=rois.shape[:1], dtype=rois.dtype)
        gt_assignment = paddle.zeros(shape=roi_labels.shape, dtype=roi_labels.dtype)

        for k in range(int(gt_labels.min().numpy()[0]), int(gt_labels.max().numpy()[0]) + 1):
            roi_mask = (roi_labels == k)
            gt_mask = (gt_labels == k)
            if roi_mask.sum().numpy()[0] > 0 and gt_mask.sum().numpy()[0] > 0:
                cur_roi = rois[roi_mask]
                cur_gt = gt_boxes[gt_mask]
                original_gt_assignment = paddle.nonzero(gt_mask).flatten()

                iou3d = iou3d_nms_utils.boxes_iou3d_gpu(cur_roi[:, :7], cur_gt[:, :7])  # (M, N)
                cur_max_overlaps= paddle.max(iou3d, axis=1)
                cur_gt_assignment = paddle.argmax(iou3d, axis=1)
                max_overlaps = paddle.scatter(max_overlaps, paddle.nonzero(roi_mask).flatten(), cur_max_overlaps)
                gt_assignment = paddle.scatter(gt_assignment, paddle.nonzero(roi_mask).flatten(), original_gt_assignment[cur_gt_assignment])

        return max_overlaps, gt_assignment
