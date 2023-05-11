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


import argparse
import os
import random

import numpy as np
import paddle

from paddle3d.apis.config import Config as cf
from paddle3d.apis.trainer import Trainer

from paddle3d.utils.checkpoint import load_pretrained_model
from paddle3d.utils.logger import logger
import cv2
import numpy as np
from paddle.inference import Config,create_predictor
import time
from paddle3d.ops.voxelize import hard_voxelize
from paddle3d.ops.pointnet2_ops import voxel_query_wrapper,  farthest_point_sample
from paddle3d.ops.iou3d_nms_cuda import nms_gpu
from paddle3d.utils.checkpoint import load_pretrained_model
from paddle3d.apis.trainer import Trainer
from paddle3d.geometries import BBoxes3D
from paddle3d.sample import Sample, SampleMeta
def convert_origin_for_eval( sample: dict):
    if sample.bboxes_3d.origin != [.5, .5, 0]:
        sample.bboxes_3d[:, :3] += sample.bboxes_3d[:, 3:6] * (
            np.array([.5, .5, 0]) - np.array(sample.bboxes_3d.origin))
        sample.bboxes_3d.origin = [.5, .5, 0]
    return sample
def parse_results_to_sample( results: dict, sample: dict):
    num_samples = len(results)
    new_results = []
    for i in range(num_samples):
        data = Sample(sample["path"][i], sample["modality"][i])
        bboxes_3d = results[i]["box3d_lidar"]
        labels = results[i]["label_preds"] - 1
        confidences = results[i]["scores"]
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
        data = convert_origin_for_eval(data)
        new_results.append(data)
    return new_results
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_file",
        type=str,
        help="Model filename, Specify this when your model is a combined model.",
        required=True)
    parser.add_argument(
        "--params_file",
        type=str,
        help=
        "Parameter filename, Specify this when your model is a combined model.",
        required=True)
    parser.add_argument(
        "--num_point_dim",
        type=int,
        default=4,
        help="Dimension of a point in the lidar file.")
    parser.add_argument(
        '--lidar_file', type=str, help='The lidar path.', required=True)
    parser.add_argument(
        "--point_cloud_range",
        dest='point_cloud_range',
        nargs='+',
        help="Range of point cloud for voxelize operation.",
        type=float,
        default=None)
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU card id.")
    parser.add_argument(
        "--use_trt",
        type=int,
        default=0,
        help="Whether to use tensorrt to accelerate when using gpu.")
    parser.add_argument(
        "--trt_precision",
        type=int,
        default=0,
        help="Precision type of tensorrt, 0: kFloat32, 1: kHalf.")
    parser.add_argument(
        "--trt_use_static",
        type=int,
        default=0,
        help="Whether to load the tensorrt graph optimization from a disk path."
    )
    parser.add_argument(
        "--trt_static_dir",
        type=str,
        help="Path of a tensorrt graph optimization directory.")
    parser.add_argument(
        "--collect_shape_info",
        type=int,
        default=0,
        help="Whether to collect dynamic shape before using tensorrt.")
    parser.add_argument(
        "--dynamic_shape_file",
        type=str,
        default="",
        help="Path of a dynamic shape file for tensorrt.")
    parser.add_argument(
        "--config",
        type=str,
        default="/home/aistudio/Paddle3D/configs/voxel_rcnn/voxel_rcnn_005voxel_kitti_car.yml",
       )
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        help='Mini batch size of one gpu or cpu',
        type=int,
        default=1)
    return parser.parse_args()
def worker_init_fn(worker_id):
    np.random.seed(1024)

def read_point(file_path, num_point_dim):
    points = np.fromfile(file_path, np.float32).reshape(-1, num_point_dim)
    points = points[:, :4]
    return points


def filter_points_outside_range(points, point_cloud_range):
    limit_range = np.asarray(point_cloud_range, dtype=np.float32)
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    points = points[mask]
    return points

def preprocess(file_path, num_point_dim, point_cloud_range):
    points = read_point(file_path, num_point_dim)
    
    points = filter_points_outside_range(points, point_cloud_range)
    return points



def init_predictor(model_file,
                   params_file,
                   gpu_id=0,
                   use_trt=False,
                   trt_precision=0,
                   trt_use_static=False,
                   trt_static_dir=None,
                   collect_shape_info=False,
                   dynamic_shape_file=None):
    config = Config(model_file, params_file)
    config.enable_memory_optim()
    config.enable_use_gpu(1000, gpu_id)
    if use_trt:
        precision_mode = paddle.inference.PrecisionType.Float32
        if trt_precision == 1:
            precision_mode = paddle.inference.PrecisionType.Half
        config.enable_tensorrt_engine(
            workspace_size=1 << 30,
            max_batch_size=1,
            min_subgraph_size=40,
            precision_mode=precision_mode,
            use_static=trt_use_static,
            use_calib_mode=False)
        if collect_shape_info:
            config.collect_shape_range_info(dynamic_shape_file)
        else:
            config.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file, True)
        if trt_use_static:
            config.set_optim_cache_dir(trt_static_dir)

    predictor = create_predictor(config)
    
    return predictor


def parse_result(box3d_lidar, label_preds, scores):
    num_bbox3d, bbox3d_dims = box3d_lidar.shape
    for box_idx in range(num_bbox3d):
        # filter fake results: score = -1
        if scores[box_idx] < 0:
            continue
        '''print(
            "Score: {} Label: {} Box(x_c, y_c, z_c, w, l, h, -rot): {} {} {} {} {} {} {}"
            .format(scores[box_idx], label_preds[box_idx],
                    box3d_lidar[box_idx, 0], box3d_lidar[box_idx, 1],
                    box3d_lidar[box_idx, 2], box3d_lidar[box_idx, 3],
                    box3d_lidar[box_idx, 4], box3d_lidar[box_idx, 5],
                    box3d_lidar[box_idx, 6]))'''


def run(predictor, points):
    # copy img data to input tensor
    input_names = predictor.get_input_names()
    for i, name in enumerate(input_names):
        if name == "data":
            input_tensor = predictor.get_input_handle(name)
            input_tensor.reshape(points.shape)
            input_tensor.copy_from_cpu(points.copy())

    # do the inference
    predictor.run()

    results = []
    # get out data from output tensor
    output_names = predictor.get_output_names()
    for i, name in enumerate(output_names):
        output_tensor = predictor.get_output_handle(name)
        if i == 0:
            box3d_lidar = output_tensor.copy_to_cpu()
        elif i == 1:
            scores = output_tensor.copy_to_cpu()
        elif i == 2:
            label_preds = output_tensor.copy_to_cpu()
    return box3d_lidar, label_preds, scores


def main(args):
   
    if not os.path.exists(args.config):
        raise RuntimeError("Config file `{}` does not exist!".format(args.config))
    cfg = cf(path=args.config, batch_size=args.batch_size)
  
    predictor = init_predictor(args.model_file, args.params_file, args.gpu_id,
                               args.use_trt, args.trt_precision,
                               args.trt_use_static, args.trt_static_dir,
                               args.collect_shape_info, args.dynamic_shape_file)
   
    if cfg.val_dataset is None:
        raise RuntimeError(
            'The validation dataset is not specified in the configuration file!'
        )
    elif len(cfg.val_dataset) == 0:
        raise ValueError(
            'The length of validation dataset is 0. Please check if your dataset is valid!'
        )
   
    dic = cfg.to_dict()
    batch_size = dic.pop('batch_size')
    dic.update({
        'dataloader_fn': {
            'batch_size': batch_size,
            'num_workers': 0,
            'worker_init_fn': worker_init_fn
        }
    })
   
 
    
    trainer = Trainer(**dic)
    start_time = time.time()
    
    if trainer.val_dataset is None:
        raise RuntimeError('No evaluation dataset specified!')
    msg = 'evaluate on validate dataset'
    metric_obj = trainer.val_dataset.metric
    print("use_trt:",args.use_trt)
    print("trt_precision:",args.trt_precision)
    
    points = preprocess(args.lidar_file, args.num_point_dim,
                        args.point_cloud_range)
    for i in range(100):
        box3d_lidar, label_preds, scores = run(predictor, points)
    total_time=0
    total=0
    
    for idx, sample in trainer.logger.enumerate(trainer.eval_dataloader, msg=msg):
        start_time = time.time()
        
        points = filter_points_outside_range(np.array(sample['data'][0]), 
                        args.point_cloud_range)
        print("Preprocess time: {}".format(time.time() - start_time))
        start_time = time.time()
        box3d_lidar, label_preds, scores = run(predictor, points)
        step_time=time.time() - start_time
        start_time=time.time()
        result=[]
        result.append({'box3d_lidar':box3d_lidar,'scores':scores,'label_preds':label_preds})
        result=parse_results_to_sample(result, sample)

        metric_obj.update(predictions=result, ground_truths=sample)
        print("parse result time: {}".format(time.time() - start_time))
        total_time=total_time+step_time
        total=total+1
    
    print("avg_time:",total_time/total)
    metrics = metric_obj.compute(verbose=True)
   
    
    
    
  


if __name__ == '__main__':
    args = parse_args()

    main(args)