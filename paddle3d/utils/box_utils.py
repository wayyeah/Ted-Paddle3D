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

import numpy as np

import scipy
import paddle
from scipy.spatial import Delaunay
from . import common_utils
from paddle3d.ops import roiaware_pool3d

def limit_period(val, offset: float = 0.5, period: float = np.pi):
    return val - np.floor(val / period + offset) * period


def boxes3d_kitti_lidar_to_lidar(boxes3d_lidar):
    """
    convert boxes from [x, y, z, w, l, h, yaw] to [x, y, z, l, w, h, heading], bottom_center -> obj_center
    """
    # yapf: disable
    w, l, h, r = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], boxes3d_lidar[:, 5:6], boxes3d_lidar[:, 6:7]
    boxes3d_lidar[:, 2] += h[:, 0] / 2
    # yapf: enable
    return np.concatenate([boxes3d_lidar[:, 0:3], l, w, h, -(r + np.pi / 2)],
                          axis=-1)


def boxes3d_lidar_to_kitti_lidar(boxes3d_lidar):
    """
    convert boxes from [x, y, z, l, w, h, heading] to [x, y, z, w, l, h, yaw], obj_center -> bottom_center
    """
    # yapf: disable
    l, w, h, heading = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], boxes3d_lidar[:, 5:6], boxes3d_lidar[:,6:7]
    boxes3d_lidar[:, 2] -= h[:, 0] / 2
    # yapf: enable
    return np.concatenate(
        [boxes3d_lidar[:, 0:3], w, l, h, -(heading + np.pi / 2)], axis=-1)


def in_hull(p, hull):
    """
    :param p: (N, K) test points
    :param hull: (M, K) M corners of a box
    :return (N) bool
    """
    try:
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)
        flag = hull.find_simplex(p) >= 0
    except scipy.spatial.qhull.QhullError:
        print('Warning: not a hull %s' % str(hull))
        flag = np.zeros(p.shape[0], dtype=np.bool)

    return flag
def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
    """
    boxes3d, is_numpy = common_utils.check_numpy_to_paddle(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = common_utils.rotate_points_along_z(corners3d.view(-1, 8, 3), boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d

def mask_boxes_outside_range_numpy(boxes, limit_range, min_num_corners=1):
    """
    Args:
        boxes: (N, 7) [x, y, z, dx, dy, dz, heading, ...], (x, y, z) is the box center
        limit_range: [minx, miny, minz, maxx, maxy, maxz]
        min_num_corners:

    Returns:

    """
    if boxes.shape[1] > 7:
        boxes = boxes[:, 0:7]
    corners = boxes_to_corners_3d(boxes)  # (N, 8, 3)
    mask = ((corners >= limit_range[0:3]) & (corners <= limit_range[3:6])).all(axis=2)
    mask = mask.sum(axis=1) >= min_num_corners  # (N)

    return mask

def remove_points_in_boxes3d(points, boxes3d):
    """
    Args:
        points: (num_points, 3 + C)
        boxes3d: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps

    Returns:

    """
    boxes3d, is_numpy = common_utils.check_numpy_to_torch(boxes3d)
    points, is_numpy = common_utils.check_numpy_to_torch(points)
    point_masks = roiaware_pool3d_utils.check_pt_in_box3d(points[:, 0:3], boxes3d)
    points = points[point_masks.sum(dim=0) == 0]

    return points.numpy() if is_numpy else points

def boxes3d_kitti_camera_to_lidar(boxes3d_camera, calib):
    """
    Args:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
        calib:

    Returns:
        boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    """
    xyz_camera = boxes3d_camera[:, 0:3]
    l, h, w, r = boxes3d_camera[:, 3:4], boxes3d_camera[:, 4:5], boxes3d_camera[:, 5:6], boxes3d_camera[:, 6:7]
    xyz_lidar = calib.rect_to_lidar(xyz_camera)
    xyz_lidar[:, 2] += h[:, 0] / 2
    return np.concatenate([xyz_lidar, l, w, h, -(r + np.pi / 2)], axis=-1)

def boxes3d_kitti_fakelidar_to_lidar(boxes3d_lidar):
    """
    Args:
        boxes3d_fakelidar: (N, 7) [x, y, z, w, l, h, r] in old LiDAR coordinates, z is bottom center

    Returns:
        boxes3d_lidar: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    """
    w, l, h, r = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], boxes3d_lidar[:, 5:6], boxes3d_lidar[:, 6:7]
    boxes3d_lidar[:, 2] += h[:, 0] / 2
    return np.concatenate([boxes3d_lidar[:, 0:3], l, w, h, -(r + np.pi / 2)], axis=-1)


def boxes3d_kitti_lidar_to_fakelidar(boxes3d_lidar):
    """
    Args:
        boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

    Returns:
        boxes3d_fakelidar: [x, y, z, w, l, h, r] in old LiDAR coordinates, z is bottom center

    """
    dx, dy, dz, heading = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], boxes3d_lidar[:, 5:6], boxes3d_lidar[:, 6:7]
    boxes3d_lidar[:, 2] -= dz[:, 0] / 2
    return np.concatenate([boxes3d_lidar[:, 0:3], dy, dx, dz, -heading - np.pi / 2], axis=-1)
def enlarge_box3d(boxes3d, extra_width=(0, 0, 0)):
    """
    Args:
        boxes3d: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
        extra_width: [extra_x, extra_y, extra_z]

    Returns:

    """
    boxes3d, is_numpy = common_utils.check_numpy_to_paddle(boxes3d)
    large_boxes3d = boxes3d.clone()

    large_boxes3d[:, 3:6] += boxes3d.new_tensor(extra_width)[None, :]
    return large_boxes3d

def boxes3d_lidar_to_kitti_camera(boxes3d_lidar, calib):
    """
    :param boxes3d_lidar: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    :param calib:
    :return:
        boxes3d_camera: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    """
    xyz_lidar = boxes3d_lidar[:, 0:3]
    l, w, h, r = boxes3d_lidar[:, 3:4], boxes3d_lidar[:, 4:5], boxes3d_lidar[:, 5:6], boxes3d_lidar[:, 6:7]

    xyz_lidar[:, 2] -= h.reshape(-1) / 2
    xyz_cam = calib.lidar_to_rect(xyz_lidar)
    # xyz_cam[:, 1] += h.reshape(-1) / 2
    r = -r - np.pi / 2
    return np.concatenate([xyz_cam, l, h, w, r], axis=-1)

def boxes3d_to_corners3d_kitti_camera(boxes3d, bottom_center=True):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, ry] in camera coords, see the definition of ry in KITTI dataset
    :param bottom_center: whether y is on the bottom center of object
    :return: corners3d: (N, 8, 3)
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    """
    boxes_num = boxes3d.shape[0]
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2], dtype=np.float32).T
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    if bottom_center:
        y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        y_corners = np.array([h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                         [zeros, ones, zeros],
                         [np.sin(ry), zeros, np.cos(ry)]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)


def boxes3d_kitti_camera_to_imageboxes(boxes3d, calib, image_shape=None):
    """
    :param boxes3d: (N, 7) [x, y, z, l, h, w, r] in rect camera coords
    :param calib:
    :return:
        box_2d_preds: (N, 4) [x1, y1, x2, y2]
    """
    corners3d = boxes3d_to_corners3d_kitti_camera(boxes3d)
    pts_img, _ = calib.rect_to_img(corners3d.reshape(-1, 3))
    corners_in_image = pts_img.reshape(-1, 8, 2)

    min_uv = np.min(corners_in_image, axis=1)  # (N, 2)
    max_uv = np.max(corners_in_image, axis=1)  # (N, 2)
    boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
    if image_shape is not None:
        boxes2d_image[:, 0] = np.clip(boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 1] = np.clip(boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
        boxes2d_image[:, 2] = np.clip(boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 3] = np.clip(boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

    return boxes2d_image




