import logging
import os
import pickle
import random
import shutil
import subprocess

import numpy as np
import paddle
import paddle.distributed as dist


def check_numpy_to_paddle(x):
    if isinstance(x, np.ndarray):
        return paddle.to_tensor(x).astype(paddle.float32), True
    return x, False


def limit_period(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_paddle(val)
    ans = val - paddle.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans

def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask
def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    
    
    voxel_centers = paddle.stack([voxel_coords[:, 2], voxel_coords[:, 1], voxel_coords[:, 0]], axis=1)
    voxel_size = paddle.to_tensor(voxel_size).astype('float32') * downsample_times
    pc_range = paddle.to_tensor(point_cloud_range[0:3]).astype('float32')
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers
def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger
def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    paddle.framework.set_cudnn_deterministic(True)
def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds
def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    Args:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = paddle.device.cuda.device_count()
    paddle.device.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['PADDLE_TRAINERS_NUM'] = str(ntasks)
    os.environ['PADDLE_TRAINER_ID'] = str(proc_id)
    dist.init_parallel_env(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank

def init_dist_paddle(tcp_port, local_rank, backend='nccl'):
    num_gpus = paddle.device.cuda.device_count()
    paddle.device.cuda.set_device(local_rank % num_gpus)
    dist.init_parallel_env(
        backend=backend,
        init_method='tcp://127.0.0.1:%d' % tcp_port,
        rank=local_rank,
        world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank

def get_dist_info():
    initialized = dist.is_initialized()
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size

def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results

