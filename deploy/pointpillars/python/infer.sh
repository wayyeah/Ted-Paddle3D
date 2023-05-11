python infer.py --model_file /home/yw/Paddle3D/output/pointpillars.pdmodel \
    --params_file /home/yw/Paddle3D/output/pointpillars.pdiparams \
    --lidar_file /home/yw/CUDA-PointPillars/mdata/000001.bin  \
    --point_cloud_range 0 -39.68 -3 69.12 39.68 1 \
    --voxel_size .16 .16 4 \
    --max_points_in_voxel 32 \
    --max_voxel_num 40000 \
    --use_trt 1  \
    --trt_precision 0 \
    --trt_use_static 1 \
    --trt_static_dir /home/yw/Paddle3D/output \
    --collect_shape_info 1 \
    --dynamic_shape_file /home/yw/Paddle3D/output

