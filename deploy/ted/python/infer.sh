python infer.py --model_file /home/yw/Paddle3D/output/ted.pdmodel \
    --params_file /home/yw/Paddle3D/output/ted.pdiparams \
    --lidar_file /home/yw/Paddle3D/000013.bin  \
    --point_cloud_range 0 -40 -3 70.4 40 1 \
    --num_point_dim 4 \
    --use_trt 0  \
    --trt_precision 0 \
    --trt_use_static 1 \
    --trt_static_dir /home/yw/Paddle3D/output \
    --collect_shape_info 1 \
    --dynamic_shape_file /home/yw/Paddle3D/output/ted.txt