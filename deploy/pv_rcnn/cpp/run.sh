./build/main --model_file /home/yw/Paddle3D/output/pv_rcnn.pdmodel  \
--params_file /home/yw/Paddle3D/output/pv_rcnn.pdiparams \
--lidar_file /mnt/8tssd/kitti/detection/training/velodyne/000002.bin \
--num_point_dim 4 \
--point_cloud_range "0 -40 -3 70.4 40 1" \
--use_trt 1 \
--trt_precision 0 \
--trt_use_static 0 \
--trt_static_dir /home/yw/Paddle3D/output \
--collect_shape_info 1 \
--dynamic_shape_file /home/yw/Paddle3D/output