python infer.py --model_file /home/yw/Paddle3D/output/centerpoint.pdmodel \
    --params_file /home/yw/Paddle3D/output/centerpoint.pdiparams \
    --lidar_file /mnt/10tdata/xqm/nuscenes_mini/samples/LIDAR_TOP/n015-2018-11-21-19-38-26+0800__LIDAR_TOP__1542801002447972.pcd.bin  \
    --num_point_dim 5 \
    --use_trt 1  \
    --trt_precision 1 \
    --trt_use_static 1 \
    --trt_static_dir /home/yw/Paddle3D/output/ \
    --collect_shape_info 1 \
    --dynamic_shape_file /home/yw/Paddle3D/output/