python3 main.py \
-mode test \
-load ../../model_weight/part_seg/0310_test.pkl \
-cuda 0 \
-epoch 20 \
-bs 4 \
-dataset ../../shapenetcore_partanno_segmentation_benchmark_v0 \
-point_num 1024 \
-support_num 1 \
-neighbor_num 50 \
#-output ../../seg_result/0311_axis1 \
#-random \
#-rotate 0 \
#-axis 1 \
#-scale 0.0 \
#-shift 0.0 \
#-normal