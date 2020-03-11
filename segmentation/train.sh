python3 main.py \
-mode train \
-save ../../model_weight/part_seg/0310_test.pkl \
-cuda 0 \
-epoch 20 \
-bs 1 \
-dataset ../../shapenetcore_partanno_segmentation_benchmark_v0 \
-point_num 1024 \
-record 0310_test.log \
-interval 1000 \
-support_num 1 \
-neighbor_num 50 \
-output ../../seg_result/0310_test \
# -normal 
