python3 main.py \
-mode train \
-cuda 1 \
-epoch 100 \
-bs 8 \
-dataset ../../ModelNet40_pointcloud_1024 \
-save ../../model_weight/cls/0306_test.pkl \
-record 0306_test.txt \
-support_num 1 \
-neighbor_num 20 \
    