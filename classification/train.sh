python3 main.py \
-mode train \
-cuda 0 \
-epoch 100 \
-bs 8 \
-dataset /mnt/HDD_1/j1a0m0e4s/ModelNet40_pointcloud_1024 \
-save ../../model_weight/cls/0306_dgcnn.pkl \
-record 0306_dgcnn.txt \
-support_num 1 \
-neighbor_num 20 \
-normal \
    