python3 main.py \
-mode test \
-cuda 1 \
-bs 8 \
-dataset ../../ModelNet40_pointcloud_1024 \
-load ../../model_weight/cls/0306_test.pkl \
-support_num 1 \
-neighbor_num 20 \
-shift 0.0 \
-scale 1.0 \
-rotate 1 \
#-normal \
-random \