python3 main.py \
-mode test \
-support 1 \
-neighbor 50 \
-load model.pkl \
-cuda 0 \
-bs 4 \
-dataset shapenetcore_partanno_segmentation_benchmark_v0 \
-point 1024 \
-output out_imgs/ \
#-random \
#-rotate 0 \
#-axis 1 \
#-scale 0.0 \
#-shift 0.0 \