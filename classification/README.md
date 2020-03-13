# Classification on ModelNet

<img src="../imgs/model_cls.png" alt="classification model" width="500" />

-random
           original | shift  1 | shift  5 | shift 10 | scale 0.5 | scale  2 | rotate 30 | rotate 45 | rotate 90 | rotate 180
PointNet |    0.884 |    0.044 |    0.024 |    0.027 |     0.546 |    0.237 |     0.648 |     0.482 |     0.314 |      0.242
DGCNN    |    0.892 |    0.324 |    0.021 |    0.158 |     0.128 |    0.613 |     0.263 |     0.296 |     0.285 |      0.275
GCN3D    |    0.915 |    0.906 |    0.903 |    0.904 |     0.903 |    0.908 |     0.756 |     0.605 |     0.379 |      0.312