import torch

param = {
    "batch_size": 256,
    "epochs": 2000,
    "kernel_size_temporal_dim": 15,
    "neighbor_distance": 2,
    "stride_encoder_conv": 2,
    "learning_rate": 1e-3,
    "lambda_root": 10,
    "lambda_ee": 10,
    "lambda_ee_reg": 1,
    "sparse_joints": [
        0,  # first should be root (as assumed by loss.py)
        3,  # left foot
        7,  # right foot
        13,  # head
        17,  # left hand
        21,  # right hand
    ],
    "window_size": 64,
    "window_step": 16,
    "seed": 2222,
}

# assert param["kernel_size_temporal_dim"] % 2 == 1
xsens_parents = [0, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]
# SMPL 和 xsens 骨架之间的映射关系 map[xsens]=smpl
S2X_map = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]
"""
    AMASS数据集中，旋转存储在poses,为弧度制；52关节，前24关节为身体关节；位移存储在trans,单位为米；
"""

SMPLPath = "model/SMPL/" # 相对于运行文件的SMPL模型文件的路径
