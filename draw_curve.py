import os
import torch
import random
import argparse
import numpy as np
from matplotlib import pyplot as plt

from net.loss import  get_info_from_npz
from net.motion_data import  EvalMotionData
from net.config import param
from pymotion.ops.forward_kinematics_torch import fk

# 为评估定制，统计每个文件的帧率和时长，考虑到帧率有可能不同，用帧的数量除以帧率得到时长
# 无法定制，每个数据集文件名定义不一样
# 这里面仍有评估参数计算而不是完全的推理，应该删减，或者另外制作推理代码统计时间
def main(args):
    # Set seed
    torch.manual_seed(param["seed"])
    random.seed(param["seed"])
    np.random.seed(param["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    # Prepare Data
    eval_dir = args.data_path
    # check if train and eval directories exist
    if not os.path.exists(eval_dir):
        raise ValueError("eval directory does not exist")

    eval_files = []
    for root, dirs, files in os.walk(eval_dir):
        for file in files:
            # 检查文件是否以.npz结尾，且文件名不是shape.npz
            if file.endswith('.npz') and file != 'shape.npz':
                full_path = os.path.join(root, file)
                eval_files.append(full_path)

    # 初始化存储结果的列表
    velocities = []  # 用于存储每个文件的速度
    jitters = []  # 用于存储每个文件的抖动
    labels = []  # 用于存储每个文件的标签（文件名）

    # 遍历所有文件并计算速度和抖动
    for filename in eval_files:
        if filename[-4:] == ".npz":
            rotations, global_pos, parents, offsets, fps = get_info_from_npz(filename, device)
            gt_joint_poses, _ = fk(rotations, global_pos, offsets, parents)
            gt_joint_poses=gt_joint_poses[600:1600]

            # 计算速度 Vel(1)
            vel = torch.norm(gt_joint_poses[1:] - gt_joint_poses[:-1], dim=-1).mean(dim=1).cpu().numpy() * 100

            # 计算抖动 Jitter
            jitter = ((gt_joint_poses[3:] - 3 * gt_joint_poses[2:-1] + 3 * gt_joint_poses[1:-2] - gt_joint_poses[
                                                                                                  :-3]) * (
                                  fps ** 3)).norm(dim=-1).mean(dim=1).cpu().numpy() / 100

            # 将结果存储到列表中
            velocities.append(vel)
            jitters.append(jitter)
            labels.append(filename.split('/')[-1])  # 使用文件名作为标签

    import matplotlib.pyplot as plt

    # 假设 velocities 和 jitters 是之前计算好的速度和抖动数据，labels 是文件名列表
    # velocities = [vel1, vel2, ...]，其中 vel1 是每个文件的速度数据
    # jitters = [jitter1, jitter2, ...]，其中 jitter1 是每个文件的抖动数据

    # 定义分段数量
    num_segments = 30

    # 初始化存储每个文件的分段均值
    velocity_means = []  # 存储每个文件的速度分段均值
    jitter_means = []  # 存储每个文件的抖动分段均值

    # 计算每个文件的分段均值
    for vel, jitter in zip(velocities, jitters):
        # 将数据划分为num_segments个区域
        segment_size = len(vel) // num_segments

        # 计算分段均值，丢弃最后一段（余数部分）
        vel_means = [np.mean(vel[i:i + segment_size]) for i in
                     range(0, len(vel) - (len(vel) % num_segments), segment_size)]
        jitter_means_list = [np.mean(jitter[i:i + segment_size]) for i in
                             range(0, len(jitter) - (len(jitter) % num_segments), segment_size)]

        # 将结果存储到列表中
        velocity_means.append(vel_means)
        jitter_means.append(jitter_means_list)

    # 计算每个文件的总面积
    velocity_areas = [np.sum(vel_means_list) for vel_means_list in velocity_means]
    jitter_areas = [np.sum(jitter_means_list) for jitter_means_list in jitter_means]

    # 定义颜色方案
    colors = ["green", "blue", "orange", "gray"]  # 指定颜色

    # 绘制柱形图
    x = np.arange(num_segments)  # x轴的分段位置
    bar_width = 1.0  # 柱形宽度，确保柱形之间没有间隔

    plt.figure(figsize=(12, 6))

    # 绘制速度的分段均值柱形图
    plt.subplot(2, 1, 1)
    for i, (means, label, area, color) in enumerate(zip(velocity_means, labels, velocity_areas, colors)):
        plt.bar(x, means, width=bar_width, label=f'{label} (Area: {area:.2f})', color=color, alpha=0.5,
                edgecolor="white")
    plt.title('Segmented Mean Velocity')
    plt.xlabel('Segment Index')
    plt.ylabel('Mean Velocity')
    plt.legend()
    plt.grid(True)

    # 绘制抖动的分段均值柱形图
    plt.subplot(2, 1, 2)
    for i, (means, label, area, color) in enumerate(zip(jitter_means, labels, jitter_areas, colors)):
        plt.bar(x, means, width=bar_width, label=f'{label} (Area: {area:.2f})', color=color, alpha=0.5,
                edgecolor="white")
    plt.title('Segmented Mean Jitter')
    plt.xlabel('Segment Index')
    plt.ylabel('Mean Jitter')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def load_model(model, model_epoch, model_name, device):
    model_path = os.path.join("./model", model_epoch, model_name + ".pt")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Motion Upsampling Network")
    parser.add_argument(
        "data_path",
        type=str,
        help="path to data directory containing one or multiple .npz for draw",
    )


    args = parser.parse_args()

    main(args)
