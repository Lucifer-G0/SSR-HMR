import os
import torch
import random
import argparse
import numpy as np

from net.loss import run_generator, run_ik
from net.skeleton_generator_architecture import Generator_Model
from net.motion_data import Train_Data, EvalMotionData
from net.ik_architecture import IK_Model
from net.config import param, xsens_parents
from pymotion.ops.skeleton_torch import from_root_dual_quat
from pymotion.ops.forward_kinematics_torch import fk
import matplotlib.pyplot as plt

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
    model_epoch = args.model_epoch
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

    eval_dataset = EvalMotionData(param, device)

    total_frame_num = 0
    # Eval Files
    for filename in eval_files:
        if filename[-4:] == ".npz":
            eval_dataset.add_motion(filename)

    # Create Models
    train_data = Train_Data(device, param)
    generator_model = Generator_Model(device, param, xsens_parents, train_data).to(device)
    ik_model = IK_Model(device, param, xsens_parents, train_data).to(device)
    # Load  Model
    load_model(generator_model, model_epoch, "generator", device)
    load_model(ik_model, model_epoch, "ik", device)
    results = run_generator(generator_model, train_data, eval_dataset)

    results_ik = run_ik(ik_model, results, train_data, eval_dataset)
    results = results_ik

    for step, result in enumerate(results):
        motion = eval_dataset.get_item(step)
        gt_offsets, gt_dqs, _, gt_joint_poses, gt_parents, fps = motion.values()
        gt_offsets = gt_offsets[:, 0].reshape(-1, 3)
        global_pos = gt_joint_poses[:, 0, :]  # 从相对于根的坐标系转换到全局坐标系

        res = result.permute(0, 2, 1).flatten(0, 1)
        dqs = res.reshape(res.shape[0], -1, 8)

        _, rots = from_root_dual_quat(dqs, gt_parents)
        rots = rots.to(device)
        joint_poses, pose_global_p = fk(rots, global_pos, gt_offsets, gt_parents)

        gt_dqs = gt_dqs.permute(1, 0)[42:]
        gt_dqs = gt_dqs.reshape(-1, 22, 8)

        _, gt_rots = from_root_dual_quat(gt_dqs, gt_parents)
        gt_rots = gt_rots.to(device)
        gt_joint_poses, pose_global_t = fk(gt_rots, global_pos, gt_offsets, gt_parents)

        # error
        joint_poses = joint_poses[:, param["sparse_joints"]]  # ignore root joint
        gt_joint_poses = gt_joint_poses[:, param["sparse_joints"]]  # ignore root joint

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_title("3D Trajectory")

        # 设置轴区间
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 2])

        # 设置刻度值
        # ticks = np.linspace(-1, 1, 5)  # 生成-1到1之间的5个刻度值
        # ax.set_xticks(ticks)
        # ax.set_yticks(ticks)
        # ax.set_zticks(ticks)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        joint_poses = joint_poses[:500].cpu().numpy()
        # 提取轨迹数据
        joints = {
            "Left Foot": joint_poses[: ,1, :],
            "Right Foot": joint_poses[:, 2, :],
            "Head": joint_poses[:, 3, :],
            "Left Hand": joint_poses[:, 4, :],
            "Right Hand": joint_poses[:, 5, :],
            # "Ground Truth": gt_joint_poses[:100, 1, :].cpu().numpy()
        }

        # 循环绘制每个关节的轨迹
        colors = ['r', 'g', 'b', 'c', 'm']  # 不同颜色
        for i, (joint_name, trajectory) in enumerate(joints.items()):
            x = trajectory[:, 0]
            y = trajectory[:, 1]
            z = trajectory[:, 2]
            ax.plot(x, y, z, label=joint_name, color=colors[i], marker='', linestyle='-')  # 添加颜色和标签

        # 添加图例
        ax.legend(fontsize=10)

        # 显示图形
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
        help="path to data directory containing one or multiple .bvh for training, last .bvh is used as test data",
    )

    parser.add_argument(
        "model_epoch",
        type=str,
        help="the epoch of the model you want to use, store at model/model_epoch/generator.pt&ik.pt",
    )

    args = parser.parse_args()

    main(args)
