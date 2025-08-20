import os
import torch
import random
import argparse
import numpy as np

from net.loss import run_generator, eval_result, run_ik, axis_angle_to_quaternion
from net.skeleton_generator_architecture import Generator_Model
from net.motion_data import Train_Data, EvalMotionData
from net.ik_architecture import IK_Model
from net.config import param, xsens_parents
from motion.ops.skeleton_torch import to_root_dual_quat, from_root_dual_quat


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

    # 输入：数据集根目录
    root_dir = eval_dir  # 替换为你的数据集根目录
    subsampled_folder_name = "../subsampled_results"
    predicted_folder_name = "../predicted_results"

    # 创建目标文件夹
    subsampled_folder = create_folder_structure(root_dir, subsampled_folder_name)
    predicted_folder = create_folder_structure(root_dir, predicted_folder_name)

    eval_files = []
    for root, dirs, files in os.walk(eval_dir):
        for file in files:
            # 检查文件是否以.npz结尾，且文件名不是shape.npz
            if file.endswith('.bvh'):
                full_path = os.path.join(root, file)
                eval_files.append(full_path)

    eval_dataset = EvalMotionData(param, device)

    # Eval Files
    for filename in eval_files:
        eval_dataset.add_motion(filename)  # 如需调整采样频率，此处更改，否则默认120下采样为60

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

    result = results[0]
    motion = eval_dataset.get_item(0)
    gt_offsets, _, _, gt_rots,gt_pos,_, gt_parents, fps = motion.values()

    res = result.permute(0, 2, 1).flatten(0, 1)
    dqs = res.reshape(res.shape[0], -1, 8)
    _, rots = from_root_dual_quat(dqs, gt_parents)
    rotations = quaternion_to_axis_angle(rots)
    gt_rotations = quaternion_to_axis_angle(gt_rots)

    motion_data = np.load("walking2_poses.npz")
    motion_data = dict(motion_data)

    # 确保数据形状一致
    min_frames = min(gt_rotations.shape[0],  motion_data['poses'].shape[0])

    motion_data['poses'] = motion_data['poses'][:min_frames]
    motion_data['trans'] = motion_data['trans'][:min_frames]
    motion_data['dmpls'] = motion_data['dmpls'][:min_frames]
    motion_data['mocap_framerate'] = 60
    motion_data['trans'] = gt_pos[:min_frames].cpu().numpy()

    # 反向映射并更新
    s2x_map = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]

    motion_data['poses'] = motion_data['poses'].reshape(-1, 52, 3)
    for i, smpl_idx in enumerate(s2x_map):
        motion_data['poses'][:, smpl_idx, :] = gt_rotations[:min_frames, i, :]
    motion_data['poses'] = motion_data['poses'].reshape(-1, 156)

    # 保存抽帧结果到 subsampled_results 文件夹
    save_motion_data(motion_data, filename, root_dir, subsampled_folder)
    print("save ", motion_data['poses'].shape[0], " frames to ", subsampled_folder_name, " fps=",
          motion_data['mocap_framerate'])

    motion_data['poses'] = motion_data['poses'].reshape(-1, 52, 3)
    # 反向映射并更新预测结果
    s2x_map = [0, 1, 4, 7, 10, 2, 5, 8, 11, 3, 6, 9, 12, 15, 13, 16, 18, 20, 14, 17, 19, 21]
    for i, smpl_idx in enumerate(s2x_map):
        motion_data['poses'][:, smpl_idx, :] = rotations[:min_frames, i, :]

    motion_data['poses'] = motion_data['poses'].reshape(-1, 156)

    # 保存预测结果到 predicted_results 文件夹
    save_motion_data(motion_data, filename, root_dir, predicted_folder)
    print("save ", motion_data['poses'].shape[0], " frames to ", predicted_folder_name, " fps=",
          motion_data['mocap_framerate'])

def create_folder_structure(root_dir, new_folder_name):
    """
    在指定根目录下创建一个新的文件夹，并保持原始文件夹结构。
    """
    new_folder_path = os.path.join(root_dir, new_folder_name)
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)
    return new_folder_path


def save_motion_data(data, filename, root_dir, target_folder):
    """
    将数据保存到目标文件夹中，保持原始文件结构。
    """
    # 获取文件的相对路径
    relative_path = os.path.relpath(filename, root_dir)
    target_file_path = os.path.join(target_folder, relative_path)

    # 确保目标文件夹存在
    os.makedirs(os.path.dirname(target_file_path), exist_ok=True)

    # 保存数据
    np.savez(target_file_path, **data)


def save_rots_to_npz(filename, rots, original_filename):
    """
    将 rots 保存到新的 .npz 文件中，同时保留原始文件的其他信息。

    参数:
        filename: 新文件的保存路径。
        rots: 要保存的旋转数据（四元数形式）。
        original_filename: 原始文件的路径，用于提取其他信息。
    """
    # 加载原始文件的数据
    motion_data = np.load(original_filename)

    # 将 rots 从 torch.Tensor 转换为 numpy 数组
    rots_np = rots.cpu().numpy()

    # 将 rots 从四元数转换回轴角表示（如果需要）
    # 这里假设原始文件中的 poses 是轴角表示
    rots_axis_angle = quaternion_to_axis_angle(rots_np)

    # 更新 poses 数据
    motion_data_dict = dict(motion_data)
    motion_data_dict['poses'] = rots_axis_angle

    # 保存到新文件
    np.savez(filename, **motion_data_dict)


def quaternion_to_axis_angle(quaternions):
    """
    将四元数转换为轴角表示。
    """
    quaternions = torch.from_numpy(quaternions) if isinstance(quaternions, np.ndarray) else quaternions
    norm = torch.norm(quaternions[..., 1:], dim=-1, keepdim=True)
    angle = 2 * torch.atan2(norm, quaternions[..., 0:1])

    # 处理 norm == 0 的情况
    zero_mask = torch.isclose(norm, torch.zeros_like(norm))
    axis = torch.where(zero_mask, torch.zeros_like(quaternions[..., 1:]), quaternions[..., 1:] / norm)

    axis_angle = axis * angle
    return axis_angle.cpu().numpy()


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
