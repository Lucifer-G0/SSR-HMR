import os

import numpy as np
import torch
import torch.nn as nn

from motion.ops.skeleton_torch import from_root_dual_quat

from motion.ops.forward_kinematics_torch import fk
from motion.rotations import quat
from net.config import param, xsens_parents, S2X_map, SMPLPath
from motion.io.bvh import BVH


class MSE_DQ(nn.Module):
    def __init__(self, parents, device) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.parents = parents
        self.device = device

    def forward_generator(self, input, target_dqs):
        # Dual Quaternions MSE Loss
        loss_joints = self.mse(input[:, 8:, :], target_dqs[:, 8:, :])
        loss_root = self.mse(input[:, :8, :], target_dqs[:, :8, :])

        return loss_root * param["lambda_root"] + loss_joints


def run_generator(generator_model, train_data, dataset, sparse_motions=None):
    generator_model.eval()
    results = []
    with torch.no_grad():  # 在其内部的所有操作中禁用梯度计算，禁止计算图的构建，从而节省内存和加速运算
        for index in range(dataset.get_len()):
            # print("dataset item ", index, "--------------------------------")
            norm_motion = dataset.get_item(index)
            train_data.set_offsets(
                norm_motion["offsets"].unsqueeze(0),  # 在第0维添加一个维度，将张量转换为具有批量维度的形式
            )
            train_data.set_motions(
                norm_motion["dqs"].unsqueeze(0),
                norm_motion["sparse_dqs"].unsqueeze(0),
                None,
            )
            if sparse_motions is not None:
                train_data.set_sparse_motion(sparse_motions[index])
            res = generator_model.forward()
            results.append(res)
    return results


def eval_result_(results, eval_dataset, device):
    array_mpjpe = torch.empty((len(results),))
    array_mpeepe = torch.empty((len(results),))
    array_vel = torch.empty((len(results),))
    array_jitter = torch.empty((len(results),))
    for step, result in enumerate(results):
        motion = eval_dataset.get_item(step)
        # Evaluate Positional Error
        mpjpe, mpeepe, vel, jitter = eval_pos_error_(motion, result, device)

        array_mpjpe[step] = mpjpe
        array_mpeepe[step] = mpeepe
        array_vel[step] = vel
        array_jitter[step] = jitter

    return torch.mean(array_mpjpe), torch.mean(array_mpeepe), torch.mean(array_vel), torch.mean(array_jitter)


def eval_pos_error_(motion, result, device):
    gt_offsets, _, _,_,_, gt_joint_poses, gt_parents, fps = motion.values()
    gt_offsets = gt_offsets[:, 0].reshape(-1, 3)
    # 从相对于根的坐标系转换到全局坐标系
    global_pos = gt_joint_poses[:, 0, :]

    result = result.permute(0, 2, 1).flatten(0, 1)
    dqs = result.reshape(result.shape[0], -1, 8)
    # get rotations and translations from dual quatenions
    _, rots = from_root_dual_quat(dqs, gt_parents)
    rots = rots.to(device)
    joint_poses, _ = fk(rots, global_pos, gt_offsets, gt_parents)

    # error
    error = torch.norm(joint_poses - gt_joint_poses, dim=-1)
    sparse_error = error[:, param["sparse_joints"][1:]]  # ignore root joint

    # 计算Vel(1)，即相邻1帧间的速度
    vel_error_1 = torch.norm(gt_joint_poses[1:] - gt_joint_poses[:-1] - (joint_poses[1:] - joint_poses[:-1]), dim=-1)

    jitter = ((joint_poses[3:] - 3 * joint_poses[2:-1] + 3 * joint_poses[1:-2] - joint_poses[:-3]) * (
            fps ** 3)).norm(dim=-1)  # N, J

    return (torch.mean(error) * 100,  # *100 将单位从m转换为cm
            torch.mean(sparse_error) * 100,  # *100 将单位从m转换为cm
            torch.mean(vel_error_1) * fps * 100,  # *100 将单位从m转换为cm
            torch.mean(jitter) / 100)


def eval_result(results, eval_dataset, device):
    array_mpjpe = torch.empty((len(results),))
    array_std_mpjpe = torch.empty((len(results),))

    array_mpjre = torch.empty((len(results),))
    array_std_mpjre = torch.empty((len(results),))

    array_mpeepe = torch.empty((len(results),))
    array_std_mpeepe = torch.empty((len(results),))

    array_vel = torch.empty((len(results),))
    array_std_vel = torch.empty((len(results),))

    array_rjitter = torch.empty((len(results),))
    array_std_rjitter = torch.empty((len(results),))

    array_jitter = torch.empty((len(results),))
    array_std_jitter = torch.empty((len(results),))

    for step, result in enumerate(results):
        motion = eval_dataset.get_item(step)
        # Evaluate Positional Error
        mpjpe, std_mpjpe, mpeepe, std_mpeepe, vel, std_vel, raw_jitter, std_raw_jitter, jitter, std_jitter, mpjre, std_mpjre = eval_pos_error(
            motion, result, device)

        array_mpjpe[step] = mpjpe
        array_mpjre[step] = mpjre
        array_mpeepe[step] = mpeepe
        array_vel[step] = vel
        array_rjitter[step] = raw_jitter
        array_jitter[step] = jitter

        array_std_mpjpe[step] = std_mpjpe
        array_std_mpjre[step] = std_mpjre
        array_std_mpeepe[step] = std_mpeepe
        array_std_vel[step] = std_vel
        array_std_rjitter[step] = std_raw_jitter
        array_std_jitter[step] = std_jitter

    return (torch.mean(array_mpjpe), torch.mean(array_std_mpjpe),
            torch.mean(array_mpeepe), torch.mean(array_std_mpeepe),
            torch.mean(array_vel), torch.mean(array_std_vel),
            torch.mean(array_rjitter), torch.mean(array_std_rjitter),
            torch.mean(array_jitter), torch.mean(array_std_jitter),
            torch.mean(array_mpjre), torch.mean(array_std_mpjre))


def eval_pos_error(motion, result, device):
    gt_offsets, gt_dqs, _,gt_rots,gt_pos, gt_joint_poses, gt_parents, fps = motion.values()
    gt_offsets = gt_offsets[:, 0].reshape(-1, 3)
    global_pos = gt_pos  # 从相对于根的坐标系转换到全局坐标系

    res = result.permute(0, 2, 1).flatten(0, 1)
    dqs = res.reshape(res.shape[0], -1, 8)

    _, rots = from_root_dual_quat(dqs, gt_parents)
    rots = rots.to(device)
    joint_poses, pose_global_p = fk(rots, global_pos, gt_offsets, gt_parents)

    # gt_dqs = gt_dqs.permute(1, 0)[42:]
    # gt_dqs = gt_dqs.reshape(-1, 22, 8)
    # _, gt_rots = from_root_dual_quat(gt_dqs, gt_parents)
    # gt_rots = gt_rots.to(device)
    _, pose_global_t = fk(gt_rots, global_pos, gt_offsets, gt_parents)

    # error
    error = torch.norm(joint_poses - gt_joint_poses, dim=-1)
    sparse_error = error[:, param["sparse_joints"][1:]]  # ignore root joint

    ae = radian_to_degree(angle_between(pose_global_p, pose_global_t))

    # 计算Vel(1)，即相邻1帧间的速度
    vel_error_1 = torch.norm(gt_joint_poses[1:] - gt_joint_poses[:-1] - (joint_poses[1:] - joint_poses[:-1]), dim=-1)

    rjitter = ((gt_joint_poses[3:] - 3 * gt_joint_poses[2:-1] + 3 * gt_joint_poses[1:-2] - gt_joint_poses[:-3]) * (
            fps ** 3)).norm(dim=-1)  # N, J

    jitter = ((joint_poses[3:] - 3 * joint_poses[2:-1] + 3 * joint_poses[1:-2] - joint_poses[:-3]) * (
            fps ** 3)).norm(dim=-1)  # N, J

    return (torch.mean(error) * 100, error.std(dim=0).mean() * 100,  # *100 将单位从m转换为cm
            torch.mean(sparse_error) * 100, sparse_error.std(dim=0).mean() * 100,  # *100 将单位从m转换为cm
            torch.mean(vel_error_1) * fps * 100, vel_error_1.std(dim=0).mean() * fps * 100,  # *100 将单位从m转换为cm
            torch.mean(rjitter) / 100, rjitter.std(dim=0).mean() / 100,
            torch.mean(jitter) / 100, jitter.std(dim=0).mean() / 100,
            torch.mean(ae), ae.std(dim=0).mean())


def radian_to_degree(q):
    r"""
    Convert radians to degrees.
    """
    return q * 180.0 / np.pi


def angle_between(rot1: torch.Tensor, rot2: torch.Tensor):
    r"""
    Calculate the angle in radians between two rotations. (torch, batch)

    :param rot1: Rotation tensor 1 that can reshape to [batch_size, rep_dim].
    :param rot2: Rotation tensor 2 that can reshape to [batch_size, rep_dim].
    :param rep: The rotation representation used in the input.
    :return: Tensor in shape [batch_size] for angles in radians.
    """
    rot1 = rot1.reshape(-1, 3, 3)
    rot2 = rot2.reshape(-1, 3, 3)
    offsets = rot1.transpose(-2, -1).bmm(rot2)
    angles = rotation_matrix_to_axis_angle(offsets).norm(dim=1)
    return angles


def rotation_matrix_to_axis_angle(r: torch.Tensor):
    r"""
    Turn rotation matrices into axis-angles. (torch, batch)

    :param r: Rotation matrix tensor that can reshape to [batch_size, 3, 3].
    :return: Axis-angle tensor of shape [batch_size, 3].
    """
    import cv2
    result = [cv2.Rodrigues(_)[0] for _ in r.clone().detach().cpu().view(-1, 3, 3).numpy()]
    result = torch.from_numpy(np.stack(result)).float().squeeze(-1).to(r.device)
    return result


def get_info_from_bvh(filename, device):
    bvh = BVH()
    bvh.load(filename)
    rot_roder = np.tile(bvh.data["rot_order"], (bvh.data["rotations"].shape[0], 1, 1))
    rots = quat.unroll(
        quat.from_euler(np.radians(bvh.data["rotations"]), order=rot_roder),
        axis=0,
    )
    rots = quat.normalize(rots)  # make sure all quaternions are unit quaternions
    pos = bvh.data["positions"][:,0,:]
    parents = bvh.data["parents"]
    parents[0] = 0  # BVH sets root as None
    offsets = bvh.data["offsets"]
    offsets[0] = np.zeros(3)  # force to zero offset for root joint
    fps = np.array(np.round(1 / bvh.data["frame_time"]))
    if fps == 120:
        rots = rots['poses'][::2]
        pos = pos['trans'][::2]
        fps = 60
    elif fps == 60:
        pass
    else:
        print("fps error ", fps)
        exit(111)

    rots = torch.from_numpy(rots).type(torch.float32).to(device)
    pos = torch.from_numpy(pos).type(torch.float32).to(device)
    parents = torch.Tensor(parents).long().to(device)
    offsets = torch.from_numpy(offsets).type(torch.float32).to(device)
    fps = torch.from_numpy(fps).float().to(device)
    return rots, pos, parents, offsets, fps


def get_info_from_npz(filename, device, step=2):
    """
        return:
            均为numpy数组
            rots: (frames,n_joints,4) 表示旋转的单位四元数
            pos: (frames,3) 表示根节点位移
            parents:    xsens 架构的父节点关系
            offsets:    来自SMPL的形状参数，仅区分男女
    """
    s2x_map = np.array(S2X_map)  # SMPL 和 xsens 骨架之间的映射关系 (parents)

    motion_data = np.load(filename)
    gender = motion_data['gender'].item().upper()
    rots = motion_data['poses'][::step].reshape(-1, 52, 3)
    pos = motion_data['trans'][::step]
    fps = motion_data.get('mocap_frame_rate', motion_data.get('mocap_framerate', None))

    # get skeleton graph from smpl model file (offsets)
    smpl_name = os.path.join(SMPLPath, 'SMPL_' + gender + '.npz')
    smpl_model = np.load(smpl_name)
    root0J = smpl_model["J"] - smpl_model["J"][0]
    kintree_table = smpl_model["kintree_table"]
    # 计算offsets
    offsets = np.zeros_like(root0J)
    for i in range(1, root0J.shape[0]):
        parent_idx = kintree_table[0, i]
        offsets[i] = root0J[i] - root0J[parent_idx]

    # 将SMPL结构的数据转换为训练需要的数据
    offsets = offsets[s2x_map, :]
    rots = rots[:, s2x_map, :]
    parents = xsens_parents

    # 将rots从SMPL轴角表示转为四元数
    rots = quat.unroll(
        axis_angle_to_quaternion(rots),
        axis=0,
    )
    rots = quat.normalize(rots)  # make sure all quaternions are unit quaternions

    rots = torch.from_numpy(rots).type(torch.float32).to(device)
    pos = torch.from_numpy(pos).type(torch.float32).to(device)
    parents = torch.Tensor(parents).long().to(device)
    offsets = torch.from_numpy(offsets).type(torch.float32).to(device)
    fps = torch.from_numpy(fps).float().to(device) / 2

    return rots, pos, parents, offsets, fps


def axis_angle_to_quaternion(axis_angle: np.array) -> np.array:
    """
    Convert SMPL's axis-angle representation to quaternion.

    Parameters
    ----------
    axis_angle : np.array[..., 3]
        Axis-angle representation where each row is [r_x, r_y, r_z] for a rotation axis.

    Returns
    -------
    quat : np.array[..., 4]
        Quaternion representation of the same rotation.
    """
    # 计算旋转的角度（模长）
    angle = np.linalg.norm(axis_angle, axis=-1, keepdims=True)

    # 判断旋转轴是否为零向量
    zero_mask = angle == 0  # 如果旋转轴是零向量，返回 True

    # 归一化旋转轴，避免除零
    axis = np.where(zero_mask, np.zeros_like(axis_angle), axis_angle / angle)  # 如果是零向量，保留为零向量

    c = np.cos(angle / 2.0)
    s = np.sin(angle / 2.0)
    return np.concatenate((c, s * axis), axis=-1)


class MSE_DQ_FK(nn.Module):
    def __init__(self, param, parents, device) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
        self.param = param
        self.parents = torch.Tensor(parents).long().to(device)
        self.device = device

        # indices without sparse input
        self.indices_no_sparse = []
        for i in range(0, 22):
            if i not in self.param["sparse_joints"]:
                self.indices_no_sparse.append(i)
        self.indices_no_sparse = torch.tensor(self.indices_no_sparse).to(self.device)

    def set_offsets(self, offsets):
        self.offsets = offsets  # denormalized

    def forward_ik(self, ik_res, gt_result):
        # ground truth
        target_joint_poses, target_joint_rot_mat = gt_result
        global_pos = target_joint_poses[:, :, 0, :]

        # compute final positions (root space) using standard FK
        ik_res = ik_res.reshape((ik_res.shape[0], -1, 8, ik_res.shape[-1])).permute(0, 3, 1, 2)
        _, local_rot = from_root_dual_quat(ik_res, self.parents)
        joint_poses, joint_rot_mat = fk(
            local_rot,
            global_pos,
            self.offsets, self.parents,
        )

        # positions error
        loss_ee = self.mse(joint_poses[:, :, self.param["sparse_joints"][1:], :],
                           target_joint_poses[:, :, self.param["sparse_joints"][1:], :])
        # rotations error
        loss_ee += self.mse(joint_rot_mat[:, :, self.param["sparse_joints"][1:], :, :],
                            target_joint_rot_mat[:, :, self.param["sparse_joints"][1:], :, :])

        # regularization
        loss_ee_reg = self.mse(target_joint_poses[:, :, self.indices_no_sparse, :],
                               joint_poses[:, :, self.indices_no_sparse, :], )
        loss_ee_reg += self.mse(target_joint_rot_mat[:, :, self.indices_no_sparse, :, :],
                                joint_rot_mat[:, :, self.indices_no_sparse, :, :], )

        # 计算Vel(1)，即相邻1帧间的速度
        vel_error_1 = torch.norm(
            target_joint_poses[:, 1:] - target_joint_poses[:, :-1] - (joint_poses[:, 1:] - joint_poses[:, :-1]), dim=-1)

        # 计算Vel(3)，即相邻3帧间的速度
        vel_error_3 = torch.norm(
            target_joint_poses[:, 3:] - target_joint_poses[:, :- 3] - (joint_poses[:, 3:] - joint_poses[:, :- 3]),
            dim=-1)

        # 计算Vel(5)，即相邻5帧间的速度
        vel_error_5 = torch.norm(
            target_joint_poses[:, 5:] - target_joint_poses[:, :- 5] - (joint_poses[:, 5:] - joint_poses[:, :- 5]),
            dim=-1)

        # 将Vel(1), Vel(3), Vel(5)相加
        loss_vel = torch.mean(vel_error_1) + torch.mean(vel_error_3) + torch.mean(vel_error_5)

        # return loss_ee * self.param["lambda_ee"] + 10 * loss_vel
        return (loss_ee * self.param["lambda_ee"]
                + loss_ee_reg * self.param["lambda_ee_reg"]
                + 10 * loss_vel)


def run_ik(ik_model, results_decoder, train_data, dataset):
    # WARNING: means and stds for the model are not set in this function... they should be set before
    ik_model.eval()
    results = []
    with torch.no_grad():
        for index in range(dataset.get_len()):
            norm_motion = dataset.get_item(index)
            train_data.set_offsets(
                norm_motion["offsets"].unsqueeze(0),
            )
            train_data.set_motions(
                norm_motion["dqs"].unsqueeze(0),
                norm_motion["sparse_dqs"].unsqueeze(0),
                None,
            )
            res = ik_model.forward(results_decoder[index])
            results.append(res)
    return results
