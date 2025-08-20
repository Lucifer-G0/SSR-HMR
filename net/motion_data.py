import torch
from motion.ops.forward_kinematics_torch import fk
from torch.utils.data import Dataset
import motion.rotations.dual_quat_torch as dquat
from motion.ops.skeleton_torch import to_root_dual_quat, from_root_dual_quat
from net.loss import get_info_from_npz, get_info_from_bvh


class Train_Data:
    def __init__(self, device, param):
        super().__init__()
        self.loss = None
        self.device = device
        self.param = param

    def set_offsets(self, offsets):
        self.offsets = offsets
        self.loss.set_offsets(offsets[0, :, 0].reshape(22, 3))

    def set_motions(self, dqs, sparse_dqs, gt_result):
        # self.motion = dqs.clone()
        # swap second and third dimensions for convolutions (last row should be time)
        self.dqs = dqs
        self.sparse_dqs = sparse_dqs
        self.gt_result = gt_result

    def set_sparse_motion(self, sparse_dqs):
        self.sparse_dqs = sparse_dqs


class TrainMotionData(Dataset):
    def __init__(self, param, device):
        self.motions = []
        self.param = param
        self.device = device

    def add_motion(self, offsets, global_pos, rotations, parents):
        """
        Parameters:
        -----------
        offsets: torch of shape (n_joints, 3)
        global_pos: torch of shape (n_frames, 3)
        rotations: torch of shape (n_frames, n_joints, 4) (quaternions)
        parents: torch of shape (n_joints)

        Returns:
        --------
        self.motions:
            offsets: tensor of shape (n_joints, 3)
            dqs: tensor of shape (windows_size, n_joints * 8) (dual quaternions)
        """
        frames = rotations.shape[0]
        # create dual quaternions
        fake_global_pos = torch.zeros((frames, 3))
        dqs = to_root_dual_quat(rotations, global_pos, parents, offsets)
        dqs = dquat.unroll(dqs, dim=0).to(self.device)  # ensure continuity (T,22,8)
        _, gt_local_rot = from_root_dual_quat(dqs, parents)
        # target_joint_poses, target_joint_rot_mat (T,22,3),(T,22,3,3)
        gt_result = fk(gt_local_rot, global_pos, offsets, parents)

        dqs = torch.flatten(dqs, 1, 2).permute(1, 0)  # (22*8,T)
        sparse_dqs = dqs.reshape(-1, 8, dqs.shape[1])
        sparse_dqs = sparse_dqs[self.param["sparse_joints"], :, :]
        sparse_dqs = sparse_dqs.flatten(start_dim=0, end_dim=1)

        # 只保留手头
        # for i in [0, 1, 2]:
        #     sparse_dqs[i * 8:(i + 1) * 8, :] = sparse_dqs[3*8:4*8, :]

        offsets = torch.tile(offsets.flatten(start_dim=0, end_dim=1).unsqueeze(-1), (1, dqs.shape[-1]))

        for start in range(0, frames, self.param["window_step"]):
            end = start + self.param["window_size"]
            if end < frames:
                motion = {
                    "offsets": offsets[:, start + 42:end],
                    "dqs": dqs[:, start:end],
                    "sparse_dqs": sparse_dqs[:, start:end],
                    "gt_result": (gt_result[0][start + 42:end], gt_result[1][start + 42:end]),  # 三层卷积后去掉前(15-1)*3帧
                }
                self.motions.append(motion)

    def __len__(self):
        return len(self.motions)

    def __getitem__(self, index):
        return self.motions[index]


class EvalMotionData:
    def __init__(self, param, device):
        self.motions = []
        self.filenames = []
        self.param = param
        self.device = device

    def add_motion(self, filename):
        rotations, global_pos, parents, offsets, fps = get_info_from_bvh(filename, self.device)

        dqs = to_root_dual_quat(rotations, global_pos, parents, offsets)
        dqs = dquat.unroll(dqs, dim=0)  # ensure continuity
        dqs = torch.flatten(dqs, 1, 2).permute(1, 0)

        sparse_dqs = dqs.reshape(-1, 8, dqs.shape[1])
        sparse_dqs = sparse_dqs[self.param["sparse_joints"], :, :]
        sparse_dqs = sparse_dqs.flatten(start_dim=0, end_dim=1)
        # 只保留手头
        # for i in [0, 1, 2]:
        #     sparse_dqs[i * 8:(i + 1) * 8, :] = sparse_dqs[3*8:4*8, :]   # 下肢与头部一致？

        # caculate joint position
        gt_rots = rotations[42:, :, :]
        gt_pos = global_pos[42:, :]
        gt_joint_poses, _ = fk(gt_rots, gt_pos, offsets, parents)

        offsets = torch.tile(offsets.flatten(start_dim=0, end_dim=1).unsqueeze(-1), (1, dqs.shape[-1] - 42))

        motion = {
            "offsets": offsets,
            "dqs": dqs,
            "sparse_dqs": sparse_dqs,
            "gt_rots": gt_rots,
            "gt_pos": gt_pos,
            "gt_joint_poses":gt_joint_poses,
            "parents": parents,
            "fps": fps
        }
        self.motions.append(motion)
        self.filenames.append(filename)

    def get_len(self):
        return len(self.motions)

    def get_item(self, index):
        return self.motions[index]
