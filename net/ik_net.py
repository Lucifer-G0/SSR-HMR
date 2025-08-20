import torch
import torch.nn as nn
import pymotion.rotations.dual_quat_torch as dquat
from torch import jit


class IK_NET(nn.Module):
    def __init__(self, param, parents, device):
        super().__init__()

        self.param = param
        self.parents = parents
        self.device = device

        channels_per_joint = 8
        channels_per_offset = 3
        hidden_size = 128

        # offsets leg: LowerLeg, Foot, Toe  # offsets arm: UpperArm, LowerArm, Hand
        # dq leg: UpperLeg, LowerLeg, Foot  # dq arm: Shoulder, UpperArm, LowerArm

        # main_body
        main_offsets = [10, 11, 12, 13]
        main_dq_in = [12]  # the end of input joint path
        self.main_dq_out = [10, 11, 12]
        # Left Leg ----------------------------------------
        l_leg_offsets = [2, 3, 4]
        l_leg_dq_in = [3]
        self.l_leg_dq_out = [1, 2, 3]
        # Right Leg ----------------------------------------
        r_leg_offsets = [6, 7, 8]
        r_leg_dq_in = [7]
        self.r_leg_dq_out = [5, 6, 7]
        # Left Arm -------M---------------------------------
        l_arm_offsets = [15, 16, 17]
        l_arm_dq_in = [16]
        self.l_arm_dq_out = [14, 15, 16]
        # Right Arm ----------------------------------------
        r_arm_offsets = [19, 20, 21]
        r_arm_dq_in = [20]
        self.r_arm_dq_out = [18, 19, 20]

        self.mainNet = TrunkNet(channels_per_offset, channels_per_joint, hidden_size, parents, device)
        self.llegNet = LimbNet(channels_per_offset, channels_per_joint, hidden_size, parents,
                               extend_channels([0], channels_per_joint), l_leg_offsets, l_leg_dq_in, device)
        self.rlegNet = LimbNet(channels_per_offset, channels_per_joint, hidden_size, parents,
                               extend_channels([1], channels_per_joint), r_leg_offsets, r_leg_dq_in, device)
        self.larmNet = LimbNet(channels_per_offset, channels_per_joint, hidden_size, parents,
                               extend_channels([3], channels_per_joint), l_arm_offsets, l_arm_dq_in, device)
        self.rarmNet = LimbNet(channels_per_offset, channels_per_joint, hidden_size, parents,
                               extend_channels([4], channels_per_joint), r_arm_offsets, r_arm_dq_in, device)

    def forward(self, decoder_output, sparse_input, offsets):
        # decoder_output shape: (batch_size, n_joints*8, frames)
        # sparse_input shape: (batch_size, n_sparse_joints*8, frames)
        self.mainNet.run(sparse_input, offsets, decoder_output)
        sparse_input = sparse_input[:, 8:, :]  # exclude root

        self.llegNet.run(sparse_input, offsets, decoder_output, self.l_leg_dq_out)
        self.rlegNet.run(sparse_input, offsets, decoder_output, self.r_leg_dq_out)
        self.larmNet.run(sparse_input, offsets, decoder_output, self.l_arm_dq_out)
        self.rarmNet.run(sparse_input, offsets, decoder_output, self.r_arm_dq_out)

        return decoder_output


class TrunkNet(nn.Module):
    def __init__(self, channels_per_offset, channels_per_joint, hidden_size, parents, device):
        super().__init__()

        offsets_index = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
        dq_in_index = [0, 9, 10, 11, 12, 14, 15, 16, 18, 19, 20]
        sparse_index = [0, 3, 4, 5]  # 6个稀疏节点中的序号

        self.dq_in_extended = None
        self.channels_per_offset = channels_per_offset
        self.channels_per_joint = channels_per_joint
        self.hidden_size = hidden_size
        self.parents = parents
        self.device = device
        self.sparse_index = sparse_index
        self.offsets_index = offsets_index
        self.offsets_in_extended = extend_channels(offsets_index, channels_per_offset)
        self.dq_in_index = dq_in_index

        self.sequential1 = self.create_sequential(offsets_index, dq_in_index, 1)
        self.sequential2 = self.create_sequential(offsets_index, dq_in_index, 3)
        self.sequential3 = self.create_sequential(offsets_index, dq_in_index, 3)

        dq_out_index1 = [11]
        dq_out_index2 = [10, 11, 12]
        dq_out_index3 = [0, 9, 10]
        self.dq_out_extended1 = extend_channels(dq_out_index1, self.channels_per_joint)
        self.dq_out_extended2 = extend_channels(dq_out_index2, self.channels_per_joint)
        self.dq_out_extended3 = extend_channels(dq_out_index3, self.channels_per_joint)

        self.sparse_indexes = extend_channels(self.sparse_index, self.channels_per_joint)

    def create_sequential(self, offsets_index, dq_in_index, modify_node_num):
        self.dq_in_extended = extend_channels(dq_in_index, self.channels_per_joint, self.parents)

        input_size = (
                len(offsets_index) * self.channels_per_offset  # offsets
                + len(self.dq_in_extended)  # input pose
                + len(self.sparse_index) * self.channels_per_joint  # sparse input (end effector)
        )
        output_size = modify_node_num * self.channels_per_joint  # output modified joints

        model = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, output_size),
        ).to(self.device)

        return model

    def run(self, sparse_input, offsets, decoder_output):
        """
        :param sparse_input: 稀疏输入应包含根节点
        :param offsets:
        :param decoder_output:
        :return:
        """

        res1 = self.sequential1(
            torch.cat(
                (offsets[:, self.offsets_in_extended, :],
                 decoder_output[:, self.dq_in_extended, :],
                 sparse_input[:, self.sparse_indexes]), dim=-2).permute(0, 2, 1)
        ).permute(0, 2, 1)

        # change res to shape (batch_size, frames, num_joints, 8)
        res1 = res1.reshape(res1.shape[0], -1, 8, res1.shape[-1]).permute(0, 3, 1, 2)
        res1 = dquat.normalize(res1)  # convert to unit dual quaternions
        res1 = res1.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)

        decoder_output[:, self.dq_out_extended1, :] = res1

        res2 = self.sequential2(
            torch.cat(
                (offsets[:, self.offsets_in_extended, :],
                 decoder_output[:, self.dq_in_extended, :],
                 sparse_input[:, self.sparse_indexes]), dim=-2).permute(0, 2, 1)
        ).permute(0, 2, 1)

        # change res to shape (batch_size, frames, num_joints, 8)
        res2 = res2.reshape(res2.shape[0], -1, 8, res2.shape[-1]).permute(0, 3, 1, 2)
        res2 = dquat.normalize(res2)  # convert to unit dual quaternions
        res2 = res2.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)

        decoder_output[:, self.dq_out_extended2, :] = res2

        res3 = self.sequential3(
            torch.cat(
                (offsets[:, self.offsets_in_extended, :],
                 decoder_output[:, self.dq_in_extended, :],
                 sparse_input[:, self.sparse_indexes]), dim=-2).permute(0, 2, 1)
        ).permute(0, 2, 1)

        # change res to shape (batch_size, frames, num_joints, 8)
        res3 = res3.reshape(res3.shape[0], -1, 8, res3.shape[-1]).permute(0, 3, 1, 2)
        res3 = dquat.normalize(res3)  # convert to unit dual quaternions
        res3 = res3.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        decoder_output[:, self.dq_out_extended3, :] = res3

        return res3


class LimbNet(nn.Module):
    def __init__(self, channels_per_offset, channels_per_joint, hidden_size,
                 parents, sparse_index, offsets_index, dq_in_index, device):
        super().__init__()

        self.dq_in_extended = None
        self.channels_per_offset = channels_per_offset
        self.channels_per_joint = channels_per_joint
        self.hidden_size = hidden_size
        self.parents = parents
        self.device = device
        self.sparse_indexes = sparse_index
        self.offsets_index = offsets_index
        self.offsets_in_extended = extend_channels(offsets_index, channels_per_offset)
        self.dq_in_index = dq_in_index

        self.sequential1 = self.create_sequential(offsets_index, dq_in_index, 1)
        self.sequential2 = self.create_sequential(offsets_index, dq_in_index, 2)
        self.sequential3 = self.create_sequential(offsets_index, dq_in_index, 3)
        # self.sequential4 = self.create_sequential(offsets_index, dq_in_index, 1)

    def create_sequential(self, offsets_index, dq_in_index, modify_node_num):
        self.dq_in_extended = extend_channels(dq_in_index, self.channels_per_joint, self.parents)

        input_size = (
                len(offsets_index) * self.channels_per_offset  # offsets
                + len(self.dq_in_extended)  # input pose
                + self.channels_per_joint  # sparse input (end effector)
        )
        output_size = modify_node_num * self.channels_per_joint  # output modified joints

        model = nn.Sequential(
            nn.Linear(input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, output_size),
        ).to(self.device)

        return model

    def run(self, sparse_input, offsets, decoder_output, dq_out_index):
        dq_out_extended = extend_channels(dq_out_index, self.channels_per_joint)

        res1 = self.sequential1(
            torch.cat(
                (offsets[:, self.offsets_in_extended, :],
                 decoder_output[:, self.dq_in_extended, :],
                 sparse_input[:, self.sparse_indexes]), dim=-2).permute(0, 2, 1)
        ).permute(0, 2, 1)

        # change res to shape (batch_size, frames, num_joints, 8)
        res1 = res1.reshape(res1.shape[0], -1, 8, res1.shape[-1]).permute(0, 3, 1, 2)
        res1 = dquat.normalize(res1)  # convert to unit dual quaternions
        res1 = res1.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)

        decoder_output[:, dq_out_extended[0:8], :] = res1

        res2 = self.sequential2(
            torch.cat(
                (offsets[:, self.offsets_in_extended, :],
                 decoder_output[:, self.dq_in_extended, :],
                 sparse_input[:, self.sparse_indexes]), dim=-2).permute(0, 2, 1)
        ).permute(0, 2, 1)

        # change res to shape (batch_size, frames, num_joints, 8)
        res2 = res2.reshape(res2.shape[0], -1, 8, res2.shape[-1]).permute(0, 3, 1, 2)
        res2 = dquat.normalize(res2)  # convert to unit dual quaternions
        res2 = res2.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)

        decoder_output[:, dq_out_extended[0:16], :] = res2

        res3 = self.sequential3(
            torch.cat(
                (offsets[:, self.offsets_in_extended, :],
                 decoder_output[:, self.dq_in_extended, :],
                 sparse_input[:, self.sparse_indexes]), dim=-2).permute(0, 2, 1)
        ).permute(0, 2, 1)

        # change res to shape (batch_size, frames, num_joints, 8)
        res3 = res3.reshape(res3.shape[0], -1, 8, res3.shape[-1]).permute(0, 3, 1, 2)
        res3 = dquat.normalize(res3)  # convert to unit dual quaternions
        res3 = res3.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        decoder_output[:, dq_out_extended, :] = res3

        return res3


def extend_channels(node_list, channels_per_joint, parents=None):
    # 将输入从 指定节点 拓展到 它到根节点的路径
    if parents is not None:
        while node_list[-1] != 0:
            node_list.append(parents[node_list[-1]])
    # 将原始列表拓展为通道列表
    channel_list = [
        i
        for j in node_list
        for i in range(j * channels_per_joint, j * channels_per_joint + channels_per_joint)
    ]
    return channel_list
