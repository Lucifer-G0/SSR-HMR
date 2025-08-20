import torch
import torch.nn as nn
from net.loss import MSE_DQ
import motion.rotations.dual_quat_torch as dquat
from net.skeleton import (
    SkeletonUnpool,
    find_neighbor,
    SkeletonConv,
    create_pooling_list,
)


class Generator_Model(nn.Module):
    def __init__(self, device, param, parents, train_data) -> None:
        super().__init__()

        self.device = device
        self.param = param
        self.parents = parents
        self.data = train_data
        self.generator = Generator(param, parents, device).to(device)

        parameters = list(self.generator.parameters())

        # Print number parameters
        dec_params = 0
        for parameter in parameters:
            dec_params += parameter.numel()
        print("# parameters generator:", dec_params)

        self.optimizer = torch.optim.AdamW(parameters, param["learning_rate"])
        self.loss = MSE_DQ(parents, device).to(device)

    def forward(self):
        self.result = self.generator(self.data.sparse_dqs)
        return self.result

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        loss = self.loss.forward_generator(
            self.result,
            self.data.dqs[:, :, -22:],  # 结果为64帧后22帧
        )
        loss.backward()
        self.optimizer.step()
        return loss


class Generator(nn.Module):
    def __init__(self, param, parents, device):
        super(Generator, self).__init__()
        self.param = param
        self.device = device
        self.layers = nn.ModuleList()
        self.parents = [parents]
        self.pooling_lists = []
        self.channel_list = []

        # Compute pooled skeletons
        number_layers = 3
        layer_parents = parents
        for l in range(number_layers):
            pooling_list, layer_parents = create_pooling_list(
                layer_parents
            )
            self.pooling_lists.append(pooling_list)
            self.parents.append(layer_parents)

        kernel_size = param["kernel_size_temporal_dim"]
        padding = (kernel_size - 1) // 2

        for i in range(number_layers):
            seq = []

            neighbor_list, _ = find_neighbor(
                self.parents[0], param["neighbor_distance"]
            )
            num_joints = len(neighbor_list)

            if i == 0:
                pooling_list = [[0, 1, 2, 5, 6, 9, 10], [2, 3, 4], [6, 7, 8], [11, 12, 13], [14, 15, 16, 17],
                                [18, 19, 20, 21]]
                unpool = SkeletonUnpool(
                    pooling_list=pooling_list,
                    channels_per_edge=8,
                    device=device,
                )
                seq.append(unpool)

            seq.append(
                SkeletonConv(
                    param=param,
                    neighbor_list=neighbor_list,
                    kernel_size=kernel_size,
                    in_channels_per_joint=8,
                    out_channels_per_joint=8,
                    joint_num=num_joints,
                    padding=padding,
                    stride=1,
                    device=device
                )
            )

            if i != number_layers - 1:
                seq.append(nn.LeakyReLU(negative_slope=0.2))
            # Append to the list of layers
            self.layers.append(nn.Sequential(*seq))

    def forward(self, x):
        """
            x: 稀疏数据(batch_size,6*8,frames)
        """

        for i, layer in enumerate(self.layers):
            x = layer(x)
            # 针对某个帧的卷积图像，绘制对应的卷积过程，要取kernel_size_temporal_dim帧画在一张图像上

        # change input to shape (batch_size, frames, num_joints, 8)
        result = x.reshape(x.shape[0], -1, 8, x.shape[-1]).permute(0, 3, 1, 2)
        # convert to unit dual quaternions
        result = dquat.normalize(result)
        # normalize rotations

        # change result to shape (batch_size,channels,frames)
        result = result.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)

        return result
