import torch
import torch.nn as nn

from net.ik_net import IK_NET
from net.loss import MSE_DQ_FK


class IK_Model(nn.Module):
    def __init__(self, device, param, parents, train_data) -> None:
        super().__init__()

        self.ik_res = None
        self.device = device
        self.param = param
        self.parents = parents
        self.data = train_data

        self.ik_net = IK_NET(param, parents, device).to(device)

        # Print number parameters
        parameters = list(self.ik_net.parameters())
        dec_params = 0
        for parameter in parameters:
            dec_params += parameter.numel()
        print("# parameters ik:", dec_params)

        self.optimizer = torch.optim.AdamW(parameters, param["learning_rate"])

        self.loss = MSE_DQ_FK(param, parents, device).to(device)
        train_data.loss = self.loss

    def forward(self, res_decoder):
        self.ik_res = self.ik_net(
            res_decoder.clone().detach(),   # 不可以取消clone否则两个阶段无法独立
            self.data.sparse_dqs[:, :, 42:],
            self.data.offsets
        )

        return self.ik_res

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        loss_ik = self.loss.forward_ik(
            self.ik_res,
            self.data.gt_result
        )
        loss_ik.backward()
        self.optimizer.step()
        return loss_ik
