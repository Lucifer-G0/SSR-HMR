import os

import netron
import torch
import random
import argparse
import numpy as np

from net.run_skeleton_generator_architecture import Generator_Model,Run_Generator_Model
from net.motion_data import Run_Data
from net.ik_architecture import IK_Model, Run_IK_Model_V2
from net.config import param, xsens_parents


def main(args):
    print(torch.__version__)
    # Set seed
    torch.manual_seed(param["seed"])
    random.seed(param["seed"])
    np.random.seed(param["seed"])

    device = torch.device("cpu")
    print("Using device:", device)
    # Additional Info when using cuda
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    model_epoch = args.model_epoch

    # Create Models
    run_data = Run_Data(device, param)
    generator_model = Generator_Model(device, param, xsens_parents, run_data).to(device)
    ik_model = IK_Model(device, param, xsens_parents, run_data).to(device)

    run_generator_model = Run_Generator_Model(device, param, xsens_parents).to(device)
    run_ik_model = Run_IK_Model_V2(device, param, xsens_parents).to(device)
    # Load  Model
    load_model(generator_model, model_epoch, "generator", device)
    load_model(ik_model, model_epoch, "ik", device)

    load_model(run_generator_model, model_epoch, "generator", device)
    load_model(run_ik_model, model_epoch, "ik", device)

    generator_model.eval()
    ik_model.eval()
    run_generator_model.eval()
    run_ik_model.eval()

    # 准备输入张量
    sparse_input_ = torch.randn(48, 43)  # 创建随机输入张量
    res_decoder_ = torch.randn(176)  # 创建随机输入张量
    sparse_dqs_ = torch.randn(48)  # 创建随机输入张量
    offsets_ = torch.randn(66)  # 创建随机输入张量

    # 导出为 ONNX 格式
    output_onnx = "generator_model_v2.onnx"  # 输出文件名
    torch.onnx.export(
        run_generator_model,  # 要导出的模型
        sparse_input_,  # 输入张量
        output_onnx,  # 输出文件路径
        export_params=True,  # 是否导出模型参数
        opset_version=16,  # ONNX 操作集版本
        do_constant_folding=True,  # 是否执行常量折叠优化
        input_names=["input"],  # 输入节点名称
        output_names=["output"],  # 输出节点名称
        dynamic_axes={}
    )
    print(f"模型已成功导出为 ONNX 格式，并保存到 {output_onnx}")

    # 导出为 ONNX 格式
    output_onnx = "ik_model_v2.onnx"  # 输出文件名
    torch.onnx.export(
        run_ik_model,  # 要导出的模型
        (res_decoder_, sparse_dqs_, offsets_),  # 输入张量
        output_onnx,  # 输出文件路径
        export_params=True,  # 是否导出模型参数
        opset_version=16,  # ONNX 操作集版本
        do_constant_folding=False,  # 是否执行常量折叠优化
        input_names=["input1", "input2", "input3"],  # 输入节点名称
        output_names=["output"],  # 输出节点名称
        dynamic_axes={}
    )

    print(f"模型已成功导出为 ONNX 格式，并保存到 {output_onnx}")
    # netron.start("ik_model_v2.onnx")

def load_model(model, model_epoch, model_name, device):
    model_path = os.path.join("./model", model_epoch, model_name + ".pt")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Motion Upsampling Network")

    parser.add_argument(
        "model_epoch",
        type=str,
        help="the epoch of the model you want to use, store at model/model_epoch/generator.pt&ik.pt",
    )

    args = parser.parse_args()

    main(args)
