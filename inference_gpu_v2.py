import os
import time

import torch
import random
import argparse
import numpy as np

from net.loss import get_info_from_npz
from net.skeleton_generator_architecture import Generator_Model
from net.motion_data import Run_Data
from net.ik_architecture import IK_Model
from net.config import param, xsens_parents

import motion.rotations.quat_torch as quat
import motion.rotations.dual_quat_torch as dquat
from motion.ops.forward_kinematics_torch import fk
from motion.rotations.dual_quat_torch import from_rotation_translation


def main(args):
    # Set seed
    torch.manual_seed(param["seed"])
    random.seed(param["seed"])
    np.random.seed(param["seed"])

    device = torch.device("cuda")
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

    filename = eval_files[0]
    rotations, global_pos, parents, offsets, _ = get_info_from_npz(filename, device)
    rotations = rotations[:43]
    global_pos = global_pos[:43]
    # caculate joint position
    translations, _ = fk(rotations, global_pos, offsets, parents)
    global_pos = global_pos[[-1]]

    # Create Models
    run_data = Run_Data(device, param)
    generator_model = Generator_Model(device, param, xsens_parents, run_data).to(device)
    ik_model = IK_Model(device, param, xsens_parents, run_data).to(device)
    # Load  Model
    load_model(generator_model, model_epoch, "generator", device)
    load_model(ik_model, model_epoch, "ik", device)

    generator_model.eval()
    ik_model.eval()

    import onnxruntime as ort

    # 加载 ONNX 模型
    model_path = "generator_model_v2.onnx"
    ort_session = ort.InferenceSession(model_path,providers='CUDAExecutionProvider')
    ik_model_path = "ik_model_v2.onnx"
    ik_ort_session = ort.InferenceSession(ik_model_path,providers='CUDAExecutionProvider')

    import tensorrt as trt

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    # 读取 ONNX 文件
    with open("model.onnx", "rb") as model:
        parser.parse(model.read())

    # 构建 TensorRT 引擎
    engine = builder.build_cuda_engine(network)
    # 进行推理
    context = engine.create_execution_context()

    # netron.start("ik_model.onnx")

    # 获取输入和输出的名称
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    # 使用 torch.cuda.Event 测量耗时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    num_repeat = 10000
    frames = 0

    offsets = offsets.flatten(start_dim=0, end_dim=1)

    start_time = time.time()  # 获取开始时间

    start_event.record()
    for i in range(num_repeat):
        frames += 1
        with torch.no_grad():  # 在其内部的所有操作中禁用梯度计算，禁止计算图的构建，从而节省内存和加速运算
            # create dual quaternions
            sparse_dqs = from_rotation_translation(rotations[:, param["sparse_joints"]],
                                                   translations[:, param["sparse_joints"]])
            sparse_dqs = dquat.unroll(sparse_dqs, dim=0).to(device)  # ensure continuity (T,6,8)
            sparse_dqs = torch.flatten(sparse_dqs, 1, 2).permute(1, 0)  # (6*8,T)

            # run_data.set_motions(
            #     sparse_dqs.unsqueeze(0),
            #     offsets.unsqueeze(0).unsqueeze(-1),
            # )
            # res_generator = generator_model.forward()   # take 7.6ms
            # result = ik_model.forward(res_generator)    # take 10+ms

            # 准备输入数据（示例输入）
            sparse_dqs = sparse_dqs.cpu().numpy()
            # 运行推理
            outputs_generator = ort_session.run(None, {input_name: sparse_dqs})

            # 运行推理
            outputs_ik = ik_ort_session.run(None, {"input1": outputs_generator[0],
                                                   "input2": sparse_dqs[:, 42],
                                                   "input3": offsets.cpu().numpy()})

            result = torch.tensor(outputs_ik[0]).to(device)
            # result = torch.tensor(result[0]).to(device)
            # Evaluate Positional Error
            dqs = result.reshape(22, 8)

            # get rotations and translations from dual quatenions
            rots = from_root_dual_quat(dqs, parents)  # take 9ms
            rots = rots.to(device)
            joint_poses, _ = fk(rots, global_pos,offsets.reshape(22, 3), parents)

    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!

    end_time = time.time()  # 获取结束时间
    cpu_time = (end_time - start_time) * 1000
    print(f"CPU time: {cpu_time} ms")
    print("CPU time delay = ", cpu_time / frames, " ms, fps=",frames / cpu_time * 1000)

    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f'Elapsed: {elapsed_time_ms:.5f}ms')  # Elapsed: 109.9ms

    print("total_frames=", frames, "fps=", frames / elapsed_time_ms * 1000)


def from_root_dual_quat(dq: torch.Tensor, parents: torch.Tensor):
    """
    Convert root-centered dual quaternion to the skeleton information.

    Parameters
    ----------
    dq : torch.Tensor[..., n_joints, 8]
        Includes as first element the global position of the root joint
    parents : torch.Tensor[n_joints]

    Returns
    -------
    rotations : torch.Tensor[..., n_joints, 4]
    """
    rotations = dq[..., :4]  # Extract the rotation part

    # Create a mask to identify non-root joints
    is_non_root = parents != 0

    # Get the parent indices for non-root joints
    parent_indices = parents[is_non_root]

    # Extract the rotations of the parent joints
    parent_rotations = rotations[..., parent_indices, :]

    # Compute the inverse of parent rotations
    inv_parent_rotations = quat.inverse(parent_rotations)

    # Compute the local rotations for non-root joints
    local_rotations = quat.mul(inv_parent_rotations, rotations[..., is_non_root, :])

    # Update the rotations tensor with the local rotations
    rotations[..., is_non_root, :] = local_rotations

    return rotations


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
