import os
import torch
import random
import argparse
import numpy as np

from motion.io.bvh import BVH
from net.loss import run_generator, eval_result, run_ik
from net.skeleton_generator_architecture import Generator_Model
from net.motion_data import Train_Data, EvalMotionData
from net.ik_architecture import IK_Model
from net.config import param, xsens_parents
from motion.ops.skeleton import from_root_dual_quat
import motion.rotations.quat as quat


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
            if file.endswith('.bvh'):
                full_path = os.path.join(root, file)
                eval_files.append(full_path)

    eval_dataset = EvalMotionData(param, device)

    # 使用 torch.cuda.Event 测量耗时
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()

    total_frame_num = 0
    # Eval Files
    for filename in eval_files:
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

    for step, filename in enumerate(eval_files):
        print(step, filename)

        bvh = BVH()
        bvh.load(filename)
        bvh.data["positions"] = bvh.data["positions"][42:42+1000]
        bvh.data["positions"][:,0,:]=np.zeros(3)
        bvh.data["rotations"] = bvh.data["rotations"][42:42+1000]

        fileno = str(step)
        # 保留去掉首部的原始文件、与结果开头对齐
        path = "down"
        filename = "down_" + fileno+".bvh"
        bvh.save(os.path.join(path, filename))

        # 保留结果bvh文件，用于直观用户尺寸评估
        dqs = results[step].permute(0, 2, 1).flatten(0, 1).cpu().detach().numpy()
        # get rotations and translations from dual quatenions
        dqs = dqs.reshape(dqs.shape[0], -1, 8)
        _, rots = from_root_dual_quat(dqs, bvh.data["parents"])
        # quaternions to euler
        rot_roder = np.tile(bvh.data["rot_order"], (rots.shape[0], 1, 1))
        rotations = np.degrees(quat.to_euler(rots, order=rot_roder))
        bvh.data["rotations"] = rotations[:1000]
        # positions
        # positions = bvh.data["positions"][: rotations.shape[0]]
        # bvh.data["positions"] = positions

        path = "predict"
        filename = "predict_" + fileno+".bvh"
        bvh.save(os.path.join(path, filename))


    mpjpe, std_mpjpe, mpeepe, std_mpeepe, vel, std_vel, rjitter, std_rjitter, jitter, std_jitter, mpjre, std_mpjre = eval_result(
        results, eval_dataset, device)
    evaluation_loss = mpjpe + mpeepe

    end_event.record()
    torch.cuda.synchronize()  # Wait for the events to be recorded!
    elapsed_time_ms = start_event.elapsed_time(end_event)
    print(f'Elapsed: {elapsed_time_ms:.1f}ms')  # Elapsed: 109.9ms

    print(eval_dataset.get_len(), " added to eval_dataset, total_frame_num = ", total_frame_num, ", time length = ",
          total_frame_num / 120, "s")
    print("Evaluate Loss: {}".format(evaluation_loss))
    print("Mean Per Joint Position Error: {}({})".format(mpjpe, std_mpjpe))  # 将单位从m转换为cm
    print("Mean Per Joint Rotation Error: {}({})".format(mpjre, std_mpjre))
    print("Mean End Effector Position Error: {}({})".format(mpeepe, std_mpeepe))  # 将单位从m转换为cm
    print("Vel: {}({})".format(vel, std_vel))
    print("RJitter: {}({})".format(rjitter, std_rjitter))
    print("Jitter: {}({})".format(jitter, std_jitter))

    print("MPJPE & MPJRE & MPEEPE & MPJVE & Jitter")
    print("{:.2f}({:.2f}) & {:.2f}({:.2f}) & {:.2f}({:.2f}) && {:.2f}({:.2f}) & {:.2f}({:.2f})".format(mpjpe, std_mpjpe,
                                                                                                       mpjre,
                                                                                                       std_rjitter,
                                                                                                       mpeepe,
                                                                                                       std_mpeepe, vel,
                                                                                                       std_vel, jitter,
                                                                                                       std_jitter))


def result_to_bvh(res, bvh, filename, save=True):
    res = res.permute(0, 2, 1)
    res = res.flatten(0, 1)
    res = res.cpu().detach().numpy()
    # get dqs and displacement
    dqs = res

    # get rotations and translations from dual quatenions
    dqs = dqs.reshape(dqs.shape[0], -1, 8)
    _, rots = from_root_dual_quat(dqs, bvh.data["parents"])
    # quaternions to euler
    rot_roder = np.tile(bvh.data["rot_order"], (rots.shape[0], 1, 1))
    rotations = np.degrees(quat.to_euler(rots, order=rot_roder))
    bvh.data["rotations"] = rotations
    # positions
    positions = bvh.data["positions"][: rotations.shape[0]]
    bvh.data["positions"] = positions
    path = None
    if save:
        path = "predict"
        filename = "eval_" + filename
        bvh.save(os.path.join(path, filename))
    return path, filename


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
