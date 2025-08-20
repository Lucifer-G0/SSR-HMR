import os
import time
import torch
import random
import argparse
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.dataloader import DataLoader

from net.skeleton_generator_architecture import Generator_Model
from net.motion_data import Train_Data, TrainMotionData, EvalMotionData
from net.loss import run_generator, get_info_from_npz, eval_result_, run_ik, eval_result
from net.config import param, xsens_parents
from net.ik_architecture import IK_Model


def main(args):
    # Set seed
    torch.manual_seed(param["seed"])
    random.seed(param["seed"])
    np.random.seed(param["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))

    # Prepare Data
    train_eval_dir = args.data_path
    train_dir = os.path.join(train_eval_dir, "train")
    if not os.path.exists(train_dir):
        raise ValueError("train directory does not exist")
    eval_dir = os.path.join(train_eval_dir, "eval")
    if not os.path.exists(eval_dir):
        raise ValueError("eval directory does not exist")

    train_files = []
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            # 检查文件是否以.npz结尾，且文件名不是shape.npz
            if file.endswith('.npz') and file != 'shape.npz':
                full_path = os.path.join(root, file)
                train_files.append(full_path)

    eval_files = []
    for root, dirs, files in os.walk(eval_dir):
        for file in files:
            # 检查文件是否以.npz结尾，且文件名不是shape.npz
            if file.endswith('.npz') and file != 'shape.npz':
                full_path = os.path.join(root, file)
                eval_files.append(full_path)

    train_dataset = TrainMotionData(param, device)
    eval_dataset = EvalMotionData(param, device)

    # Train Files
    for filename in train_files:
        if filename[-4:] == ".npz":
            rots, pos, parents, offsets,_ = get_info_from_npz(filename, device)
            # Train Dataset
            train_dataset.add_motion(
                offsets,
                pos,  # only global position
                rots,
                parents,
            )
    print(len(train_dataset), " added to train_dataset")

    # Eval Files
    for filename in eval_files:
        if filename[-4:] == ".npz":
            eval_dataset.add_motion(filename)
    print(eval_dataset.get_len(), " added to eval_dataset")

    train_dataloader = DataLoader(train_dataset, param["batch_size"], shuffle=False)

    # Create Models
    train_data = Train_Data(device, param)
    generator_model = Generator_Model(device, param, xsens_parents, train_data).to(device)
    ik_model = IK_Model(device, param, xsens_parents, train_data).to(device)

    scheduler_g = ReduceLROnPlateau(generator_model.optimizer, 'min', factor=0.5, patience=20)
    scheduler_ik = ReduceLROnPlateau(ik_model.optimizer, 'min', factor=0.5, patience=20)

    mylog = open('trunknet-vr-global.log', mode='a', encoding='utf-8')
    # Training Loop
    best_evaluation = float("inf")
    store_evaluation = float("inf")
    start_time = time.time()
    for epoch in range(param["epochs"]):
        for step, motion in enumerate(train_dataloader):
            # Forward
            train_data.set_offsets(motion["offsets"])
            train_data.set_motions(
                motion["dqs"],
                motion["sparse_dqs"],
                motion["gt_result"]
            )
            generator_model.train()
            res_decoder = generator_model.forward()

            ik_model.train()
            ik_model.forward(res_decoder)

            # Loss
            generator_model.optimize_parameters()
            ik_model.optimize_parameters()


        # 当所有训练数据使用完成
        results = run_generator(generator_model, train_data, eval_dataset)
        results_ik = run_ik(ik_model, results, train_data, eval_dataset)

        mpjpe, mpeepe, vel, jitter = eval_result_(results_ik,eval_dataset, device)
        evaluation_loss = mpjpe + mpeepe
        evaluation_loss = torch.round(evaluation_loss * 1000)/1000

        # 学习率更新只看小数点后三位
        scheduler_g.step(evaluation_loss)
        scheduler_ik.step(evaluation_loss)

        # If best, save model
        was_best = False
        if best_evaluation > evaluation_loss:
            best_evaluation = evaluation_loss
            was_best = True
        if epoch > 400 and (store_evaluation - evaluation_loss >= 0.01):
            save_model(generator_model, ik_model, epoch)
            store_evaluation = evaluation_loss
        elif epoch > 200 and (store_evaluation - evaluation_loss >= 0.1):
            save_model(generator_model, ik_model, epoch)
            store_evaluation = evaluation_loss

        # Print
        current_lr = generator_model.optimizer.param_groups[0]['lr']
        print(
            "Epoch: {} - MPJPE: {:.8f} - MPEEPE: {:.8f} - MPJVE: {:.8f} - Jitter: {:.8f} - lr:{}".format(epoch,
                                                                                                         mpjpe,
                                                                                                         mpeepe,
                                                                                                         vel,
                                                                                                         jitter,
                                                                                                         current_lr)
            + ("*" if was_best else "")
        )
        print(
            "Epoch: {} - MPJPE: {:.8f} - MPEEPE: {:.8f} - MPJVE: {:.8f} - Jitter: {:.8f} - lr:{}".format(epoch,
                                                                                                         mpjpe,
                                                                                                         mpeepe,
                                                                                                         vel,
                                                                                                         jitter,
                                                                                                         current_lr)
            + ("*" if was_best else ""), file=mylog)
    end_time = time.time()
    print("Training Time:", end_time - start_time)
    print("Training Time:", end_time - start_time, file=mylog)

    mylog.close()


def load_model(model, model_name, device):
    model_path = os.path.join("./model", model_name + ".pt")
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def save_model(generator_model, ik_model, epoch):
    model_dir = f'./model/{epoch}'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    generator_path = f"./model/{epoch}/generator.pt"
    ik_path = f"./model/{epoch}/ik.pt"
    if generator_model is not None:
        torch.save(
            {
                "model_state_dict": generator_model.state_dict(),
            },
            generator_path,
        )
    if ik_model is not None:
        torch.save(
            {
                "model_state_dict": ik_model.state_dict(),
            },
            ik_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Motion Upsampling Network")
    parser.add_argument(
        "data_path",
        type=str,
        help="path to data directory containing one or multiple .npz for training, last .bvh is used as test data",
    )

    args = parser.parse_args()

    main(args)
