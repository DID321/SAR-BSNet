"""
@time: 2025/01/08
@file: utils.py
@author: WD                     ___       __   ________            
@contact: wdnudt@163.com        __ |     / /   ___  __ \
                                __ | /| / /    __  / / /
                                __ |/ |/ /     _  /_/ / 
                                ____/|__/      /_____/  


"""

import os
import sys
import json
import pickle
import random
import math
from functools import partial
import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from torch.nn import functional as F
from utils.evaluate import MeanAbsoluteError, F1Score
def criterion(inputs, target):
    return F.binary_cross_entropy_with_logits(inputs, target)

def show_config(params):
    print('Configurations:')
    print('-' * 70)
    print('|%25s | %40s|' % ('keys', 'values'))
    print('-' * 70)
    for key, value in params.items():
        print('|%25s | %40s|' % (str(key), str(value)))
    print('-' * 70)

def seed_everything(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#---------------------------------------------------#
#   设置Dataloader的种子
#---------------------------------------------------#
def worker_init_fn(worker_id, rank, seed):
    worker_seed = rank + seed
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    train_loss = 0.0  # 累计损失
    train_mae = 0.0   # 累计预测正确的样本数
    train_f1 = 0.0
    lr = 0.0
    # 评价指标评估器
    mae_metric = MeanAbsoluteError()
    f1_metric = F1Score()
    # sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        # sample_num += images.shape[0]
        with torch.no_grad():
            if device is not None:
                images = images.to(device)
                labels = labels.to(device)
        optimizer.zero_grad()
        # 正向传播
        outputs = model(images)
        # 计算损失
        # loss = criterion(outputs, labels)
        loss = criterion(outputs, labels)

        # 反向传播
        loss.backward()
        optimizer.step()

        outputs_sigmoid = torch.sigmoid(outputs)
        # -------------------------------#
        #   计算MAE, f_score
        # -------------------------------#
        for i in range(outputs_sigmoid.shape[0]):
            mae_metric.update(outputs_sigmoid[i].unsqueeze(0), labels[i].unsqueeze(0))
            f1_metric.update(outputs_sigmoid[i].unsqueeze(0), labels[i].unsqueeze(0))

        train_loss += loss.item()
        train_mae  += mae_metric.compute()
        train_f1   += f1_metric.compute()

        lr = optimizer.param_groups[0]["lr"]
        data_loader.desc = "[train epoch {}] loss: {:.4f}, MAE: {:.4f}, F1: {:.4f}, lr: {:.5f}".format(
            epoch,
            train_loss / (step + 1),
            train_mae / (step + 1),
            train_f1 / (step + 1),
            lr
        )

        # 如果 loss 包含无穷大（inf）或非数字（NaN）值，torch.isfinite(loss) 将返回 False
        # if not torch.isfinite(loss):
        #     print('⚠️ WARNING: non-finite loss, ending training ', loss)
        #     sys.exit(1)

    mae_metric.gather_from_all_processes()
    f1_metric.reduce_from_all_processes()
    mae_info, f1_info = mae_metric.compute(), f1_metric.compute()

    # return train_loss, train_mae, train_f1
    return train_loss / (step + 1), mae_info, f1_info, lr


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    # 切换到评估模式
    model.eval()
    val_loss = 0.0
    val_mae = 0.0
    val_f1 = 0.0
    mae_metric = MeanAbsoluteError()
    f1_metric = F1Score()
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        # sample_num += images.shape[0]
        with torch.no_grad():
            if device is not None:
                images = images.to(device)
                labels = labels.to(device)
        # 正向传播
        outputs = model(images)
        # 计算损失
        loss = criterion(outputs, labels)
        outputs_sigmoid = torch.sigmoid(outputs)
        # -------------------------------#
        #   计算MAE, f_score
        # -------------------------------#
        for i in range(outputs_sigmoid.shape[0]):
            mae_metric.update(outputs_sigmoid[i].unsqueeze(0), labels[i].unsqueeze(0))
            f1_metric.update(outputs_sigmoid[i].unsqueeze(0), labels[i].unsqueeze(0))

        val_loss += loss.item()
        val_mae += mae_metric.compute()
        val_f1 += f1_metric.compute()

        data_loader.desc = "[valid epoch {}] loss: {:.4f}, MAE: {:.4f}, F1: {:.4f}".format(
            epoch,
            val_loss / (step + 1),
            val_mae / (step + 1),
            val_f1 / (step + 1),
        )

    mae_metric.gather_from_all_processes()
    f1_metric.reduce_from_all_processes()
    mae_info, f1_info = mae_metric.compute(), f1_metric.compute()

    # return train_loss, train_mae, train_f1
    return val_loss / (step + 1), mae_info, f1_info

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0 + math.cos(math.pi* (iters - warmup_total_iters) / (total_iters - warmup_total_iters - no_aug_iter))
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "Cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func

def set_optimizer_lr(optimizer, lr_scheduler_func, epoch):
    lr = lr_scheduler_func(epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_params_groups(model: torch.nn.Module, weight_decay: float = 1e-5):
    # 记录optimize要训练的权重参数
    parameter_group_vars = {"decay": {"params": [], "weight_decay": weight_decay},
                            "no_decay": {"params": [], "weight_decay": 0.}}

    # 记录对应的权重名称
    parameter_group_names = {"decay": {"params": [], "weight_decay": weight_decay},
                             "no_decay": {"params": [], "weight_decay": 0.}}

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights

        if len(param.shape) == 1 or name.endswith(".bias"):
            group_name = "no_decay"
        else:
            group_name = "decay"

        parameter_group_vars[group_name]["params"].append(param)
        parameter_group_names[group_name]["params"].append(name)

    print("Param groups = %s" % json.dumps(parameter_group_names, indent=2))
    return list(parameter_group_vars.values())
