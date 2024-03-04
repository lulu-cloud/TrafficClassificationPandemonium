# -*- coding: utf-8 -*-
# @Time : 2024/3/4 15:15
# @Author : XULu
# @Email : xulu_lili@163.com
# @File : valid.py
"""
@Description: 验证/测试脚本
"""

from utils.helper import AverageMeter, accuracy
import torch

from log.set_log import logger

def valid_process(valid_loader, model, alpha, criterion_c, criterion_r,epoch, device, print_freq):
    """训练一个 epoch 的流程

    Args:
        train_loader (dataloader): [description]
        model ([type]): [description]
        criterion ([type]): [description]
        optimizer ([type]): [description]
        epoch (int): 当前所在的 epoch
        device (torch.device): 是否使用 gpu
        print_freq ([type]): [description]
    """
    c_losses = AverageMeter()  # 在一个 train loader 中的 loss 变化
    r_losses = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()  # 记录在一个 train loader 中的 accuracy 变化

    model.eval()  # 切换为训练模型

    for i, (pay, seq, statistic, target) in enumerate(valid_loader):
        # pay = pay.reshape(-1,256,1)
        pay = pay.to(device)
        seq = seq.to(device)
        statistic = statistic.to(device)
        target = target.to(device)
        with torch.no_grad():
            output = model(pay, seq, statistic)  # 得到模型预测结果
            classify_result, fake_rebuild = output

            loss_c = criterion_c(classify_result, target)  # 计算 分类的 loss
            if fake_rebuild != None:
                loss_r = criterion_r(statistic, fake_rebuild)  # 计算 重构 loss
                r_losses.update(loss_r.item(), pay.size(0))
            else:
                loss_r = 0
                alpha = 1
            loss = alpha * loss_c + loss_r  # 将两个误差组合在一起

            prec1 = accuracy(classify_result.data, target)
            c_losses.update(loss_c.item(), pay.size(0))
            losses.update(loss.item(), pay.size(0))
            top1.update(prec1[0].item(), pay.size(0))

            # # 反向传播
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            if (i + 1) % print_freq == 0:
                logger.info(
                    'Epoch: [{0}][{1}/{2}], Loss {loss.val:.4f} ({loss.avg:.4f}), Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                        epoch, i, len(valid_loader), loss=losses, top1=top1))

    logger.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg,losses.val,top1.val
