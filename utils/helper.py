# -*- coding: utf-8 -*-
# @Time : 2024/3/4 15:27
# @Author : XULu
# @Email : xulu_lili@163.com
# @File : helper.py
"""
@Description: 训练的工具类
"""

import torch


class AverageMeter(object):
    """Computes and stores the average and current value
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k
    """
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, largest=True, sorted=True) # 找到前 topk 的 indices
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(state, is_best, filename='alex_checkpoint.pth'):
    """保存中间模型, 暂时只保存最优的一个结果

    Args:
        state ([type]): [description]
        is_best (bool): [description]
        filename (str, optional): [description]. Defaults to 'alex_checkpoint.pth'.
    """
    # torch.save(state, filename)
    if is_best:
        torch.save(state, filename)
        # shutil.copyfile(filename, './checkpoint/alex_model_best.pth')


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs
    """
    lr = init_lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

