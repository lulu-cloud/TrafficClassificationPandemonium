# -*- coding: utf-8 -*-
# @Time : 2024/3/4 15:42
# @Author : XULu
# @Email : xulu_lili@163.com
# @File : 2_train_test_model.py
"""
@Description: 训练与测试模型
"""

import sys

import os

# 把当前文件所在文件夹的父文件夹路径加入到PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from torch import nn, optim
from log.set_log import logger
from utils.set_config import setup_config
# from models.cnn1d import cnn1d as train_model
from models.app_net import app_net as train_model
from train_valid.train import train_process
from train_valid.valid import valid_process
from dataloader.data_loader import data_loader
from dataloader.get_tensor import get_tensor_data

from utils.helper import adjust_learning_rate, save_checkpoint

from utils.evaluate_tools import display_model_performance_metrics

from torch.utils.tensorboard import SummaryWriter

# logger = init_logger(log_path='/home/xl/TrafficClassificationPandemonium/log/log_file/train.log')


def train_pipeline():
    yaml_path = r"/home/xl/TrafficClassificationPandemonium/configuration/traffic_classification_configuration.yaml"
    cfg = setup_config(yaml_path) # 获取 config 文件
    logger.info(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info('是否使用 GPU 进行训练, {}'.format(device))

    os.makedirs(cfg.train.model_dir, exist_ok=True)
    model_path = os.path.join(cfg.train.model_dir, cfg.train.model_name)  # 模型的路径
    num_classes = len(cfg.test.label2index)

    model = train_model(model_path, pretrained=cfg.test.pretrained,
                  num_classes=num_classes).to(device)  # 定义模型)

    criterion_c = nn.CrossEntropyLoss()  # 分类用的损失函数
    criterion_r = nn.L1Loss()  # 重构误差的损失函数
    optimizer = optim.Adam(model.parameters(), lr=cfg.train.lr)  # 定义优化器
    logger.info('成功初始化模型.')

    train_loader = data_loader(pcap_file=cfg.train.train_pay, seq_file=cfg.train.train_seq,
                               statistic_file=cfg.train.train_sta,
                               label_file=cfg.train.train_label,
                               batch_size=cfg.train.BATCH_SIZE)  # 获得 train dataloader
    test_loader = data_loader(pcap_file=cfg.train.test_pay, seq_file=cfg.train.test_seq,
                              statistic_file=cfg.train.test_sta,
                              label_file=cfg.train.test_label,
                              batch_size=cfg.train.BATCH_SIZE)  # 获得 train dataloader
    logger.info('成功加载数据集.')

    if cfg.test.evaluate:  # 是否只进行测试
        logger.info('进入测试模式.')
        prec1, val_loss, val_acc = valid_process(test_loader, model, 1, criterion_c, criterion_r,0, device,
                                                 1)  # evaluate on validation set
        torch.cuda.empty_cache()  # 清除显存
        # 计算每个类别详细的准确率
        index2label = {j: i for i, j in cfg.test.label2index.items()}  # index->label 对应关系
        print(index2label)
        label_list = [index2label.get(i) for i in range(len(index2label))]  # 17 个 label 的标签
        pcap_data, seq_data, statistic_data, label_data = get_tensor_data(pcap_file=cfg.train.test_pay,
                                                                          seq_file=cfg.train.test_seq,
                                                                          statistic_file=cfg.train.test_sta,
                                                                          label_file=cfg.train.test_label)
        start_index = 0
        y_pred = None
        int_test_nums = len(test_loader) * (cfg.train.BATCH_SIZE - 1)
        int_test_nums = (int)(int_test_nums / 100) * 100

        for i in list(range(100, int_test_nums + 100, 100)):
            pay = pcap_data[start_index:i]
            seq = seq_data[start_index:i]
            sta = statistic_data[start_index:i]

            y_pred_batch, _ = model(pay.to(device), seq.to(device), sta.to(device))

            start_index = i
            if y_pred == None:
                y_pred = y_pred_batch.cpu().detach()
            else:
                y_pred = torch.cat((y_pred, y_pred_batch.cpu().detach()), dim=0)
                print(y_pred.shape)

        _, pred = y_pred.topk(1, 1, largest=True, sorted=True)

        Y_data_label = [index2label.get(i.tolist()) for i in label_data]  # 转换为具体名称
        pred_label = [index2label.get(i.tolist()) for i in pred.view(-1).cpu().detach()]

        Y_data_label = Y_data_label[:int_test_nums]
        display_model_performance_metrics(true_labels=Y_data_label, predicted_labels=pred_label,confusion_path = cfg.test.confusion_path, classes=label_list)
        return

    best_prec1 = 0
    loss_writer = SummaryWriter(log_dir=os.path.join("/home/xl/TrafficClassificationPandemonium/result/tensorboard", "loss"))
    acc_writer = SummaryWriter(log_dir=os.path.join("/home/xl/TrafficClassificationPandemonium/result/tensorboard", "acc"))

    for epoch in range(cfg.train.epochs):
        adjust_learning_rate(optimizer, epoch, cfg.train.lr)  # 动态调整学习率

        train_loss, train_acc = train_process(train_loader, model, 1, criterion_c, criterion_r, optimizer, epoch,
                                              device,
                                              2)  # train for one epoch
        prec1, val_loss, val_acc = valid_process(test_loader, model, 1, criterion_c, criterion_r,epoch, device,
                                                 1)  # evaluate on validation set

        loss_writer.add_scalars("loss", {'train': train_loss, 'val': val_loss}, epoch)
        acc_writer.add_scalars("train_acc", {'train': train_acc, 'val': val_acc}, epoch)

        # remember the best prec@1 and save checkpoint
        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        # 保存最优的模型
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer': optimizer.state_dict()
        }, is_best, model_path)

    loss_writer.close()
    acc_writer.close()
    logger.info('Finished! (*￣︶￣)')


if __name__ == "__main__":
    train_pipeline()  # 用于测试
