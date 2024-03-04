# -*- coding: utf-8 -*-
# @Time : 2024/3/4 15:11
# @Author : XULu
# @Email : xulu_lili@163.com
# @File : data_loader.py
"""
@Description: 定义数据加载器
"""


import torch
import numpy as np
import torch.utils.data
from log.set_log import logger

def data_loader(pcap_file, seq_file,statistic_file, label_file,  batch_size=256,pin_memory=True):

    # 载入 npy 数据
    pcap_data = np.load(pcap_file)  # 获得 pcap 文件
    seq_data = np.load(seq_file)
    if statistic_file != 'None':
        statistic_data = np.load(statistic_file)
        # statistic_data = torch.from_numpy(statistic_data).float()
    else:
        statistic_data = np.random.rand(pcap_data.shape[0], pcap_data.shape[1])
    label_data = np.load(label_file)  # 获得 label 数据

    # 将 npy 数据转换为 tensor 数据
    pcap_data = torch.from_numpy(pcap_data.reshape(-1,1,pcap_data.shape[1])).float()
    # (batch_size, seq_len, input_size)
    seq_data = torch.from_numpy(seq_data.reshape(-1,seq_data.shape[1],1)).float()
    statistic_data = torch.from_numpy(statistic_data).float()

    label_data = torch.from_numpy(label_data).long()

    logger.info(
            'pcap 文件大小, {}; seq文件大小:{}; sta文件大小: {}; label 文件大小: {}'.format(pcap_data.shape,
                                                                                                 seq_data.shape,
                                                                                                statistic_data.shape,
                                                                                                 label_data.shape))

    dataset = torch.utils.data.TensorDataset(pcap_data, seq_data, statistic_data, label_data)  # 合并数据
    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin_memory,
        num_workers=1  # set multi-work num read data
    )

    return dataloader

