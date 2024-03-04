# -*- coding: utf-8 -*-
# @Time : 2024/3/4 15:14
# @Author : XULu
# @Email : xulu_lili@163.com
# @File : get_tensor.py
"""
@Description: numpy转tensor
"""

import torch
import numpy as np

from log.set_log import logger


def get_tensor_data(pcap_file, seq_file, statistic_file, label_file):
    # 载入 npy 数据
    pcap_data = np.load(pcap_file)  # 获得 pcap 文件
    seq_data = np.load(seq_file)
    if statistic_file != 'None':
        statistic_data = np.load(statistic_file)
        
    else:
        statistic_data = np.random.rand(pcap_data.shape[0], pcap_data.shape[1])
    # statistic_data = torch.from_numpy(statistic_data).float()
    label_data = np.load(label_file)  # 获得 label 数据

    # 将 npy 数据转换为 tensor 数据
    pcap_data = torch.from_numpy(pcap_data.reshape(-1, 1, pcap_data.shape[1])).float()
    # (batch_size, seq_len, input_size)
    seq_data = torch.from_numpy(seq_data.reshape(-1, seq_data.shape[1], 1)).float()
    statistic_data = torch.from_numpy(statistic_data).float()


    label_data = torch.from_numpy(label_data).long()

    logger.info(
            'pcap 文件大小, {}; seq文件大小:{}; sta文件大小: {}; label 文件大小: {}'.format(pcap_data.shape,
                                                                                                 seq_data.shape,
                                                                                                statistic_data.shape,
                                                                                                 label_data.shape))

    return pcap_data, seq_data, statistic_data, label_data
