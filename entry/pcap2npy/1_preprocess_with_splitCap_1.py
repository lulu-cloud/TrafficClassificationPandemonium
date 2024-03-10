# -*- coding: utf-8 -*-
# @Time : 2024/3/10 10:18
# @Author : XULu
# @Email : xulu_lili@163.com
# @File : 1_preprocess_with_splitCap_1.py
"""
@Description: 使用splitCap分流工具加统计特征提取
"""


import sys
import os

from preprocess.process_pcap_with_splitCap_1 import split_pcap_2_session, clipping, getPcapMesg, normalization
from utils.split_numpy_data import split_data, split_data_with_spiltCap

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
from utils.set_config import setup_config

from log.set_log import logger

def main():
    yaml_path = r"../../configuration/traffic_classification_configuration.yaml"
    cfg = setup_config(yaml_path) # 获取 config 文件
    logger.info("begin")
    process(
        pcap_path=cfg.preprocess.traffic_path,
        work_flow_data_dir=cfg.preprocess.splitCap_1.work_flow_path,
        tool_path=cfg.preprocess.splitCap_1.splitCap_exe_path,
        npy_path=cfg.preprocess.splitCap_1.datasets,
        train_size=cfg.preprocess.train_size,
        threshold=cfg.preprocess.threshold,
        ip_length=cfg.preprocess.ip_length,
        n=cfg.preprocess.packet_num,
        m=cfg.preprocess.byte_num,
    )
    logger.info("over!")


def process(pcap_path, work_flow_data_dir, tool_path,
            npy_path, train_size, threshold, ip_length, n, m):
    """
    1. 切割会话
    2. 提取特征
    3. 归一化
    4. 存储为npy文件
    5. 切分数据集
    """

    split_pcap_2_session(pcap_path, work_flow_data_dir, tool_path)
    clipping(work_flow_data_dir)

    pay, seq, sta, label = getPcapMesg(work_flow_data_dir, threshold, ip_length, n, m)

    pay, seq, sta = normalization(pay, seq, sta)

    split_data_with_spiltCap(pay, seq, sta, label, train_size, npy_path)

    logger.info("over!")

if __name__ == "__main__":
    main()
