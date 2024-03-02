# -*- coding: utf-8 -*-
# @Time : 2024/3/2 19:38
# @Author : XULu
# @Email : xulu_lili@163.com
# @File : process_pcap_with_flowcontainer.py
"""
@Description: 使用flowcontainer方式预处理pcap文件，输入pcap，输出pay、seq
"""

import os
from log.set_log import init_logger
from flowcontainer.extractor import extract
import numpy as np

logger = init_logger(log_path= '../log/log_file/preprocess-flowcontainer.log')

def hex_to_dec(hex_str, target_length):
    dec_list = []
    for i in range(0, len(hex_str), 2):
        dec_list.append(int(hex_str[i:i + 2], 16))
    dec_list = pad_or_truncate(dec_list, target_length)
    return dec_list


def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [0] * (target_len - len(some_list))



def get_pay_seq(pcap, threshold, ip_length, n, m):
    """

    :param pcap: 原始pcap
    :param n: 前n个包
    :param m: 前m字节
    :return:
    """
    result = extract(pcap, extension=['tcp.payload', 'udp.payload'])
    # 假设有k个流
    pay_load = []
    seq_load = []
    for key in result:
        value = result[key]
        ip_len = value.ip_lengths
        if len(ip_len) < threshold:
            continue
        # 统一长度
        ip_len = pad_or_truncate(ip_len,ip_length)
        seq_load.append(ip_len)

        packet_num = 0
        if 'tcp.payload' in value.extension:
            # 提取tcp负载
            tcp_payload = []
            for packet in value.extension['tcp.payload']:
                if packet_num < n:
                    # packet[0]是负载，1是标注该报文在流的顺序
                    load = packet[0]
                    tcp_payload.extend(hex_to_dec(load, m))
                    packet_num += 1
                else:
                    break
            # 当前包数太少，加0
            if packet_num < n:
                tcp_payload = pad_or_truncate(tcp_payload, m * n)
            pay_load.append(tcp_payload)
        elif 'udp.payload' in value.extension:
            # 提取ucp负载
            udp_payload = []
            for packet in value.extension['udp.payload']:
                if packet_num < n:
                    # packet[0]是负载，1是标注该报文在流的顺序
                    load = packet[0]
                    udp_payload.extend(hex_to_dec(load, m))
                    packet_num += 1
                else:
                    break
            # 当前包数太少，加0
            if packet_num < n:
                udp_payload = pad_or_truncate(udp_payload, m * n)
            pay_load.append(udp_payload)
    pay_load = np.array(pay_load)
    seq_load = np.array(seq_load)
    return pay_load,seq_load


def getPcapIPLength(pcap_folder, threshold, ip_length, packet_num, byte_num):
    """提取序列长度，与前packet_num个报文的前byte_num字节,当流序列长度小于threshold忽略该流"""
    label2num = {}
    class_num = 0
    pay_list = []
    seq_list = []
    label_list = []
    for label_name in os.listdir(pcap_folder):
        label2num[label_name] = class_num
        label_path = os.path.join(pcap_folder, label_name)
        for pcap in os.listdir(label_path):
            # 遍历每一个pcap,提取前packet_num个包的byte_num字节
            pcap = os.path.join(label_path, pcap)
            # 生成ip长度序列seq与负载数据pay
            pay, seq = get_pay_seq(pcap, threshold, ip_length, packet_num, byte_num)

            label = np.full((seq.shape[0],), class_num)
            pay_list.extend(pay)
            seq_list.extend(seq)
            label_list.extend(label)

        class_num += 1
        logger.info("提取完成了{}的数据".format(label_name))
    pay_list = np.array(pay_list)
    seq_list = np.array(seq_list)
    label_list = np.array(label_list)
    logger.info("提取完成了数据")
    logger.info("标签与序号的对应关系:{}".format(label2num))
    num2label = {}
    for (key, value) in label2num.items():
        num2label[value] = key
    logger.info("序号与标签的对应关系:{}".format(num2label))

    return pay_list, seq_list, label_list


