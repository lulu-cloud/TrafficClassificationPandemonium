# -*- coding: utf-8 -*-
# @Time : 2024/3/10 10:27
# @Author : XULu
# @Email : xulu_lili@163.com
# @File : process_pcap_with_splitCap_1.py
"""
@Description: 使用splitCap提取特征
"""

import json
import os
import subprocess
from shutil import move

from sklearn.preprocessing import MinMaxScaler

from preprocess.util.FeaturesCalc import FeaturesCalc
from log.set_log import logger
import scapy.all as scapy
import binascii
import numpy as np
import iisignature

from tqdm import tqdm

def customAction(pcap):
    """对一个 session 中的每一个 packet 进行匿名化的处理

    Args:
        pcap: 每一个 packet 文件
    """
    src_ip = "0.0.0.0"
    src_ipv6 = "0:0:0:0:0:0:0:0"
    src_port = 0
    src_mac = "00:00:00:00:00:00"

    dst_ip = "0.0.0.0"
    dst_ipv6 = "0:0:0:0:0:0:0:0"
    dst_port = 0
    dst_mac = "00:00:00:00:00:00"

    if 'Ether' in pcap:
        pcap.src = src_mac  # 修改源 mac 地址
        pcap.dst = dst_mac  # 修改目的 mac 地址
    if 'IP' in pcap:
        pcap["IP"].src = src_ip
        pcap["IP"].dst = dst_ip
    if 'IPv6' in pcap:
        pcap["IPv6"].src = src_ipv6
        pcap["IPv6"].dst = dst_ipv6
    if 'TCP' in pcap:
        pcap['TCP'].sport = src_port
        pcap['TCP'].dport = dst_port
    if 'UDP' in pcap:
        pcap['UDP'].sport = src_port
        pcap['UDP'].dport = dst_port
    if 'ARP' in pcap:
        pcap["ARP"].psrc = src_ip
        pcap["ARP"].pdst = dst_ip
        pcap["ARP"].hwsrc = src_mac
        pcap["ARP"].hwdst = dst_mac



def hex_to_dec(hex_str, target_length):
    dec_list = []
    for i in range(0, len(hex_str), 2):
        dec_list.append(int(hex_str[i:i + 2], 16))
    dec_list = pad_or_truncate(dec_list, target_length)
    return dec_list


def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [0] * (target_len - len(some_list))


def get_pay_seq_get_pay_seq_statis(pcap, threshold, ip_length, n, m):
    """
    :param pcap: 原始pcap
    :param n: 前n个包
    :param m: 前m字节
    :return:
    """
    featuresCalc = FeaturesCalc(min_window_size=1)  # 初始化计算统计特征的类

    packets = scapy.rdpcap(pcap, count=ip_length)

    if len(packets) < threshold:
        return [],[],[]
    pay_flow = []
    seq_flow = []
    # static_flow = []
    for i, p in enumerate(packets):
        if i == 0:
            dst = p.dst
            src = p.src
            dic = {dst: 1, src: -1}
        word_packet = p.copy()
        # 提取包长
        dst = word_packet.dst
        ip_len = dic[dst] * len(word_packet)
        seq_flow.append(ip_len)
        # 处理负载数据，作为一个一个的word
        if i < n:
            # 匿名 掩盖包mac与ip地址
            customAction(word_packet)
            words_string = (binascii.hexlify(bytes(word_packet)))
            # byte转string
            words_string = words_string.decode()
            payload = hex_to_dec(words_string,m)
            pay_flow.extend(payload)
    seq_flow = pad_or_truncate(seq_flow,ip_length)
    # 处理统计特征
    static_flow = featuresCalc.compute_features(packets_list=packets)
    # static_flow.append(static)
    return pay_flow,seq_flow,static_flow




def getPcapMesg(pcap_folder, threshold, ip_length, packet_num, byte_num):
    """提取序列长度，与前packet_num个报文的前byte_num字节以及统计特征"""
    label2num = {}
    class_num = 0
    pay_list = []
    seq_list = []
    statistic_list = []
    label_list = []
    for label_name in os.listdir(pcap_folder):
        label2num[label_name] = class_num
        label_path = os.path.join(pcap_folder, label_name)
        for pcap in os.listdir(label_path):
            # 遍历每一个pcap,提取前packet_num个包的byte_num字节
            pcap = os.path.join(label_path, pcap)
            # 生成ip长度序列seq与负载数据pay
            pay, seq, statistic = get_pay_seq_get_pay_seq_statis(pcap, threshold, ip_length, packet_num, byte_num)
            if len(pay)==0:
                continue
            label = class_num
            pay_list.append(pay)
            seq_list.append(seq)
            statistic_list.append(statistic)
            label_list.append(label)

        class_num += 1
        logger.info("提取完成了{}的数据".format(label_name))
    pay_list = np.array(pay_list)
    seq_list = np.array(seq_list)
    statistic_list = np.array(statistic_list)
    label_list = np.array(label_list)
    logger.info("提取完成了数据")
    logger.info("标签与序号的对应关系:{}".format(label2num))
    num2label = {}
    for (key, value) in label2num.items():
        num2label[value] = key
    logger.info("序号与标签的对应关系:{}".format(num2label))
    with open("label2num.json", "w") as f:
        json.dump(label2num, f)

    return pay_list, seq_list, statistic_list, label_list




def clipping(pcap_folder):
    logger.info("文件剪切开始辣！！！")
    for apps in os.listdir(pcap_folder):
        logger.info("开始进行{}文件剪切！！！".format(apps))
        apps_path = os.path.join(pcap_folder, apps)
        for flows in os.listdir(apps_path):
            flows_path = os.path.join(apps_path, flows)
            for flow in os.listdir(flows_path):
                flow_path = os.path.join(flows_path, flow)
                # 移动到上级目录
                move(flow_path, apps_path)
            os.removedirs(flows_path)
        logger.info("完成{}文件剪切辣！！！".format(apps))
    logger.info("完成所有文件剪切辣！！！")


def split_pcap_2_session(pcap_folder, traffic_path, splitcap_path):
    splitcap_path = os.path.normpath(splitcap_path)  # 处理成 windows 下的路径格式
    for (root, _, files) in os.walk(pcap_folder):
        # root 是根目录
        # dirs 是在 root 目录下的文件夹, 返回的是一个 list;
        # files 是在 root 目录下的文件, 返回的是一个 list, 所以 os.path.join(root, files) 返回的就是 files 的路径
        for Ufile in files:
            pcap_file_path = os.path.join(root, Ufile)  # pcap 文件的完整路径
            Ufile_list = Ufile.split('.')
            # pcap_name = Ufile_list[0]  # pcap 文件的名字
            pcap_suffix = Ufile_list[-1]  # 文件的后缀名
            # pcap_name = Ufile_list[-1]  # pcap 文件的名字
            Ufile_list.pop()
            pcap_name = ".".join(Ufile_list)
            try:
                assert pcap_suffix == 'pcap'
            except:
                logger.warning('查看 pcap 文件的后缀')
                assert pcap_suffix == 'pcap'
            dir_suffix = root.split("\\")[-1]
            dir_name = os.path.join(traffic_path, dir_suffix)
            os.makedirs(os.path.join(dir_name, pcap_name), exist_ok=True)  # 新建文件夹
            prog = subprocess.Popen([splitcap_path,
                                     "-p", "100000",
                                     "-b", "100000",
                                     "-r", pcap_file_path,
                                     "-o", os.path.join(dir_name, pcap_name)],  # 只提取应用层可以加上, "-y", "L7"
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            _, _ = prog.communicate()

            # os.remove(pcap_file_path) # 删除原始的 pcap 文件
            logger.info('处理完成文件 {}'.format(Ufile))
    logger.info('完成 pcap 转换为 session.')
    logger.info('============\n')


def normalization(pay, seq, sta):
    """数据归一化"""
    pay = pay.astype(float) / 255.0
    seq = MinMaxScaler().fit_transform(seq)
    sta = MinMaxScaler().fit_transform(sta)
    return pay, seq, sta