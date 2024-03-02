# -*- coding: utf-8 -*-
# @Time : 2024/3/2 20:00
# @Author : XULu
# @Email : xulu_lili@163.com
# @File : 1_preprocess_with_flowcontainer.py
"""
@Description: 使用flowcontainer预处理pcap文件
"""



from utils.set_config import setup_config
from preprocess.process_pcap_with_flowcontainer import getPcapIPLength
from utils.split_numpy_data import split_data

def main():
    yaml_path = r"../configuration/traffic_classification_configuration.yaml"
    cfg = setup_config(yaml_path) # 获取 config 文件
    pay, seq, label = getPcapIPLength(
        cfg.preprocess.traffic_path,
        cfg.preprocess.threshold,
        cfg.preprocess.ip_length,
        cfg.preprocess.packet_num,
        cfg.preprocess.byte_num)
    split_data(pay,seq,label,cfg.preprocess.train_size,cfg.preprocess.datasets)

if __name__=="__main__":
    main()
