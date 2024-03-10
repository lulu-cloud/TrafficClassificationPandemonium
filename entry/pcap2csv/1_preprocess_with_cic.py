# -*- coding: utf-8 -*-
# @Time : 2024/3/10 14:32
# @Author : XULu
# @Email : xulu_lili@163.com
# @File : 1_preprocess_with_cic.py.py
"""
@Description: 
"""

import os

def generate_batch_csv(pcap_dir,csv_dir,cic_path):
    """批量将pcap转csv"""
    for label in os.listdir(pcap_dir):
        label_path = os.path.join(pcap_dir, label)
        csv_label_dir = os.path.join(csv_dir,label)
        if not os.path.exists(csv_label_dir):
            os.makedirs(csv_label_dir)
        for pcap in os.listdir(label_path):
            pcap_file = os.path.join(label_path,pcap)
            print(f"Analyzing {pcap_file}...")
            # 调用cfm.bat脚本对pcap文件进行分析
            os.chdir(cic_path)
            os.system(f'call cfm.bat "{pcap_file}" "{csv_label_dir}"')
            print("Done.")
        print("完成了{}的处理".format(label))
    print("所有的都处理完成")

def main():
    # 设置pcap和csv文件夹路径
    pcap_dir = r"D:\PyProject\TrafficClassificationPandemonium\traffic_path\android"
    csv_dir = r"D:\PyProject\TrafficClassificationPandemonium\datasets\cic-flowmeter\android"
    cic_dir = r"D:\PyProject\TrafficClassificationPandemonium\tool\CICFlowMeter-4.0\bin"
    generate_batch_csv(pcap_dir,csv_dir,cic_dir)


if __name__ == "__main__":
    main()

