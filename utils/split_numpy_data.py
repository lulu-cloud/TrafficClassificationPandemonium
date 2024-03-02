# -*- coding: utf-8 -*-
# @Time : 2024/3/2 19:54
# @Author : XULu
# @Email : xulu_lili@163.com
# @File : split_numpy_data.py
"""
@Description: 分割numpy data，划分训练测试集合
"""

# -*- coding: utf-8 -*-
# @Time : 2023/5/17 15:47
# @Author :
# @Email :
# @File : split_all_data.py
"""
@Description: 按照训练与测试占比划分数据，存在本地文件夹里面
"""
from log.set_log import init_logger
import os
from sklearn.model_selection import train_test_split
import numpy as np

logger = init_logger(log_path= '../log/log_file/preprocess-flowcontainer.log')

def split_data(pay_data, seq_data,label_data, train_size, npy_path):

    pay_train, pay_test, seq_train,seq_test,label_train, label_test = train_test_split(pay_data, seq_data,label_data, train_size=train_size)
    os.makedirs(os.path.join(npy_path, 'train'), exist_ok=True)
    os.makedirs(os.path.join(npy_path, 'test'), exist_ok=True)
    np.save(os.path.join(npy_path, 'train/', 'pay_load.npy'), pay_train)
    np.save(os.path.join(npy_path, 'test/', 'pay_load.npy'), pay_test)
    np.save(os.path.join(npy_path, 'train/', 'ip_length.npy'), seq_train)
    np.save(os.path.join(npy_path, 'test/', 'ip_length.npy'), seq_test)
    np.save(os.path.join(npy_path, 'train/', 'label.npy'), label_train)
    np.save(os.path.join(npy_path, 'test/', 'label.npy'), label_test)
    logger.info("数据划分为训练集与测试集，训练集比例为{}".format(train_size))