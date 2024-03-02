# -*- coding: utf-8 -*-
# @Time : 2024/3/2 19:40
# @Author : XULu
# @Email : xulu_lili@163.com
# @File : set_log.py
"""
@Description: 生成loger
"""

import logging

def init_logger(log_path):
    log_format = logging.Formatter("[%(asctime)s %(levelname)s] %(message)s")
    logger = logging.getLogger()
    logger.setLevel("INFO")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_path is not None:
        file_handler = logging.FileHandler(log_path, encoding="UTF-8")
        file_handler.setLevel("INFO")
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)

    return logger

# logger = init_logger(r"D:\xy12\app-net\traffic_log\log\all.log")