# -*- coding: utf-8 -*-
# @Time : 2024/3/2 20:01
# @Author : XULu
# @Email : xulu_lili@163.com
# @File : set_config.py
"""
@Description: 
"""

import yaml
from easydict import EasyDict

def setup_config(path):
    """获取配置信息
    """
    with open(path, encoding='utf8') as f:
        cfg = yaml.safe_load(f)
    cfg = EasyDict(cfg)
    return cfg