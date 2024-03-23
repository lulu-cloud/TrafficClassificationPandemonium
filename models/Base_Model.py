# -*- coding: utf-8 -*-
# @Time : 2024/3/23 16:50
# @Author : XULu
# @Email : xulu_lili@163.com
# @File : Base_Model.py
"""
@Description: 
"""

# -*- coding: utf-8 -*-
# @Time : 2024/3/4 14:22
# @Author : XULu
# @Email : xulu_lili@163.com
# @File : app_net.py
"""
@Description: APP_Net模型
"""

import torch
import torch.nn as nn
from abc import ABC,abstractmethod

class BaseModel(nn.Module):
    def __init__(self):
        super(BaseModel, self).__init__()

    @abstractmethod
    def forward(self, x_payload, x_sequence,x_sta):
        x_payload, x_sequence, x_sta = self.data_trans(x_payload, x_sequence,x_sta)
        # 第一个是分类结果，第二个是重构结果
        return None,None
    @abstractmethod
    def data_trans(self,x_payload, x_sequence,x_sta):
        pass


