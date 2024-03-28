# -*- coding: utf-8 -*-
# @Time : 2024/3/28 20:40
# @Author : XULu
# @Email : xulu_lili@163.com
# @File : cnn2d.py
"""
@Description: 二维卷积神经网络
"""
from math import sqrt

import torch
import torch.nn as nn
from models.base_model import BaseModel


class Cnn2d(BaseModel):
    def __init__(self, num_classes=12):
        super(Cnn2d, self).__init__()
        # 卷积层+池化层
        self.features = nn.Sequential(
            nn.Conv2d(kernel_size=5,in_channels=1,out_channels=32,stride=1,padding=2), # b,32,32,32
            nn.MaxPool2d(kernel_size=2), # b,32,16,16
            nn.Conv2d(kernel_size=5,in_channels=32,out_channels=64,stride=1,padding=2), # b,64,16,16
            nn.MaxPool2d(kernel_size=2), # b,64,8,8
        )
        # 全连接层
        self.classifier = nn.Sequential(
            # 29*64
            nn.Flatten(),
            nn.Linear(in_features=64 * 64, out_features=1024),  # 1024:64*64
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

    def forward(self, pay, seq, sta):
        pay, seq, sta = self.data_trans(pay, seq, sta)

        pay = self.features(pay)  # 卷积层, 提取特征
        pay = self.classifier(pay)  # 分类层, 用来分类
        return pay, None

    def data_trans(self, x_payload, x_sequence, x_sta):
        # 转换
        x_0,x_1,x_2 = x_payload.shape[0],x_payload.shape[1],x_payload.shape[2]
        x_payload = x_payload.reshape(x_0,x_1,int(sqrt(x_2)),int(sqrt(x_2)))
        return x_payload, x_sequence, x_sta


def cnn2d(model_path, pretrained=False, **kwargs):
    """
    CNN 1D model architecture

    Args:
        pretrained (bool): if True, returns a model pre-trained model
    """
    model = Cnn2d(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
    return model


def main():
    a = sqrt(1024)
    x_pay = torch.rand(8,1,1024)
    cnn = Cnn2d()
    x = cnn(x_pay,x_pay,x_pay)

if __name__=="__main__":
    main()