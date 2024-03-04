# -*- coding: utf-8 -*-
# @Time : 2024/3/4 14:12
# @Author : XULu
# @Email : xulu_lili@163.com
# @File : cnn1d.py
"""
@Description: 一维卷积神经网络
"""


import torch
import torch.nn as nn
import netron
class Cnn1d(nn.Module):
    def __init__(self, num_classes=12):
        super(Cnn1d, self).__init__()
        # 卷积层+池化层
        self.features = nn.Sequential(
            # 256->256
            nn.Conv1d(kernel_size=25, in_channels=1, out_channels=32, stride=1, padding=12), # (1,768)->(32,768)
            nn.BatchNorm1d(32), # 加上BN的结果
            nn.ReLU(),
            # 256->86
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1), # (32,768)->(32,256)
            # 86->86
            nn.Conv1d(kernel_size=25, in_channels=32, out_channels=64, stride=1, padding=12), # (32,256)->(64,256)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            # 86->29
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1), # (64,256)->(64*86)
        )
        # 全连接层
        self.classifier = nn.Sequential(
            # 29*64
            nn.Flatten(),
            nn.Linear(in_features=114*64, out_features=1024),  # 768:86*64 # 1024:114*64 # 1500:167*64
            nn.Dropout(0.5),
            nn.Linear(in_features=1024, out_features=num_classes)
        )
    def forward(self, pay,seq,sta):
        # x = x.view(x.size(0),1,-1) # 将图片摊平
        pay = self.features(pay) # 卷积层, 提取特征
        # print(x.shape)
        # x = x.view(x.size(0), -1) # 展开
        pay = self.classifier(pay) # 分类层, 用来分类
        return pay,None

def cnn1d(model_path, pretrained=False, **kwargs):
    """
    CNN 1D model architecture

    Args:
        pretrained (bool): if True, returns a model pre-trained model
    """
    model = Cnn1d(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
    return model

