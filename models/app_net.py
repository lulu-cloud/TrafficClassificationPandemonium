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


class APP_Net(nn.Module):
    def __init__(self, input_size = 1, hidden_size = 256, num_layers = 2, bidirectional = True, num_classes=12):
        super(APP_Net, self).__init__()
        # rnn配置
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = num_layers, bidirectional=bidirectional, batch_first=True)
        self.fc0 = nn.Linear(hidden_size, num_classes)
        self.fc1 = nn.Linear(hidden_size * 2, num_classes)

        self.cnn_feature = nn.Sequential(
            # 卷积层1
            nn.Conv1d(kernel_size=25, in_channels=1, out_channels=32, stride=1, padding=12),  # (1,1024)->(32,1024)
            nn.BatchNorm1d(32),  # 加上BN的结果
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),  # (32,1024)->(32,342)

            # 卷积层2
            nn.Conv1d(kernel_size=25, in_channels=32, out_channels=64, stride=1, padding=12),  # (32,342)->(64,342)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3, padding=1),  # (64,342)->(64,114)
        )
        # 全连接层
        self.cnn_classifier = nn.Sequential(
            # 64*114
            nn.Flatten(),
            nn.Linear(in_features=64 * 114, out_features=1024),  # 784:88*64, 1024:114*64, 4096:456*64
        )

        self.cnn = nn.Sequential(
            self.cnn_feature,
            self.cnn_classifier,
        )

        self.rnn = nn.Sequential(
            # (batch_size, seq_len, input_size)
            nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=True),
        )
        self.classifier_bi = nn.Sequential(
            nn.Linear(in_features=1024 + hidden_size * 2, out_features=1024),
            nn.Dropout(p=0.7),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=1024 + hidden_size, out_features=1024),
            nn.Dropout(p=0.7),
            nn.Linear(in_features=1024, out_features=num_classes)
        )

    def forward(self, x_payload, x_sequence,x_sta):
        x_payload = self.cnn(x_payload)
        x_sequence = self.rnn(x_sequence)
        x_sequence = x_sequence[0][:, -1, :]
        x = torch.cat((x_payload, x_sequence), 1)
        if self.bidirectional == True:
            x = self.classifier_bi(x)
        else:
            x = self.classifier(x)
        return x,None


def app_net(model_path, pretrained=False, **kwargs):
    """
    CNN 1D model architecture

    Args:
        pretrained (bool): if True, returns a model pre-trained model
    """
    model = APP_Net(**kwargs)
    if pretrained:
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dict'])
    return model
