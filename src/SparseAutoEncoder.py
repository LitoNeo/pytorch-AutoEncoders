#!/usr/bin/env python
# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.nn.functional
import torch.optim as optim
import torch.utils.data.dataloader as dataloader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from common.datas import get_mnist_loader

import os
import time
import matplotlib.pyplot as plt
from PIL import Image

batch_size = 100
num_epochs = 50
in_dim = 784
hidden_size = 30
expect_tho = 0.05


def KL_devergence(p, q):
    """
    Calculate the KL-divergence of (p,q)
    :param p:
    :param q:
    :return:
    """
    q = torch.nn.functional.softmax(q, dim=0)
    q = torch.sum(q, dim=0)/batch_size  # dim:缩减的维度,q的第一维是batch维,即大小为batch_size大小,此处是将第j个神经元在batch_size个输入下所有的输出取平均
    s1 = torch.sum(p*torch.log(p/q))
    s2 = torch.sum((1-p)*torch.log((1-p)/(1-q)))
    return s1+s2


class AutoEncoder(nn.Module):
    def __init__(self, in_dim=784, hidden_size=30, out_dim=784):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        return encoder_out, decoder_out


train_loader, test_loader = get_mnist_loader(batch_size=batch_size, shuffle=True)
autoEncoder = AutoEncoder(in_dim=in_dim, hidden_size=hidden_size, out_dim=in_dim)
if torch.cuda.is_available():
    autoEncoder.cuda()  # 注:将模型放到GPU上,因此后续传入的数据必须也在GPU上

Loss = nn.BCELoss()
Optimizer = optim.Adam(autoEncoder.parameters(), lr=0.001)

# 定义期望平均激活值和KL散度的权重
tho_tensor = torch.FloatTensor([expect_tho for _ in range(hidden_size)])
if torch.cuda.is_available():
    tho_tensor = tho_tensor.cuda()
_beta = 3

# def kl_1(p, q):
#     p = torch.nn.functional.softmax(p, dim=-1)
#     _kl = torch.sum(p*(torch.log_softmax(p,dim=-1)) - torch.nn.functional.log_softmax(q, dim=-1),1)
#     return torch.mean(_kl)

for epoch in range(num_epochs):
    time_epoch_start = time.time()
    for batch_index, (train_data, train_label) in enumerate(train_loader):
        if torch.cuda.is_available():
            train_data = train_data.cuda()
            train_label = train_label.cuda()
        input_data = train_data.view(train_data.size(0), -1)
        encoder_out, decoder_out = autoEncoder(input_data)
        loss = Loss(decoder_out, input_data)

        # 计算并增加KL散度到loss
        _kl = KL_devergence(tho_tensor, encoder_out)
        loss += _beta * _kl

        Optimizer.zero_grad()
        loss.backward()
        Optimizer.step()

        print('Epoch: {}, Loss: {:.4f}, Time: {:.2f}'.format(epoch + 1, loss, time.time() - time_epoch_start))
