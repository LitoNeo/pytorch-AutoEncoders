#!/usr/bin/env python
# -*-coding:utf-8-*-
import sys
sys.path.append('../')

from common.datas import get_mnist_loader
from models import AutoEncoderLayer, StackedAutoEncoder
import torch
from torch.nn import BCELoss
from torch import optim
import torchvision

num_tranin_layer_epochs = 20
num_tranin_whole_epochs = 50
batch_size = 100
shuffle = True


def train_layers(layers_list=None, layer=None, epoch=None, validate=True):
    """
    逐层贪婪预训练 --当训练第i层时, 将i-1层冻结
    :param layers_list:
    :param layer:
    :param epoch:
    :return:
    """
    if torch.cuda.is_available():
        for model in layers_list:
            model.cuda()
    train_loader, test_loader = get_mnist_loader(batch_size=batch_size, shuffle=True)
    optimizer = optim.SGD(layers_list[layer].parameters(), lr=0.001)
    criterion = BCELoss()

    # train
    for epoch_index in range(epoch):
        sum_loss = 0.

        # 冻结当前层之前的所有层的参数  --第0层没有前置层
        if layer != 0:
            for index in range(layer):
                layers_list[index].lock_grad()
                layers_list[index].is_training_layer = False  # 除了冻结参数,也要设置冻结层的输出返回方式

        for batch_index, (train_data, _) in enumerate(train_loader):
            # 生成输入数据
            if torch.cuda.is_available():
                train_data = train_data.cuda()  # 注意Tensor放到GPU上的操作方式,和model不同
            out = train_data.view(train_data.size(0), -1)

            # 对前(layer-1)冻结了的层进行前向计算
            if layer != 0:
                for l in range(layer):
                    out = layers_list[l](out)

            # 训练第layer层
            pred = layers_list[layer](out)

            optimizer.zero_grad()
            loss = criterion(pred, out)
            sum_loss += loss
            loss.backward()
            optimizer.step()
            if (batch_index + 1) % 10 == 0:
                print("Train Layer: {}, Epoch: {}/{}, Iter: {}/{}, Loss: {:.4f}".format(
                    layer, (epoch_index + 1), epoch, (batch_index + 1), len(train_loader), loss
                ))

        if validate:
            pass


def train_whole(model=None, epoch=50, validate=True):
    print(">> start training whole model")
    if torch.cuda.is_available():
        model.cuda()

    # 解锁因预训练单层而冻结的参数
    for param in model.parameters():
        param.require_grad = True

    train_loader, test_loader = get_mnist_loader(batch_size=batch_size, shuffle=shuffle)
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    # criterion = BCELoss()
    criterion = torch.nn.MSELoss()

    # 生成/保存原始test图片 --取一个batch_size
    test_data, _ = next(iter(test_loader))
    torchvision.utils.save_image(test_data, './test_images/real_test_images.png')

    # train
    for epoch_index in range(epoch):
        sum_loss = 0.
        for batch_index, (train_data, _) in enumerate(train_loader):
            if torch.cuda.is_available():
                train_data = train_data.cuda()
            x = train_data.view(train_data.size(0), -1)

            out = model(x)

            optimizer.zero_grad()
            loss = criterion(out, x)
            sum_loss += loss
            loss.backward()
            optimizer.step()

            if (batch_index + 1) % 10 == 0:
                print("Train Whole, Epoch: {}/{}, Iter: {}/{}, Loss: {:.4f}".format(
                    (epoch_index + 1), epoch, (batch_index + 1), len(train_loader), loss
                ))
            if batch_index == len(train_loader) - 1:
                torchvision.utils.save_image(out.view(100, 1, 28, 28), "./test_images/out_{}_{}.png".format(epoch_index, batch_index))

        # 每个epoch验证一次
        if validate:
            if torch.cuda.is_available():
                test_data = test_data.cuda()
            x = test_data.view(test_data.size(0), -1)
            out = model(x)
            loss = criterion(out, x)
            print("Test Epoch: {}/{}, Iter: {}/{}, test Loss: {}".format(
                epoch_index + 1, epoch, (epoch_index + 1), len(test_loader), loss
            ))
            image_tensor = out.view(batch_size, 1, 28, 28)
            torchvision.utils.save_image(image_tensor, './test_images/test_image_epoch_{}.png'.format(epoch_index))
    print("<< end training whole model")


if __name__ == '__main__':
    import os
    if not os.path.exists('test_images'):
        os.mkdir('test_images')
    if not os.path.exists('models'):
        os.mkdir('models')

    nun_layers = 5
    encoder_1 = AutoEncoderLayer(784, 256, SelfTraining=True)
    encoder_2 = AutoEncoderLayer(256, 64, SelfTraining=True)
    decoder_3 = AutoEncoderLayer(64, 256, SelfTraining=True)
    decoder_4 = AutoEncoderLayer(256, 784, SelfTraining=True)
    layers_list = [encoder_1, encoder_2, decoder_3, decoder_4]

    # 按照顺序对每一层进行预训练
    for level in range(nun_layers - 1):
        train_layers(layers_list=layers_list, layer=level, epoch=num_tranin_layer_epochs, validate=True)

    # 统一训练
    SAE_model = StackedAutoEncoder(layers_list=layers_list)
    train_whole(model=SAE_model, epoch=num_tranin_whole_epochs, validate=True)

    # 保存模型 refer: https://pytorch.org/docs/master/notes/serialization.html
    torch.save(SAE_model, './models/sae_model.pt')
