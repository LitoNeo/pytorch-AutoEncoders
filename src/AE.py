#!/usr/bin/env python
# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.dataloader as dataloader

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import os
import time
import matplotlib.pyplot as plt
from PIL import Image

num_epochs = 20
hidden_layer_sizes = 30
batch_size = 100
if not os.path.exists('data'):
    os.mkdir('data')

# train dataset
dataset = datasets.MNIST(
    '../data',
    train=True,
    transform=transforms.ToTensor(),  # 直接转换成Tensor
    download=True
)
data_loader = dataloader.DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True,
)
data_iter = iter(data_loader)  # data_loader is iterable
images_batch_0, labels_batch_0 = next(data_iter)
if torch.cuda.is_available():
    images_batch_0 = images_batch_0.cuda()
torchvision.utils.save_image(images_batch_0, './data/real_image.png')

# validate dataset
testset = datasets.MNIST(
    '../data',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
testdata_loader = dataloader.DataLoader(
    dataset=testset,
    batch_size=batch_size,
    shuffle=True
)
testdata_iter = iter(testdata_loader)
test_images, _ = next(testdata_iter)
if torch.cuda.is_available():
    test_images = test_images.cuda()
torchvision.utils.save_image(test_images, './data/origin_test_images.png')
image_real = Image.open('./data/origin_test_images.png')


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

    def forward(self, *input):
        out = self.encoder(*input)
        out = self.decoder(out)
        return out


in_dim = images_batch_0.size(2) * images_batch_0.size(3)

autoEncoder = AutoEncoder(in_dim=in_dim, hidden_size=10, out_dim=in_dim)
if torch.cuda.is_available():
    autoEncoder.cuda()  # 注:将模型放到GPU上,因此后续传入的数据必须也在GPU上

Loss = nn.BCELoss()
Optimizer = optim.Adam(autoEncoder.parameters(), lr=0.001)

for epoch in range(num_epochs):
    t_epoch_start = time.time()
    for i, (image_batch, _) in enumerate(data_loader):
        # flatten batch
        image_batch = image_batch.view(image_batch.size(0), -1)
        if torch.cuda.is_available():
            image_batch = image_batch.cuda()
        predict = autoEncoder(image_batch)

        Optimizer.zero_grad()
        loss = Loss(predict, image_batch)
        loss.backward()
        Optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch {}/{}, Iter {}/{}, loss: {:.4f}, time: {:.2f}s'.format(
                epoch + 1, num_epochs, (i + 1), len(dataset) // batch_size, loss.data, time.time() - t_epoch_start
            ))
    val_out = autoEncoder(test_images.view(test_images.size(0), -1).cuda())
    val_out = val_out.view(test_images.size(0), 1, 28, 28)
    filename = './data/reconstruct_images_{}.png'.format(epoch + 1)
    torchvision.utils.save_image(val_out, filename)

    # # show image
    # img_reconstructed = Image.open(filename)
    # plt.figure()
    # plt.subplot(1,2,1),plt.title('real_images')
    # plt.imshow(image_real), plt.axis('off')
    # plt.subplot(1,2,2), plt.title('reconstructed_images')
    # plt.imshow(img_reconstructed), plt.axis('off')
    # plt.show()
