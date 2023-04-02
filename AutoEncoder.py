# -*- coding: utf-8 -*-

"""
 AutoEncoderモデルの転移学習
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import seaborn as sns
import os
import copy
import time

# デバイスの指定（CUDA）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

transform = transforms.Compose([
    transforms.ToTensor()
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.en = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.de = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.en(x)
        x = self.de(x)
        return x

model = ConvAE()
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 15
losses = []
for epoch in range(num_epochs):
    running_loss = 0.0
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, imgs)
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
    running_loss /= len(train_loader)
    losses.append(running_loss)
    print("epoch:{}, loss:{}".format(epoch, running_loss))

plt.style.use("ggplot")
plt.plot(losses, label="train loss")
plt.show()

# 適当なデータで圧縮→復元
data_iter = iter(train_loader)
imgs, _ = next(data_iter)
img = imgs[0]
img_permute = img.permute(1, 2, 0)
sns.heatmap(img_permute[:, :, 0])
plt.show()

x_en = model.en(imgs.to(device))
x_en2 = x_en[0].permute(1, 2, 0)
sns.heatmap(x_en2[:, :, 0].detach().to('cpu'))
plt.show()

x_ae = model(imgs.to(device))
sns.heatmap(x_ae[0].permute(1, 2, 0).detach().to('cpu')[:, :, 0])
plt.show()
