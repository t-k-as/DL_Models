# -*- coding: utf-8 -*-

"""
 ResNetモデルの転移学習
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import os
import copy
import time

# デバイスの指定（CUDA）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])


# データセットのダウンロード（bee、ant）
data_dir = 'data/hymenoptera_data/train'
train_datasets = datasets.ImageFolder(data_dir, transform=transform)
train_loader = DataLoader(train_datasets, batch_size=32, shuffle=True)
data_iter = iter(train_loader)
imgs, labels = next(data_iter)
print(labels)

img = imgs[0]
img_permute = img.permute(1, 2, 0)
img_permute = 0.5 * img_permute + 0.5
img_permute = np.clip(img_permute, 0, 1)
plt.imshow(img_permute)
plt.show()

"""
# イメージ画像の表示
def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

imgs, labels = next(iter(dataloaders['train']))
out = torchvision.utils.make_grid(imgs)
imshow(out, title=[class_names[x] for x in labels])
"""

# resnetモデルの読み込み
model = models.resnet18(pretrained=True)
print(model)

# 勾配計算のOFF
for param in model.parameters():
    param.requires_grad = False

# fc層の書き換え
model.fc = nn.Linear(512, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

num_epochs = 15
losses = []
accs = []

for epoch in range(num_epochs):
    running_loss = 0.0
    running_acc = 0.0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model(imgs)
        loss = criterion(output, labels)
        running_loss += loss.item()
        pred = torch.argmax(output, dim=1)
        running_acc += torch.mean(pred.eq(labels).float())
        loss.backward()
        optimizer.step()
    running_loss /= len(train_loader)
    running_acc /= len(train_loader)
    losses.append(running_loss)
    running_acc = running_acc.detach().cpu().numpy()  # グラフ表示のため、GPU→CPUにTensorを転送
    accs.append(running_acc)
    print("epoch:{}, loss:{}, acc:{}".format(epoch, running_loss, running_acc))

plt.plot(losses)
plt.show()
plt.plot(accs)
plt.show()


"""
# モデルの保存
params = model.state_dict()
torch.save(params, "model_ResNet.prm")

# モデルのロード
param_load = torch.load("model_ResNet.prm")
model.load_state_dict(param_load)
"""
