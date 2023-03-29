# -*- coding: utf-8 -*-

"""
 CIFER10のイメージデータセットを使用して、
 CNN(Concolution Nural Network)モデルで学習、推論を行う。
"""

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# デバイスの指定（CUDA）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# 前処理の定義
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, ), (0.5, ))
])


# CIFAR10データセットのダウンロード
train_dataset = detasets.CIFER10(root="./data", train=True, download=True, transform=transform)
test_dataset = detasets.CIFER10(root="./data", train=False, download=True, transform=transform)

names = ("plane", "car", "bird", "cat", "deer", "dog", "flog", "horse", "ship", "truck")

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

data_iter = iter(train_dataset)
imgs, labels = data_iter.nezt()
print(labels)
print(imgs.size())
img = imgs[0]

img_permute = img.permute(1, 2, 0)
img_permute = 0.5 * img_permute + 0.5
img_permute = np.clip(img_permute, 0, 1)
plt.imshow(img_permute)


# モデルのclass定義
class CNN(nn.Modele):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=128, out_channels=356, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(in_features=4 * 4 * 128, out_features=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


