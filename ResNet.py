# -*- coding: utf-8 -*-

"""
 ResNetモデルの実装（学習せず、モデル構築のみ）
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def shortcut(self, x):
        x = self.conv3(x)
        x = self.bn(x)
        return x

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x += self.shortcut(identity)
        return x


class ResNet(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.linear = nn.Linear(in_features=28*28*64, out_features=10)
        self.layer = self._make_layer(block, 3, 3, 64)

    def _make_layer(self, block, num_residual_blocks, in_channels, out_channels):
        layers = []
        for i in range(num_residual_blocks):
            if i == 0:
                layers.append(block(in_channels, out_channels))
            else:
                layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

model = ResNet(ResidualBlock)
print(model)