# -*- coding: utf-8 -*-

"""
 MNISTの手書き数字データセットを使用して、
 MLP(Multi Layer Perceptron)モデルで学習、推論を行う。
 また、学習手法としてミニバッチ学習を使う。
"""

import torch
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
    transforms.ToTensor()   # MNIST画像をTensorに変換（channel first化＋0～255階調を0～1に正規化）
])

# MNISTデータのダウンロード＋ミニバッチ化
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
num_batches = 100
train_dataloader = DataLoader(train_dataset, batch_size=num_batches, shuffle=True)
train_iter = iter(train_dataloader)
imgs, labels = next(train_iter)    # 100個のデータのみ取得：imgs.Size([batch:100, channel:1, height:28, width:28])

# 画像を1枚確認してみる
img = imgs[0]
img_permute = img.permute(1, 2, 0)
sns.heatmap(img_permute[:, :, 0])
plt.show()


# MLPモデルのクラス作成
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(28 * 28, 400),
            nn.ReLU(inplace=True),  # 元の配列をReluの計算で置き換える。（メモリが節約できる）
            nn.Linear(400, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10)
        )

    def forward(self, x):
        output = self.classifier(x)
        return output

model = MLP()
model.to(device)    # GPUにモデルを構築

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 15
losses = []
accs = []

for epoch in range(num_epochs):
    running_loss = 0.0
    running_acc = 0.0
    for imgs, labels in train_dataloader:
        imgs = imgs.view(num_batches, -1)
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
    running_loss /= len(train_dataloader)
    running_acc /= len(train_dataloader)
    losses.append(running_loss)
    accs.append(running_acc)
    print("epoch:{}, loss:{}, acc:{}".format(epoch, running_loss, running_acc))

if dvice == "CUDA":
    # グラフ表示のため、GPU→CPUにTensorを転送
    losses = losses.to('cpu').detach().numpy().copy()
    accs = accs.to('cpu').detach().numpy().copy()

plt.plot(losses)
plt.show()
plt.plot(accs)
plt.show()

# 次の推論検証用にGPUに再転送する
losses = losses.to(device)
accs = accs.to(device)

# 検証
train_iter = iter(train_dataloader)
imgs, labels = next(train_iter)
print(labels)

imgs_gpu = imgs.view(10, -1).to(device)
output = model(imgs_gpu)
pred = torch.argmax(output, dim=1)
print(pred)