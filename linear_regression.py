# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")

# デバイスの指定（CUDA）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# 乱数のseedを固定
torch.manual_seed(123)

# ばらつきデータの作成
a = 3
b = 2
x = torch.linspace(0, 5, 100).view(100, 1)  # start:0. stop:1, N:100の行列（等差数列）

eps = torch.randn(100, 1)  # 平均:0, 分散:1, 行列:[100, 1]の正規分布乱数
y = a * x + b + eps
plt.scatter(x, y)
plt.show()


# 線形回帰モデルのクラス作成
class LR(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        output = self.linear(x)
        return output


model = LR()

# 未学習状態で予測してみる
x_test = torch.tensor([[1.0], [2.0]])
model(x_test)

x2 = torch.linspace(0, 3, 100).view(100, 1)
y_pred = model(x2)
plt.plot(x2, y_pred.detach(), label='prediction')   # detach()メソッド:勾配計算をしない
plt.scatter(x, y, label='data')
plt.legend()
plt.show()


# 損失関数と最適化アルゴリズムの設定
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# モデルの学習
losses = []
num_epoch = 500

for epoch in range(num_epoch):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print("epoch:{}, loss:{}".format(epoch, loss.item()))   # item()メソッド:要素の値を取得　※複数要素は取り出せない
        losses.append(loss.item())

plt.plot(losses)
plt.show()

# テストデータで推論を実行
x_test = torch.linspace(0, 5, 100).view(100, 1)
y_test = model(x_test)
plt.plot(x_test, y_test.detach(), label='prediction')
plt.scatter(x, y, label='data')
plt.legend()
plt.show()
