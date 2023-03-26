# -*- coding: utf-8 -*-
# 手書き文字認識のトレーニング
from sklearn.datasets import load_digits
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


# デバイスの指定（CUDA）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


# データセットの読み込み
digits = load_digits()
X, y = digits.data, digits.target
print(X.shape, y.shape)


# Pytorchのデータ形式に変換
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64)

if torch.cuda.is_available():
    X.to(device)
    y.to(device)
    print('GPUで計算を実行します')
else:
    print('CPUで計算を実行します')

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)


# モデル（多層nn）の定義
model = nn.Sequential(
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 16),
    nn.ReLU(),
    nn.Linear(16, 10)
)
model.train()
lossfun = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)


# 学習の開始
losses = []

for ep in range(100):
    optimizer.zero_grad()
    # yの予測値を算出
    out = model(X)

    # 損失を計算
    loss = lossfun(out, y)
    loss.backward()

    # 勾配を計算
    optimizer.step()

    losses.append(loss.item())

_, pred = torch.max(out, 1)
print((pred == y).sum().item() / len(y))


# 損失のグラフ出力
plt.plot(losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()


if __name__ == '__main__':
    pass