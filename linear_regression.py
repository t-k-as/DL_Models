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

torch.manual_seed(123)
a = 3
b = 2
x = torch.linspace(0, 5, 100).view(100, 1)

eps = torch.randn(100, 1)
y = a * x + b + eps
plt.scatter(x, y)
# plt.show()


class LR(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        output = self.linear(x)
        return output

model = LR()

x_test = torch.tensor([[1.0], [2.0]])
model(x_test)

x2 = torch.linspace(0, 3, 100).view(100, 1)

y_pred = model(x2)
plt.plot(x2, y_pred.detach(), label='prediction')
plt.scatter(x, y, label='data')
plt.legend()
plt.show()

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

losses = []
num_epoch = 500

for epoch in range(num_epoch):
    optimizer.zero_grad()
    y_pred = model(x)
    loss = criterion(y_pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print("epoch:{}, loss:{}".format(epoch, loss.item()))
        losses.append(loss.item())

plt.plot(losses)
plt.show()

x_test = torch.linspace(0, 5, 100).view(100, 1)
y_test = model(x_test)
plt.plot(x_test, y_test.detach(), label = 'prediction')
plt.scatter(x, y, label='data')
plt.legend()
plt.show()