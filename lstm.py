# -*- coding: utf-8 -*-

"""
 LSTMモデルの学習
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
plt.style.use('ggplot')

# デバイスの指定（CUDA）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

x = np.linspace(0, 499, 500)
y = np.sin(x * 2 * np.pi / 50)
plt.plot(x, y)
plt.show()


def make_sequence_data(y, num_sequence):
    num_data = len(y)
    seq_data = []
    target_data = []
    for i in range(num_data - num_sequence):
        seq_data.append(y[i:i+num_sequence])
        target_data.append(y[i+num_sequence: i+num_sequence+1])
    seq_arr = np.array(seq_data)
    target_arr = np.array(target_data)
    return seq_arr, target_arr


seq_length = 40
y_seq, y_target = make_sequence_data(y, seq_length)
print(y_seq.shape)
print(y_target.shape)

num_test = 10
y_seq_train = y_seq[:-num_test]
y_seq_test = y_seq[-num_test:]
y_target_train = y_target[:-num_test]
y_target_test = y_target[-num_test:]

y_seq_t = torch.FloatTensor(y_seq_train)
y_target_t = torch.FloatTensor(y_target_train)


class LSTM(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=1, hidden_size=self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x_last = x[-1]
        x = self.linear(x_last)
        return x

model = LSTM(100)

criterion =nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

y_seq_t = y_seq_t.permute(1, 0)
y_target_t = y_target_t.permute(1, 0)

y_seq_t = y_seq_t.unsqueeze(dim=-1)
y_target_t = y_target_t.unsqueeze(dim=-1)

num_epochs = 80
losses = []
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(y_seq_t)
    loss = criterion(output, y_target_t)
    loss.backward()
    losses.append(loss.item())
    optimizer.step()
    if epoch % 10 == 0:
        print('epoch:{}, loss:{}'.format(epoch, loss.item()))

plt.plot(losses)
plt.show()


y_seq_test_t = torch.FloatTensor(y_seq_test)
y_seq_test_t = y_seq_test_t.permute(1, 0)
y_seq_test_t = y_seq_test_t.unsqueeze(dim=-1)

y_pred = model(y_seq_test_t)
plt.plot(x, y)
plt.plot(np.arange(490, 500), y_pred.detach())
plt.xlim([450, 500])
plt.show()