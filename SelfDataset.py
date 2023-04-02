import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

file_list =os.listdir('./data/train')
cat_files = [file_name for file_name in file_list if 'cat' in file_name]
dog_files = [file_name for file_name in file_list if 'dog' in file_name]

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])


class CatDogDataset(Dataset):
    def __init__(self, file_list, dir, transform=None):
        self.file_list = file_list
        self.dir = dir
        self.transform = transform
        if 'dog' in self.file_list[0]:
            self.label = 1
        else:
            self.label = 0

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.dir, self.file_list[idx])
        img = Image.open(file_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, self.label


dir_path = './data/train/'
cat_dataset = CatDogDataset(cat_files, dir_path, transform=transform)
dog_dataset = CatDogDataset(dog_files, dir_path, transform=transform)

cat_dog_dataset = ConcatDataset([cat_dataset, dog_dataset])
data_loader = DataLoader(cat_dog_dataset, batch_size=32, shuffle=True)
data_iter = iter(data_loader)
imgs, labels = next(data_iter)
print(labels)

grid_imgs = torchvision.utils.make_grid(imgs[:24])
grid_imgs_arr = grid_imgs.numpy()

plt.figure(figsize=(16, 24))
plt.imshow(np.transpose(grid_imgs_arr, (1, 2, 0)))
plt.show()
