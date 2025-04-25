# IMPORT PACKAGES
from PIL import Image
import yaml
import os

from torchvision import transforms
from torch.utils import data
from torch.utils.data import random_split
import torch

class COVIDDataset(data.Dataset):
    def __init__(self, root, transform=None, train=True):
        super().__init__()
        self.transform = transform

        if train is True:
            self.normal_path = os.path.join(root, 'train/NORMAL')
            self.covid_path = os.path.join(root, 'train/COVID19')
            self.pneumonia_path = os.path.join(root, 'train/PNEUMONIA')
        else:
            self.normal_path = os.path.join(root, 'test/NORMAL')
            self.covid_path = os.path.join(root, 'test/COVID19')
            self.pneumonia_path = os.path.join(root, 'test/PNEUMONIA')

        self.normal_list = os.listdir(self.normal_path)
        self.covid_list = os.listdir(self.covid_path)
        self.pneumonia_list = os.listdir(self.pneumonia_path)

        self.normal_path_list = [os.path.join(self.normal_path, i) for i in self.normal_list]
        self.covid_path_list = [os.path.join(self.covid_path, i) for i in self.covid_list]
        self.pneumonia_path_list = [os.path.join(self.pneumonia_path, i) for i in self.pneumonia_list]

        self.normal_label_list = [0] * len(self.normal_path_list)
        self.covid_label_list = [1] * len(self.covid_path_list)
        self.pneumonia_label_list = [2] * len(self.pneumonia_path_list)

        self.img_list = self.normal_path_list + self.covid_path_list + self.pneumonia_path_list
        self.label_list = self.normal_label_list + self.covid_label_list + self.pneumonia_label_list

    def __len__(self):
        return len(self.img_list)
    
    def __getitem__(self, index):
        img_path = self.img_list[index]
        label = self.label_list[index]

        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, label
    
if __name__ == '__main__':
    transform_ = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize([256,256])
    ])
    train_dataset = COVIDDataset(root=r'data\chestxray_dataset', transform=transform_)
    test_dataset = COVIDDataset(root=r'data\chestxray_dataset', transform=transform_, train=False)

    print(len(train_dataset))
    print(len(test_dataset))
    # print(dataset.covid_path_list)
    
