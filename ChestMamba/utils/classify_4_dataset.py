# IMPORT PACKAGES
from PIL import Image
import yaml
import os

from torchvision import transforms
from torch.utils import data
from torch.utils.data import random_split
import torch

class COVIDDataset(data.Dataset):
    def __init__(self, root, transform=None):
        super().__init__()
        self.transform = transform

        self.normal_path = os.path.join(root, 'Normal/images')
        self.covid_path = os.path.join(root, 'COVID/images')
        self.viral_path = os.path.join(root, 'Viral Pneumonia/images')
        self.opacity_path = os.path.join(root, 'Lung_Opacity/images')

        self.normal_list = os.listdir(self.normal_path)
        self.covid_list = os.listdir(self.covid_path)
        self.viral_list = os.listdir(self.viral_path)
        self.opacity_list = os.listdir(self.opacity_path)

        self.normal_path_list = [os.path.join(self.normal_path, i) for i in self.normal_list]
        self.covid_path_list = [os.path.join(self.covid_path, i) for i in self.covid_list]
        self.viral_path_list = [os.path.join(self.viral_path, i) for i in self.viral_list]
        self.opacity_path_list = [os.path.join(self.opacity_path, i) for i in self.opacity_list]

        self.normal_label_list = [0] * len(self.normal_path_list)
        self.covid_label_list = [1] * len(self.covid_path_list)
        self.viral_label_list = [2] * len(self.viral_path_list)
        self.opacity_label_list = [3] * len(self.opacity_path_list)

        self.img_list = self.normal_path_list + self.covid_path_list + self.viral_path_list + self.opacity_path_list
        self.label_list = self.normal_label_list + self.covid_label_list + self.viral_label_list + self.opacity_label_list

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
    dataset = COVIDDataset(root=r'data\COVID-19_Radiography_Dataset', transform=transform_)
    train_size = int(len(dataset) * 0.7)
    val_size = int(len(dataset) * 0.1)
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(1234))

    print(len(train_dataset))
    print(len(test_dataset))
    print(len(dataset))
    # print(dataset.covid_path_list)
    
