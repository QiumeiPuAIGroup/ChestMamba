# IMPORT PACKAGES
from PIL import Image
import os

import torch
from torchvision import transforms
from torch.utils import data
import torchvision.utils
from torch.utils.data import random_split


def make_data_path_list(data_path):
    img_list = []

    img_name_list = os.listdir(data_path)
    for i in img_name_list:
        img_list.append(os.path.join(data_path, i))
    
    return img_list


class ImageTransformGan():
    def __init__(self):
        self.data_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((256,256)),
            transforms.Normalize((0.5),(0.5)),
            # transforms.RandomHorizontalFlip(0.5),
            # transforms.RandomRotation(15)
        ])

    def __call__(self, img):
        return self.data_transform(img)
    

class GAN_Img_Dataset(data.Dataset):
    def __init__(self, file_list, transform):
        super().__init__()
        self.file_list = file_list
        self.transform = transform

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        img_path = self.file_list[index]
        img = Image.open(img_path).convert('L')

        img_transformed = self.transform(img)
        return img_transformed
    
def get_dataloader_gan(data_path, batch_size=8, num_workers=2):
    file_list = make_data_path_list(data_path)

    dataset = GAN_Img_Dataset(file_list, transform=ImageTransformGan())

    train_size = int(len(dataset) * 0.7)
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(1234))

    train_dataloader = data.DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    test_dataloader = data.DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    return train_dataloader, test_dataloader
