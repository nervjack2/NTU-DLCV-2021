import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset 
from augmentation import img_augment

class ImgDataset(Dataset):
    def __init__(self, data_paths, data_labels=None):
        mean = [0.5,0.5,0.5]
        std = [0.5,0.5,0.5]
        self.paths = data_paths
        self.labels = data_labels
        self.transform = transforms.Compose([
                            transforms.Resize(64),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ])
       
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        img = self.transform(img).float()
        if self.labels != None:
            label = torch.LongTensor([self.labels[index]])
            return img, label
        return img 

class SVHN_ImgDataset(Dataset):
    def __init__(self, data_paths, data_labels=None):
        mean = [0.5,0.5,0.5]
        std =  [0.5,0.5,0.5]
        self.paths = data_paths
        self.labels = data_labels
        self.augment = img_augment
        self.transform = transforms.Compose([
                            transforms.Resize(28),
                            transforms.CenterCrop(28),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ])
       
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        img = self.augment(img)
        img = self.transform(img).float()
        if self.labels != None:
            label = torch.LongTensor([self.labels[index]])
            return img, label
        return img 

class SVHN_EvalImgDataset(Dataset):
    def __init__(self, data_paths, data_labels=None):
        mean = [0.5,0.5,0.5]
        std = [0.5,0.5,0.5]
        self.paths = data_paths
        self.labels = data_labels
        self.transform = transforms.Compose([
                            transforms.Resize(28),
                            transforms.CenterCrop(28),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ])
       
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        img = self.transform(img).float()
        if self.labels != None:
            label = torch.LongTensor([self.labels[index]])
            return img, label
        return img