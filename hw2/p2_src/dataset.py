import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset 

mean = [0.5,0.5,0.5]
std = [0.5,0.5,0.5]

class ImgDataset(Dataset):
    def __init__(self, data_paths, data_labels):
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
        label = torch.LongTensor([self.labels[index]])
        return img, label

class EvalImgDataset(Dataset):
    def __init__(self, data_paths, data_labels):
        self.paths = data_paths
        self.labels = data_labels
        self.transform = transforms.Compose([
                            transforms.Resize(28),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ])
       
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        img = Image.open(self.paths[index]).convert('RGB')
        img = self.transform(img).float()
        label = torch.LongTensor([self.labels[index]])
        return img, label