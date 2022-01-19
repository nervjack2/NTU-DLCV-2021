import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset 
from augmentation import img_augment

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

class ImgDataset(Dataset):
    def __init__(self, data_paths, data_labels=None, train=True):
        self.paths = data_paths
        self.labels = data_labels
        self.augment = img_augment
        self.transform = transforms.Compose([
                            transforms.Resize(299),
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ])
        self.train = train
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        img = Image.open(self.paths[index])
        if self.train:
            img = self.augment(img)
        img = self.transform(img).float()
        if self.labels == None:
            return img 
        label = torch.LongTensor([self.labels[index]])
        return img, label