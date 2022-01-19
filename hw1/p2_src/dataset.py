import torch
import os
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset 
from augmentation import img_augment
from utils import color_to_label

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


class ImgDataset(Dataset):
    def __init__(self, data_dir, train=True, inference=False):
        data_name = sorted(os.listdir(data_dir))
        self.data_paths = sorted([os.path.join(data_dir,x) for x in data_name if x.endswith('jpg')])
        self.labels_paths = sorted([os.path.join(data_dir,x) for x in data_name if x.endswith('png')])
        self.train = train 
        self.inference = inference
        self.augment = img_augment
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std)
                        ])
    def __len__(self):
        return len(self.data_paths)
    
    def __getitem__(self, index):
        img = Image.open(self.data_paths[index])
        if not self.inference:
            label = Image.open(self.labels_paths[index])
        if self.train:
            img, label = self.augment(img, label)
        img = self.transform(img).float()
        if self.inference:
            return img 
        label = color_to_label(label)
        label = torch.LongTensor(label)
        return img, label