import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset 

mean = [0.5,0.5,0.5]
std = [0.5,0.5,0.5]

class ImgDataset(Dataset):
    def __init__(self, data_paths):
        self.paths = data_paths
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize(mean, std),
                        ])
       
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, index):
        img = Image.open(self.paths[index])
        img = self.transform(img).float()
        return img