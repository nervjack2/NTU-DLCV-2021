import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
from torch.utils.data import Dataset 
from augmentation import img_augment

class ImgDataset(Dataset):
    def __init__(self, data_paths, data_labels=None, train=True):
        self.paths = data_paths
        self.labels = data_labels
        self.augment = img_augment
        self.transform = transforms.Compose([
                            transforms.Resize((384,384)),
                            transforms.ToTensor(),
                            transforms.Normalize(0.5, 0.5),
                        ])
        self.train = train
        # Filtering out invalid images 
        self.valid_paths = []
        if self.labels != None:
            self.valid_labels = []
        for i, img_path in enumerate(self.paths):
            img = Image.open(img_path).convert('RGB')
            img = self.transform(img).float()
            if img.shape[0] != 3:
                continue
            self.valid_paths.append(img_path)
            if self.labels != None:
                self.valid_labels.append(self.labels[i])

    def __len__(self):
        return len(self.valid_paths)
    def __getitem__(self, index):
        img = Image.open(self.valid_paths[index]).convert('RGB')
        if self.train:
            img = self.augment(img)
        img = self.transform(img).float()
        if self.labels == None:
            return img 
        label = torch.LongTensor([self.valid_labels[index]])
        return img, label