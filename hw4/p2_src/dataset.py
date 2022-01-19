import os 
import torch
from os.path import join
import torchvision.transforms as transforms
from PIL import Image 
from torch.utils.data import Dataset 


class ImgDataset(Dataset):
    def __init__(self, data_dir, img_size):
        self.paths = [join(data_dir,x) for x in os.listdir(data_dir)]
        self.images = [Image.open(path).convert('RGB') for path in self.paths] 
        self.transforms = transforms.Compose([
                            transforms.Resize(img_size),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.transforms(self.images[idx])

class FineTunedImgDataset(Dataset):
    def __init__(self, data, img_size):
        self.paths = data[0]
        self.labels = data[1]
        self.images = [Image.open(path).convert('RGB') for path in self.paths] 
        self.transforms = transforms.Compose([
                            transforms.Resize((img_size,img_size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        image = self.transforms(self.images[idx])
        label = torch.LongTensor([self.labels[idx]]).unsqueeze(0)
        return image, label

class EvalImgDataset(Dataset):
    def __init__(self, data_path, img_size):
        self.paths = data_path
        self.images = [Image.open(path).convert('RGB') for path in self.paths] 
        self.transforms = transforms.Compose([
                            transforms.Resize((img_size,img_size)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        image = self.transforms(self.images[idx])
        return image