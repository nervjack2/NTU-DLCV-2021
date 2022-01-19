import torch
import torchvision.transforms as transforms
import numpy as np 
from PIL import Image 
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler

class ImgDataset(Dataset):
    def __init__(self, data):
        self.paths = data[0]
        self.labels = data[1]
        self.images = [Image.open(path).convert('RGB') for path in self.paths] 
        self.transforms = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        image = self.transforms(self.images[idx])
        label = torch.Tensor([self.labels[idx]]).unsqueeze(0)
        return image, label

class ImgSampler(Sampler):
    def __init__(self, label_list, hp):
        self.hp = hp
        self.labels = torch.Tensor(label_list)
        self.classes, self.counts = torch.unique(self.labels, return_counts=True)
        self.indexes = torch.empty((len(self.classes), max(self.counts)), dtype=torch.int32)
        self.n_data = torch.empty(len(self.classes), dtype=torch.int32)
        for i, c in enumerate(self.classes):
            c_data_idx = torch.nonzero(self.labels == c).view(1,-1)
            self.indexes[i, :c_data_idx.shape[1]] = c_data_idx
            self.n_data[i] = c_data_idx.shape[1]

    def __iter__(self):
        total_num_per_cls = self.hp.n_query + self.hp.k_shot
        for it in range(self.hp.n_eps_per_epoch):
            batch = torch.LongTensor(self.hp.batch_size)
            batch_cls_idx = torch.randperm(len(self.classes))[:self.hp.n_way] 
            for i, c_idx in enumerate(batch_cls_idx):
                s = slice(i*total_num_per_cls, (i+1)*total_num_per_cls)
                class_data_idx = torch.randperm(self.n_data[c_idx])[:total_num_per_cls]
                batch[s] = self.indexes[c_idx, class_data_idx]
            yield batch 
    def __len__(self):
        return self.hp.n_eps_per_epoch        