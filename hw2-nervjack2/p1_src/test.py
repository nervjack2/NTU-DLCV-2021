from is_metrice import inception_score
from dataset import ImgDataset
import torch
import torchvision.transforms as transforms
import numpy as np
import os 
from PIL import Image
from torch.utils.data import Dataset 
from os.path import join
from os import listdir
from model import Generator, Discriminator
from hyper import hyper as hp
from utils import same_seeds, gen_img

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

device = 'cuda'
model_G = Generator(hp.vec_dim, hp.latent_dim).to(device)
model_G.eval()
model_G.load_state_dict(torch.load('../best_generator_p1.mdl'))
same_seeds(4)
gen_img(model_G, '../p1_result', hp, device)
os.system(f'python -m pytorch_fid ../hw2_data/face/test ../p1_result')
    

data_dir = '../p1_result'
data_names = listdir(data_dir)
data_paths = [join(data_dir, name) for name in data_names]
dataset = ImgDataset(data_paths)
score = inception_score(dataset, resize=True)
print(score)
