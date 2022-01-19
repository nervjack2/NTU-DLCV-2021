import torch
import torch.nn as nn
import numpy as np 
import os 
import sys
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from os import listdir
from os.path import join
from dataset import ImgDataset, SVHN_EvalImgDataset
from model import DANN, DANN_SVHN
from hyper import hyper as hp
from utils import same_seeds, get_data

def main(
    target_dir: str,
    target_domain: str,
    save_path: str,
    model_path: str 
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_names = sorted(os.listdir(target_dir))
    t_data_paths = [join(target_dir, x) for x in data_names]
    if target_domain != 'svhn':
        t_dataset = ImgDataset(t_data_paths)
    else:
        t_dataset = SVHN_EvalImgDataset(t_data_paths)
    t_dataloader = DataLoader(t_dataset, batch_size=32, shuffle=False)
    fp = open(save_path, 'w')
    fp.write('image_name,label\n')
    model = torch.load(model_path)
    if target_domain != 'svhn':
        model = DANN(hp.img_dim, hp.n_cls, 45).to(device)
    else:
        model = DANN_SVHN(hp.img_dim, hp.n_cls, 64).to(device)
    
    model.load_state_dict(torch.load(model_path))
    model.eval()
    same_seeds(0)
    alpha = 0 
    for i, t_img in enumerate(t_dataloader):
        print(f'{len(t_dataloader)}/{i+1}', end='\r')
        t_img = t_img.to(device)
        cls_logits, _ = model(t_img, alpha)
        preds = cls_logits.argmax(dim=1)
        for j, x in enumerate(preds):
            fp.write(f'{data_names[i*32+j]},{x.item()}\n')
            
if __name__ == '__main__':
    target_dir, target_domain, save_path, model_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    main(target_dir, target_domain, save_path, model_path)











