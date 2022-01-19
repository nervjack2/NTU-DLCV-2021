import torch
import os 
import sys
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torch import optim
from os import listdir
from os.path import join
from dataset import ImgDataset
from model import VGG16_FCN32, VGG16_FCN16
from utils import mean_iou_score
from hyper import *


phase = ['validation']

def main(
    data_dir: str,
    model_path: str,
    num_cls: int
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dirs = {p: join(data_dir,p) for p in phase}
    dataset = {p: ImgDataset(data_dirs[p], train=(False if p == 'validation' else True)) for p in phase}
    dataloader = {p: DataLoader(dataset[p], batch_size=6, shuffle=(True if p =='train' else False)) for p in phase}
    model = VGG16_FCN16(num_cls).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    optimizer = optim.Adam(model.parameters(), lr=lr) 
    best_loss, best_miou = float('inf'), 0
    p = 'validation'
    val_preds = torch.FloatTensor()
    val_labels = torch.LongTensor()
    for i, (inputs, labels) in enumerate(dataloader[p]):
        print(f'{len(dataloader[p])}/{i}', end='\r')
        inputs, labels = inputs.to(device), labels.to(device)
        logits = model(inputs)
        preds = logits.argmax(dim=1)
        val_preds = torch.cat((val_preds,preds.float().detach().cpu()),dim=0)
        val_labels = torch.cat((val_labels,labels.detach().cpu()),dim=0)
    val_miou = mean_iou_score(val_preds.numpy().astype(int), val_labels.numpy().astype(int))
    print(f'Vailidation MIOU: {val_miou}')

if __name__ == '__main__':
    data_dir, model_path, num_cls = sys.argv[1], sys.argv[2], int(sys.argv[3])
    main(data_dir, model_path, num_cls)