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
from model import VGG16_FCN16
from utils import mean_iou_score
from hyper import *

phase = ['train','validation']

def main(
    data_dir: str,
    save_dir: str,
    num_cls: int
):
    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp = open(join(save_dir,'log.txt'), 'w')
    data_dirs = {p: join(data_dir,p) for p in phase}
    dataset = {p: ImgDataset(data_dirs[p], train=(False if p == 'validation' else True)) for p in phase}
    dataloader = {p: DataLoader(dataset[p], batch_size=6, shuffle=(True if p =='train' else False)) for p in phase}
    model = VGG16_FCN16(num_cls).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_func = F.cross_entropy 
    best_loss, best_miou = float('inf'), 0
    for e in range(1,epoch+1):
        for p in phase:
            train_loss, val_loss = 0,0
            val_preds = torch.FloatTensor()
            val_labels = torch.LongTensor()
            model.train() if p == 'train' else model.eval()
            for i, (inputs, labels) in enumerate(dataloader[p]):
                print(f'{p} epoch {e}: {len(dataloader[p])}/{i}', end='\r')
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                preds = logits.argmax(dim=1)
                loss = loss_func(logits, labels)
                if p == 'validation':
                    val_loss += loss.item()
                    val_preds = torch.cat((val_preds,preds.detach().cpu().float()),dim=0)
                    val_labels = torch.cat((val_labels,labels.detach().cpu()),dim=0)
                    continue
                train_loss += loss.item()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if p == 'validation':
                val_miou = mean_iou_score(val_preds.numpy().astype(int), val_labels.numpy().astype(int))
                val_loss /= len(dataloader['validation'])
                if val_miou > best_miou:
                    best_miou = val_miou
                    best_loss = val_loss
                    torch.save(model.state_dict(), join(save_dir,'best_checkpoint.mdl'))
                if e % 4 == 1:
                    torch.save(model.state_dict(), join(save_dir,f'checkpoint{e}.mdl'))
                print(f'Epoch {e}: val loss {val_loss}, val miou {val_miou}')
                fp.write(f'Epoch {e}: val loss {val_loss}, val miou {val_miou}\n')
                fp.flush()
                continue
            train_loss /= len(dataloader['train'])
            print(f'Epoch {e}: train loss {train_loss}')
            fp.write(f'Epoch {e}: train loss {train_loss}\n')
            fp.flush()
    print(f'Best loss: {best_loss}, Best miou: {best_miou}')
    fp.write(f'Best loss: {best_loss}\n')
    fp.write(f'Best miou: {best_miou}\n')
    fp.flush()

if __name__ == '__main__':
    data_dir, save_dir, num_cls = sys.argv[1], sys.argv[2], int(sys.argv[3])
    main(data_dir, save_dir, num_cls)