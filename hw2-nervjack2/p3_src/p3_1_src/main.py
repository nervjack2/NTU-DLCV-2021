import torch
import torch.nn as nn
import os 
import sys
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from os import listdir
from os.path import join
from dataset import ImgDataset
from model import Discriminator
from hyper import hyper as hp
from utils import same_seeds, get_data

def main(
    source_dir: str,
    target_dir: str,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    fp = open(join(save_dir,'log.txt'), 'w')
    data_paths, data_labels = get_data(source_dir)
    print(len(data_paths), len(data_labels))
    dataset = ImgDataset(data_paths, data_labels)
    dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True)
    e_data_paths, e_data_labels = get_data(target_dir, phase='test')
    print(len(e_data_paths), len(e_data_labels))
    e_dataset = ImgDataset(e_data_paths, e_data_labels)
    e_dataloader = DataLoader(e_dataset, batch_size=hp.batch_size, shuffle=False)

    criterion = nn.NLLLoss()
    
    model_cls = Discriminator(hp.img_dim, hp.n_cls, hp.latent_dim).to(device)
    if hp.fine_tuned:
        model_cls.load_state_dict(torch.load(join(hp.fine_tuned_path,'best_cls.mdl')))
  
    opt = torch.optim.Adam(model_cls.parameters(), lr=hp.lr)

    same_seeds(0)
    best_acc = 0

    for e in range(hp.n_epoch):
        model_cls.train()
        e_loss, e_acc = 0,0
        for i, (r_img, cls_label) in enumerate(dataloader):
            print(f'{len(dataloader)}/{i+1}', end='\r')
            r_img = r_img.to(device)
            cls_label = cls_label.squeeze().to(device)
            logits = model_cls(r_img)
            loss = criterion(logits, cls_label)
            preds = logits.argmax(dim=1)
            e_acc += (preds == cls_label).sum()/len(r_img)
            opt.zero_grad()
            loss.backward()
            opt.step()
            e_loss += loss.item()
            
        e_loss /= len(dataloader)
        e_acc /= len(dataloader)
        print(f'Epoch {e}: training loss = {e_loss}, acc = {e_acc}')
        fp.write(f'Epoch {e}: training loss = {e_loss}, acc = {e_acc}\n')
        fp.flush()
        model_cls.eval()
        e_acc = 0
        for i, (r_img, cls_label) in enumerate(e_dataloader):
            print(f'{len(e_dataloader)}/{i+1}', end='\r')
            r_img = r_img.to(device)
            cls_label = cls_label.squeeze().to(device)
            logits = model_cls(r_img)
            preds = logits.argmax(dim=1)
            e_acc += (preds == cls_label).sum()/len(r_img)
        e_acc /= len(e_dataloader)
        print(f'Epoch {e}: test acc = {e_acc}')
        fp.write(f'Epoch {e}: test acc = {e_acc}\n')
        fp.flush()
    
if __name__ == '__main__':
    source_dir, target_dir, save_dir = sys.argv[1], sys.argv[2], sys.argv[3]
    main(source_dir, target_dir, save_dir)












