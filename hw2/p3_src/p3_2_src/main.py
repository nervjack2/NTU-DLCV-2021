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
from dataset import ImgDataset, SVHN_ImgDataset, SVHN_EvalImgDataset
from model import DANN, DANN_SVHN
from hyper import hyper as hp
from utils import same_seeds, get_data

def main(
    source_dir: str,
    target_dir: str,
    save_dir: str,
    target_domain: str
):
    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp = open(join(save_dir,'log.txt'), 'w')
    s_data_paths, s_data_labels = get_data(source_dir)
    t_data_paths, _ = get_data(target_dir)
    e_data_paths, e_data_labels = get_data(target_dir, phase='test')
    if target_domain != 'svhn':
        s_dataset = ImgDataset(s_data_paths, s_data_labels)
        t_dataset = ImgDataset(t_data_paths)
        e_dataset = ImgDataset(e_data_paths, e_data_labels)
    else:
        s_dataset = SVHN_ImgDataset(s_data_paths, s_data_labels)
        t_dataset = SVHN_ImgDataset(t_data_paths)
        e_dataset = SVHN_EvalImgDataset(e_data_paths, e_data_labels)
    s_dataloader = DataLoader(s_dataset, batch_size=hp.batch_size, shuffle=True)
    t_dataloader = DataLoader(t_dataset, batch_size=hp.batch_size, shuffle=True)
    e_dataloader = DataLoader(e_dataset, batch_size=hp.batch_size, shuffle=False)

    criterion = nn.NLLLoss()
    cls_criterion = nn.NLLLoss()
    
    # Should change
    if target_domain != 'svhn':
        model = DANN(hp.img_dim, hp.n_cls, hp.latent_dim).to(device)
    else:
        model = DANN_SVHN(hp.img_dim, hp.n_cls, hp.latent_dim).to(device)
    if hp.fine_tuned:
        model.load_state_dict(torch.load(join(hp.fine_tuned_path,'best_cls.mdl')))
    
    for params in model.parameters():
        params.requires_grad = True

    opt = torch.optim.Adam(model.parameters(), lr=hp.lr)

    best_acc = 0

    for e in range(hp.n_epoch):
        model.train()
        e_cls_loss, e_s_acc = 0,0
        e_domain_loss = 0
        data_len = min(len(s_dataloader), len(t_dataloader))
        data_source_iter = iter(s_dataloader)
        data_target_iter = iter(t_dataloader)

        idx = 0 

        while idx < data_len:
            print(f'{data_len}/{idx+1}', end='\r')
            # Set up alpha
            a = float(idx + e * data_len) 
            a = a / hp.n_epoch
            a = a / data_len
            alpha = 2. / (1. + np.exp(-10 * a)) - 1

            s_img, cls_label = data_source_iter.next()
            t_img = data_target_iter.next() 

            s_bs = s_img.size(0)
            t_bs = t_img.size(0)
            s_img = s_img.to(device)
            t_img = t_img.to(device)
            cls_label = cls_label.squeeze().to(device)
            s_domain_label = torch.full((s_bs,), 1).to(device)
            cls_logits, s_domain_logits = model(s_img, alpha)
            preds = cls_logits.argmax(dim=1)
            cls_loss = cls_criterion(cls_logits, cls_label)
            s_domain_loss = criterion(s_domain_logits, s_domain_label)
            e_s_acc += (preds == cls_label).sum()/s_bs
            t_domain_label = torch.full((t_bs,), 0).to(device)
            _, t_domain_logits = model(t_img, alpha)
            t_domain_loss = criterion(t_domain_logits, t_domain_label)
            loss = cls_loss + s_domain_loss + t_domain_loss
            opt.zero_grad()
            loss.backward()
            opt.step()
            e_cls_loss += cls_loss.item()
            e_domain_loss += (s_domain_loss.item()+t_domain_loss.item())/2
            idx += 1 

        e_cls_loss /= data_len
        e_domain_loss /= data_len
        e_s_acc /= data_len
        print(f'Epoch {e}: training loss = {e_cls_loss}/{e_domain_loss}, acc = {e_s_acc}')
        fp.write(f'Epoch {e}: training loss = {e_cls_loss}/{e_domain_loss}, acc = {e_s_acc}\n')
        fp.flush()
        model.eval()
        e_acc = 0
        for i, (r_img, cls_label) in enumerate(e_dataloader):
            print(f'{len(e_dataloader)}/{i+1}', end='\r')
            r_img = r_img.to(device)
            cls_label = cls_label.squeeze().to(device)
            cls_logits, _ = model(r_img, 0)
            preds = cls_logits.argmax(dim=1)
            e_acc += (preds == cls_label).sum()/len(r_img)
        e_acc /= len(e_dataloader)
        print(f'Epoch {e}: test acc = {e_acc}')
        fp.write(f'Epoch {e}: test acc = {e_acc}\n')
        fp.flush()
        if e_acc > best_acc:
            best_acc = e_acc 
            torch.save(model.state_dict(), join(save_dir, 'best_classifier.mdl'))
    
if __name__ == '__main__':
    source_dir, target_dir, save_dir, target_domain = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    main(source_dir, target_dir, save_dir, target_domain)












