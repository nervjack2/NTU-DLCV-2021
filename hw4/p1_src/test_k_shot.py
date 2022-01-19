import os 
import sys
import torch
import numpy as np 
from os.path import join
from tqdm import tqdm 
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from hyper import hyper as hp
from model import ProtoNet, ParaDis
from dataset import ImgDataset, ImgSampler
from utils import define_label
from loss import cal_loss_and_acc

def main(
    data_dir: str,
    save_path: str,
    dis_method: str,
    k_shot: int,
    model_path: str
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hp.k_shot = k_shot
    hp.batch_size = hp.n_way * (hp.k_shot + hp.n_query)
    hp.n_eps_per_epoch = 600
    # Define the label of each classes  
    train_data, val_data = define_label(data_dir) 
    # Define Dataset, Sampler, Dataloader 
    val_dataset = ImgDataset(val_data)
    val_sampler =  ImgSampler(val_data[1], hp)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)
    # Define model, optimizer, lr scheduler
    if dis_method != 'para':
        model = ProtoNet().to(device)
        optimizer = Adam(model.parameters(), lr=hp.lr)
    else:
        model = ParaDis(emb_dim=hp.emb_dim).to(device)
        optimizer = Adam(model.parameters(), lr=hp.lr)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # Start to test
    val_acc = []
    for x, y in tqdm(val_loader, desc=f'Testing:'):
        x, y = x.to(device), y.squeeze().to(device)
        pred_emb = model(x)
        if dis_method != 'para':
            loss, acc = cal_loss_and_acc(pred_emb, y, hp, device, dis_method)
        else:
            loss, acc = cal_loss_and_acc(pred_emb, y, hp, device, dis_method, model.para)     
        val_acc.append(acc.item())

    episodic_acc = np.array(val_acc)
    mean = episodic_acc.mean()
    std = episodic_acc.std()

    print('Accuracy: {:.2f} +- {:.2f} %'.format(mean * 100, 1.96 * std / (600)**(1/2) * 100))

    with open(save_path, mode='w') as fp:
        fp.write('Accuracy: {:.2f} +- {:.2f} %'.format(mean * 100, 1.96 * std / (600)**(1/2) * 100))

if __name__ == '__main__':
    data_dir, save_path, dis_method, k_shot, model_path = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    main(data_dir, save_path, dis_method, int(k_shot), model_path)