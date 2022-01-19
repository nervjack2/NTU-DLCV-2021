import os 
import sys
import torch
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
    k_shot: int
):
    os.makedirs(save_path, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hp.k_shot = k_shot
    hp.batch_size = hp.n_way * (hp.k_shot + hp.n_query)
    # Define the label of each classes  
    train_data, val_data = define_label(data_dir) 
    # Define Dataset, Sampler, Dataloader 
    train_dataset = ImgDataset(train_data)
    val_dataset = ImgDataset(val_data)
    train_sampler = ImgSampler(train_data[1], hp)
    val_sampler =  ImgSampler(val_data[1], hp)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler)
    # Define model, optimizer, lr scheduler
    if dis_method != 'para':
        model = ProtoNet().to(device)
        optimizer = Adam(model.parameters(), lr=hp.lr)
    else:
        model = ParaDis(emb_dim=hp.emb_dim).to(device)
        optimizer = Adam(model.parameters(), lr=hp.lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                        gamma=hp.lr_scheduler_gamma, step_size=hp.lr_scheduler_step)
    # Start to train 
    best_acc = 0
    for e in range(hp.n_epoch):
        train_loss, val_loss = 0,0
        train_acc, val_acc = 0,0
        model.train()
        for x, y in tqdm(train_loader, desc=f'Training epoch {e}:'):
            x, y = x.to(device), y.squeeze().to(device)
            pred_emb = model(x)
            if dis_method != 'para':
                loss, acc = cal_loss_and_acc(pred_emb, y, hp, device, dis_method)
            else:
                loss, acc = cal_loss_and_acc(pred_emb, y, hp, device, dis_method, model.para)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() / len(x)           
            train_acc += acc
        lr_scheduler.step()
        model.eval()
        for x, y in tqdm(val_loader, desc=f'Validation epoch {e}:'):
            x, y = x.to(device), y.squeeze().to(device)
            pred_emb = model(x)
            if dis_method != 'para':
                loss, acc = cal_loss_and_acc(pred_emb, y, hp, device, dis_method)
            else:
                loss, acc = cal_loss_and_acc(pred_emb, y, hp, device, dis_method, model.para)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            val_loss += loss.item() / len(x)           
            val_acc += acc
        train_acc, val_acc = train_acc / len(train_loader), val_acc / len(val_loader)
        train_loss, val_loss = train_loss / len(train_loader), val_loss / len(val_loader)
        print(f'Epoch {e}: train acc={train_acc}, train loss={train_loss}')
        print(f'Epoch {e}: val acc={val_acc}, val loss={val_loss}')
        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), join(save_path, 'best_model.mdl'))
        if (e+1) % 10 == 0:
            torch.save(model.state_dict(), join(save_path, f'checkpoint_{e+1}.mdl'))

if __name__ == '__main__':
    data_dir, save_path, dis_method, k_shot = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    main(data_dir, save_path, dis_method, int(k_shot))