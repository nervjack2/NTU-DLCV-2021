import torch
import sys
import os 
import torch.nn as nn 
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader
from os.path import join
from byol_pytorch import BYOL
from torchvision import models
from torch.optim import Adam
from hyper import hyper as hp
from dataset import FineTunedImgDataset
from model import ImgClassifier
from utils import define_label

def main(
    data_dir: str, 
    model_path: str,
    save_dir: str,
    fix_backbone: int
):
    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Define fine-tuned model 
    model = ImgClassifier(model_path, fix_backbone, hp.fine_tuned.n_cls, hp.fine_tuned.in_dim).to(device)
    # Load data 
    train_data, val_data = define_label(data_dir)
    # Define dataset and dataloader 
    train_set = FineTunedImgDataset(train_data, hp.fine_tuned.img_size)
    val_set = FineTunedImgDataset(val_data, hp.fine_tuned.img_size)
    train_loader = DataLoader(train_set, batch_size=hp.fine_tuned.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=hp.fine_tuned.batch_size)
    opt = Adam(model.parameters(), lr=hp.fine_tuned.lr)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0
    for e in range(hp.fine_tuned.n_epoch):
        model.classifier.train()
        if not fix_backbone:
            model.feat_extractor.train()
        else:
            model.feat_extractor.eval()
        train_acc, val_acc = 0,0
        train_loss, val_loss = 0,0
        for images, labels in tqdm(train_loader, desc=f'Training epoch {e}:'):
            images, labels = images.to(device), labels.squeeze().to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            train_acc += (pred == labels).sum()/len(pred)
        model.eval()
        for images, labels in tqdm(val_loader, desc=f'Validation epoch {e}:'):
            images, labels = images.to(device), labels.squeeze().to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)
            val_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            val_acc += (pred == labels).sum()/len(pred)
        train_loss, val_loss = train_loss/len(train_loader), val_loss/len(val_loader)
        train_acc, val_acc = train_acc/len(train_loader), val_acc/len(val_loader)
        print(f'Epoch {e}: training acc={train_acc}, val acc={val_acc}')
        print(f'Epoch {e}: training loss={train_loss}, val loss={val_loss}')
        if best_acc < val_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), join(save_dir, 'best_model.mdl'))
        if (e+1)%10 == 0:
            torch.save(model.state_dict(), join(save_dir, f'checkpoint{e+1}.mdl'))

if __name__ == '__main__':
    data_dir, model_path, save_dir, fix_backbone = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    main(data_dir, model_path, save_dir, int(fix_backbone))