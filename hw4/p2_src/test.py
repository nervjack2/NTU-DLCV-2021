import torch
import sys
import os 
from tqdm import tqdm
from torch.utils.data import DataLoader
from os.path import join
from byol_pytorch import BYOL
from torchvision import models
from torch.optim import Adam
from hyper import hyper as hp
from dataset import ImgDataset


def main(
    data_dir: str, 
    model_path: str,
    save_path: str 
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    feat_extractor
    model = ImgClassifier()
    # Define dataloader 
    train_set = ImgDataset(join(data_dir,'train'), hp.img_size)
    val_set = ImgDataset(join(data_dir,'val'), hp.img_size)
    train_loader = DataLoader(train_set, batch_size=hp.batch_size)
    val_loader = DataLoader(val_set, batch_size=hp.batch_size)
    opt = Adam(learner.parameters(), lr=hp.lr)
    best_loss = float('inf')
    for e in range(hp.n_epoch):
        learner.train()
        train_loss, val_loss = 0,0
        for images in tqdm(train_loader, desc=f'Training epoch {e}:'):
            images = images.to(device)
            loss = learner(images)
            opt.zero_grad()
            loss.backward()
            opt.step()
            learner.update_moving_average() # update moving average of target encoder
            train_loss += loss.item()
        for images in tqdm(val_loader, desc=f'Validation epoch {e}:'):
            loss = learner(images)
            val_loss += loss.item()
        train_loss, val_loss = train_loss/len(train_loader), val_loss/len(val_loader)
        print(f'Epoch {e}: training loss={train_loss}, val loss={val_loss}')
        if best_loss > val_loss:
            best_loss = val_loss
            torch.save(resnet.state_dict(), join(save_dir, 'best_model.mdl'))
        if (e+1)%10 == 0:
            torch.save(resnet.state_dict(), join(save_dir, f'checkpoint{e+1}.mdl'))

if __name__ == '__main__':
    data_dir, model_path, save_path = sys.argv[1], sys.argv[2], sys.argv[3]
    main(data_dir, model_path, save_path)