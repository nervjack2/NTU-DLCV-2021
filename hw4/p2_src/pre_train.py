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
    save_dir: str 
):
    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    resnet = models.resnet50(pretrained=False)

    learner = BYOL(
        resnet,
        image_size = hp.pretraining.img_size,
        hidden_layer = 'avgpool'
    ).to(device)
    # Define dataloader 
    train_set = ImgDataset(join(data_dir,'train'), hp.pretraining.img_size)
    val_set = ImgDataset(join(data_dir,'val'), hp.pretraining.img_size)
    train_loader = DataLoader(train_set, batch_size=hp.pretraining.batch_size)
    val_loader = DataLoader(val_set, batch_size=hp.pretraining.batch_size)
    opt = Adam(learner.parameters(), lr=hp.pretraining.lr)
    best_loss = float('inf')
    for e in range(hp.pretraining.n_epoch):
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
        learner.eval()
        for images in tqdm(val_loader, desc=f'Validation epoch {e}:'):
            images = images.to(device)
            loss = learner(images)
            val_loss += loss.item()
        train_loss, val_loss = train_loss/len(train_loader), val_loss/len(val_loader)
        print(f'Epoch {e}: training loss={train_loss}, val loss={val_loss}')
        if best_loss > train_loss:
            best_loss = train_loss
            torch.save(resnet.state_dict(), join(save_dir, 'best_model.mdl'))
        if (e+1)%20 == 0:
            torch.save(resnet.state_dict(), join(save_dir, f'checkpoint{e+1}.mdl'))

if __name__ == '__main__':
    data_dir, save_dir = sys.argv[1], sys.argv[2]
    main(data_dir, save_dir)