import torch
import os 
import sys
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from os import listdir
from os.path import join
from dataset import ImgDataset
from model import ViT_Classifier
from hyper import hyper as hp

phase = ['train','val']

def main(
    data_dir: str,
    save_dir: str,
    num_cls: int
):
    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp = open(join(save_dir,'log.txt'), 'w')
    data_dirs = {p: join(data_dir, p) for p in phase}
    data_names = {p: listdir(data_dirs[p]) for p in phase}
    data_paths = {p: [join(data_dirs[p], name) for name in data_names[p]] for p in phase}
    data_labels = {p: [int(name.split('_')[0]) for name in data_names[p]] for p in phase}
    dataset = {p: ImgDataset(data_paths[p], data_labels[p], train=(False if p == 'val' else True)) for p in phase}
    dataloader = {p: DataLoader(dataset[p], batch_size=hp.bs, shuffle=(True if p =='train' else False)) for p in phase}
    model = ViT_Classifier('B_16_imagenet1k', num_cls=num_cls, pretrained=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=hp.lr)
    loss_func = F.cross_entropy 
    best_loss, best_acc = float('inf'), 0
    for e in range(1,hp.epoch+1):
        print(e%3)
        for p in phase:
            val_correct, val_loss = 0,0
            train_correct, train_loss = 0,0
            model.train() if p == 'train' else model.eval()
            for i, (inputs, labels) in enumerate(dataloader[p]):
                print(f'{p} epoch {e}: {len(dataloader[p])}/{i}', end='\r')
                inputs, labels = inputs.to(device), labels.squeeze().to(device)
                logits = model(inputs)
                preds = logits.argmax(dim=1)
                loss = loss_func(logits, labels)
                if p == 'val':
                    val_loss += loss.item()
                    val_correct += (preds==labels).sum().float()/len(inputs)
                    continue
                train_loss += loss.item()
                train_correct += (preds==labels).sum().float()/len(inputs)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if p == 'val':
                val_correct /= len(dataloader[p])
                val_loss /= len(dataloader[p])
                if val_correct > best_acc:
                    best_loss = val_loss
                    best_acc = val_correct
                    torch.save(model.state_dict(), join(save_dir,f'best_model.mdl'))
                if (e % 3) == 0:
                    torch.save(model.state_dict(), join(save_dir,f'checkpoint_{e}.mdl'))
                print(f'Epoch {e}: val loss {val_loss}, val acc {val_correct}')
                fp.write(f'Epoch {e}: val loss {val_loss}, val acc {val_correct}\n')
                fp.flush()
                continue
            train_correct /= len(dataloader[p])
            train_loss /= len(dataloader[p])
            print(f'Epoch {e}: train loss {train_loss}, train acc {train_correct}')
            fp.write(f'Epoch {e}: train loss {train_loss}, train acc {train_correct}\n')
            fp.flush()
    print(f'Best loss: {best_loss}, Best acc: {best_acc}')
    fp.write(f'Best loss: {best_loss}\n')
    fp.write(f'Best acc: {best_acc}\n')
    fp.flush()

if __name__ == '__main__':
    data_dir, save_dir, num_cls = sys.argv[1], sys.argv[2], int(sys.argv[3])
    main(data_dir, save_dir, num_cls)