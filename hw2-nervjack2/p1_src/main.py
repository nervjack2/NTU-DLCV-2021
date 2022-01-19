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
from model import Generator, Discriminator
from hyper import hyper as hp
from utils import same_seeds, gen_img

def main(
    data_dir: str,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(join(save_dir,'tmp'), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp = open(join(save_dir,'log.txt'), 'w')
    data_dirs = join(data_dir, 'train')
    data_names = listdir(data_dirs)
    data_paths = [join(data_dirs, name) for name in data_names]
    dataset = ImgDataset(data_paths)
    dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True)

    criterion = nn.BCELoss()
    model_G = Generator(hp.vec_dim, hp.latent_dim).to(device)
    model_D = Discriminator(hp.img_dim, hp.latent_dim).to(device)
    if hp.fine_tuned:
        model_G.load_state_dict(torch.load(join(hp.fine_tuned_path, 'best_generator.mdl')))
        model_D.load_state_dict(torch.load(join(hp.fine_tuned_path, 'best_discriminator.mdl')))

    opt_G = torch.optim.Adam(model_G.parameters(), lr=hp.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(model_D.parameters(), lr=hp.lr, betas=(0.5, 0.999))
    
    same_seeds(0)
    best_score = float('inf')

    for e in range(hp.n_epoch):
        model_G.train()
        model_D.train()
        e_loss_D, e_loss_G = 0,0
        for i, r_img in enumerate(dataloader):
            print(f'{len(dataloader)}/{i+1}', end='\r')
            """
            Train discriminator
            """
            bs = r_img.size(0)
            noise = torch.randn(bs, hp.vec_dim).to(device)
            
            r_img = r_img.to(device)
            f_img = model_G(noise)
            
            r_label = torch.ones((bs)).to(device)
            f_label = torch.zeros((bs)).to(device)

            r_logits = model_D(r_img)
            f_logtis = model_D(f_img)

            r_loss = criterion(r_logits, r_label)
            f_loss = criterion(f_logtis, f_label)
            loss_D = r_loss + f_loss
            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
            """
            Train Generator
            """
            noise = torch.randn(bs, hp.vec_dim).to(device)
            f_img = model_G(noise)
            f_label = torch.ones((bs)).to(device)
            f_logtis = model_D(f_img)
            loss_G = criterion(f_logtis, f_label)
            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            e_loss_D += loss_D.item()
            e_loss_G += loss_G.item()
        e_loss_D /= len(dataloader)
        e_loss_G /= len(dataloader)
        print(f'Epoch {e}: D_loss = {e_loss_D}, G_loss = {e_loss_G}')
        fp.write(f'Epoch {e}: D_loss = {e_loss_D}, G_loss = {e_loss_G}\n')
        if (e+1) % 10 == 0:
            torch.save(model_G.state_dict(), join(save_dir, f'checkpoint_{e+1}.mdl'))
        model_G.eval()
        model_D.eval()
        gen_img(model_G, join(save_dir,'tmp'), hp, device)
        os.system(f'python -m pytorch_fid ../hw2_data/face/test {join(save_dir, "tmp")} > {join(save_dir, "score.txt")}')
        with open(join(save_dir, "score.txt"), 'r') as fp_tmp:
            score = float(fp_tmp.readline().split(':')[1].strip())
        if score < best_score:
            best_score = score
            torch.save(model_G.state_dict(), join(save_dir, 'best_generator.mdl'))
            torch.save(model_D.state_dict(), join(save_dir, 'best_discriminator.mdl'))
            print(f'Epoch {e}: New best checkpoint with fid={best_score}')
        fp.flush()

if __name__ == '__main__':
    data_dir, save_dir = sys.argv[1], sys.argv[2]
    main(data_dir, save_dir)












