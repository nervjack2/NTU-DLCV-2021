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
from utils import same_seeds, get_data, gen_img
from digit_classifier import log_acc

def main(
    data_dir: str,
    save_dir: str,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(join(save_dir,'tmp'), exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    fp = open(join(save_dir,'log.txt'), 'w')
    data_paths, data_labels = get_data(data_dir)
    print(len(data_paths), len(data_labels))
    dataset = ImgDataset(data_paths, data_labels)
    dataloader = DataLoader(dataset, batch_size=hp.batch_size, shuffle=True)

    criterion = nn.BCELoss()
    cls_criterion = nn.NLLLoss()
    
    model_G = Generator(hp.vec_dim, hp.n_cls, hp.latent_dim).to(device)
    model_D = Discriminator(hp.img_dim, hp.n_cls, hp.latent_dim).to(device)
    if hp.fine_tuned:
        model_G.load_state_dict(torch.load(join(hp.fine_tuned_path,'best_generator.mdl')))
        model_D.load_state_dict(torch.load(join(hp.fine_tuned_path,'best_discriminator.mdl')))
  
    opt_G = torch.optim.Adam(model_G.parameters(), lr=hp.lr, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(model_D.parameters(), lr=hp.lr, betas=(0.5, 0.999))

    # label smoothening
    real_labels = 0.7 + 0.5 * torch.rand(10).to(device)  
    fake_labels = 0.3 * torch.rand(10).to(device)  
    same_seeds(0)
    best_acc = 0

    for e in range(hp.n_epoch):
        model_G.train()
        model_D.train()
        e_loss_D, e_loss_G = 0,0
        for i, (r_img, cls_label) in enumerate(dataloader):
            print(f'{len(dataloader)}/{i+1}', end='\r')
            bs = r_img.size(0)
            real_label = real_labels[i % 10]
            fake_label = fake_labels[i % 10]
             # label flipping
            if i % 7 == 0:
                real_label, fake_label = fake_label, real_label

            r_label = torch.full((bs,), real_label).to(device)
            f_label = torch.full((bs,), fake_label).to(device)
            hard_r_label = torch.full((bs,), 1.0).to(device)

            r_img = r_img.to(device)
            cls_label = cls_label.squeeze().to(device)
            """
            Train Discriminator
            """
            noise = torch.randn(bs, hp.vec_dim).to(device)
            sample_labels = torch.randint(0, hp.n_cls, (bs,), dtype=torch.long).to(device)
            
            f_img = model_G(noise, sample_labels)
            
            r_logits, r_cls_logits = model_D(r_img)
            f_logtis, f_cls_logits = model_D(f_img)

            r_loss = criterion(r_logits, r_label)
            f_loss = criterion(f_logtis, f_label)
            r_cls_loss = cls_criterion(r_cls_logits, cls_label)
            f_cls_loss = cls_criterion(f_cls_logits, sample_labels)
            loss_D = r_loss + f_loss + r_cls_loss + f_cls_loss

            opt_D.zero_grad()
            loss_D.backward()
            opt_D.step()
            """
            Train Generator
            """
            noise = torch.randn(bs, hp.vec_dim).to(device)
            sample_labels = torch.randint(0, hp.n_cls, (bs,), dtype=torch.long).to(device)

            f_img = model_G(noise, sample_labels)
            f_logtis, f_cls_logits = model_D(f_img)
            
            adv_loss = criterion(f_logtis, hard_r_label)
            g_cls_loss = cls_criterion(f_cls_logits, sample_labels)
            loss_G = adv_loss + g_cls_loss

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()
            e_loss_D += loss_D.item()
            e_loss_G += loss_G.item()
        e_loss_D /= len(dataloader)
        e_loss_G /= len(dataloader)
        print(f'Epoch {e}: D_loss = {e_loss_D}, G_loss = {e_loss_G}')
        fp.write(f'Epoch {e}: D_loss = {e_loss_D}, G_loss = {e_loss_G}\n')
        fp.flush()
        model_G.eval()
        model_D.eval()
        gen_img(join(save_dir,'tmp'), hp.vec_dim, model_G, device)
        epoch_acc = log_acc(join(save_dir,'tmp'), e)
        if (e+1) % 10 == 0:
            torch.save(model_G.state_dict(), join(save_dir, f'checkpoint_{e+1}.mdl'))
        if epoch_acc > best_acc:
            print(f'Epoch {e}: New best classify accuracy with {epoch_acc.item()}')
            fp.write(f'Epoch {e}: New best classify accuracy with {epoch_acc.item()}\n')
            fp.flush()
            best_acc = epoch_acc
            torch.save(model_G.state_dict(), join(save_dir, 'best_generator.mdl'))
            torch.save(model_D.state_dict(), join(save_dir, 'best_discriminator.mdl'))

if __name__ == '__main__':
    data_dir, save_dir = sys.argv[1], sys.argv[2]
    main(data_dir, save_dir)












