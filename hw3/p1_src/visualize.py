import torch
import os 
import sys
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torch import optim
from seaborn import heatmap
from os import listdir
from os.path import join
from dataset import ImgDataset
from model import ViT_Classifier
from hyper import hyper as hp

phase = ['train','val']

transform = transforms.Compose([
                            transforms.Resize((384,384)),
                            transforms.ToTensor(),
                            transforms.Normalize(0.5, 0.5),
                        ])
transform_eval = transforms.Compose([
                            transforms.Resize((384,384)),
                            transforms.ToTensor(),
                        ])


def main(
    data_dir: str,
    save_dir: str,
    num_cls: int,
    model_path: str,
    visual_type: str,
):
    os.makedirs(save_dir, exist_ok=True)
    #device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu'
    model = ViT_Classifier('B_16_imagenet1k', num_cls=num_cls, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    if visual_type == 'pos':
        # Visualize positional embedding 
        pos_emb = model.model.positional_embedding.pos_embedding.data.squeeze().detach().cpu().numpy()[1:,:]
        pos_emb_corr = np.matmul(pos_emb, pos_emb.T)
        sub_corr_img = []
        for x in pos_emb_corr:
            sub_corr_img.append(x.reshape(24,24))
        fig = plt.figure(figsize=(8, 8))
        columns = 24
        rows = 24
        for i, img in enumerate(sub_corr_img):
            print(f'{len(sub_corr_img)}/{i+1}', end='\r')
            fig.add_subplot(rows, columns, i+1)
            heatmap(img, cbar=False)
            plt.axis('off')
        plt.show()
    elif visual_type == 'attn':
        # Visualize attention map 
        att_cache = []
        img_cache = []
        img_path_list = ['val/26_5064.jpg', 'val/29_4718.jpg', 'val/31_4838.jpg']
        for img_path in img_path_list:
            img_path = join(data_dir, img_path)
            img = Image.open(img_path)
            img_eval = transform_eval(img)
            img_cache.append(img_eval.squeeze().detach().numpy())
            img = transform(img).float().unsqueeze(0).to(device)
            logits = model(img)
            att_map = model.model.transformer.blocks[-1].attn.scores
            x = att_map.squeeze().mean(0)[0].squeeze()[1:]
            att_cache.append(x.detach().cpu().numpy())
        fig = plt.figure(figsize=(20, 20))
        columns = 2
        rows = 2
        for i, (att, img) in enumerate(zip(att_cache, img_cache)):
            att = att.reshape(24,24)
            im = Image.fromarray(att)
            att = np.array(im.resize((384,384), resample=Image.BILINEAR))
            img = img.transpose(1,2,0)
            fig.add_subplot(rows, columns, i+1)
            plt.imshow(img)
            plt.imshow(att, cmap='jet', alpha=0.3)
            plt.axis('off')
        plt.savefig(join(save_dir,'result.png'), bbox_inches='tight')

if __name__ == '__main__':
    data_dir, save_dir, num_cls, model_path, visual_type = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
    main(data_dir, save_dir, num_cls, model_path, visual_type)