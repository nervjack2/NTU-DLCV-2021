import torch
import os 
import sys
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image
from torch import optim
from os import listdir
from os.path import join
from dataset import ImgDataset
from model import VGG16_FCN32, VGG16_FCN16
from utils import label_to_seg_img
from hyper import *


def main(
    data_dir: str,
    model_path: str,
    num_cls: int,
    out_dir: str 
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_name = sorted([x.split('.')[0] for x in os.listdir(data_dir) if x.endswith('jpg')])
    dataset = ImgDataset(data_dir, train=False, inference=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    model = VGG16_FCN16(num_cls).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    val_preds = torch.FloatTensor()
    for i, inputs in enumerate(dataloader):
        print(f'{len(dataloader)}/{i}', end='\r')
        inputs = inputs.to(device)
        logits = model(inputs)
        preds = logits.argmax(dim=1).squeeze().detach().cpu().numpy()
        seg_map = label_to_seg_img(preds)
        im = Image.fromarray(seg_map)
        im.save(join(out_dir, data_name[i]+'.png'))


if __name__ == '__main__':
    data_dir, model_path, num_cls, out_dir = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    main(data_dir, model_path, num_cls, out_dir)