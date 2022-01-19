import torch
import os 
import sys
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
from PIL import Image
from os import listdir
from os.path import join
from dataset import ImgDataset
from model import VGG16_FCN32, VGG16_FCN16
from utils import mean_iou_score, label_to_seg_img
from hyper import *


phase = ['validation']

def main(
    data_dir: str,
    model_dir: str,
    num_cls: int
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dirs = {p: join(data_dir,p) for p in phase}
    dataset = {p: ImgDataset(data_dirs[p], train=(False if p == 'validation' else True)) for p in phase}
    datas = [dataset['validation'][i] for i in [10,97,107]]
    model_index = [1,25,45]

    for i, data_index in enumerate(['0010','0097','0107']):
        img_path = join(data_dirs['validation'], f'{data_index}_sat.jpg')
        img = Image.open(img_path)
        plt.subplot(3,4,4*i+1)
        plt.imshow(img)
        if i == 0:
            plt.title('Original')
        plt.ylabel(data_index,rotation=0,labelpad=22.0)
        plt.xticks(ticks=[])
        plt.yticks(ticks=[])
    for i, (inputs, labels) in enumerate(datas):
        inputs, labels = inputs.unsqueeze(0).to(device), labels.unsqueeze(0).to(device)
        for j, epoch in enumerate(model_index,1):
            model = VGG16_FCN16(num_cls).to(device)
            model.load_state_dict(torch.load(join(model_dir,f'checkpoint{epoch}.mdl')))
            model.eval()
            logits = model(inputs)
            preds = logits.argmax(dim=1).squeeze().detach().cpu().numpy()
            seg_map = label_to_seg_img(preds)
            plt.subplot(3,4,4*i+j+1)
            plt.imshow(seg_map)
            plt.xticks(ticks=[])
            plt.yticks(ticks=[])
            if i == 0:
                if j == 1:
                    plt.title('Early')
                elif j == 2:
                    plt.title('Middle')
                elif j == 3:
                    plt.title('Final')
    plt.show()


if __name__ == '__main__':
    data_dir, model_dir, num_cls = sys.argv[1], sys.argv[2], int(sys.argv[3])
    main(data_dir, model_dir, num_cls)