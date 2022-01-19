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
from dataset import EvalImgDataset
from model import EvalImgClassifier
from utils import define_label, load_eval_data

def main(
    data_dir: str, 
    data_csv: str, 
    model_path: str,
    output_csv: str,
    label2class: str
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load label to class mapping
    with open(label2class, 'r') as fp:
        l2c = fp.readline().strip().split(' ')
    # Define fine-tuned model 
    model = EvalImgClassifier(hp.fine_tuned.n_cls, hp.fine_tuned.in_dim).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # Load data 
    val_path, val_name = load_eval_data(data_dir, data_csv)
    # Define dataset and dataloader 
    val_set = EvalImgDataset(val_path, hp.fine_tuned.img_size)
    val_loader = DataLoader(val_set, batch_size=hp.fine_tuned.batch_size)

    output_label = []
    for images in tqdm(val_loader, desc=f'Testing:'):
        images = images.to(device)
        logits = model(images)
        pred = torch.argmax(logits, dim=1)
        for x in pred:
            output_label.append(x.item())

    with open(output_csv, 'w') as fp:
        fp.write('id,filename,label\n')
        for i, x in enumerate(output_label):
            fp.write(f'{i},{val_name[i]},{l2c[x]}\n')

if __name__ == '__main__':
    data_dir, data_csv, model_path, output_csv, label2class = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    main(data_dir, data_csv, model_path, output_csv, label2class)