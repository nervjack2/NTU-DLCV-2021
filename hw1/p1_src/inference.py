import torch
import os 
import sys
import torch.nn as nn
from os import listdir
from os.path import join
from torch.utils.data import DataLoader
from dataset import ImgDataset
from hyper import *
from model import ImgClassifier

def main(
    data_dir: str,
    model_path: str,
    num_cls: int,
    csv_path: str
):
    fp = open(csv_path, 'w')
    fp.write('image_id,label\n')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_names = sorted(listdir(data_dir))
    data_paths = [join(data_dir, name) for name in data_names]
    dataset = ImgDataset(data_paths, None, train=False)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) 
    model = ImgClassifier(num_cls).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
  
    for i, (inputs,name) in enumerate(zip(dataloader, data_names)):
        inputs = inputs.to(device)
        logits = model(inputs)
        preds = logits.argmax(dim=1)
        fp.write(f'{name},{preds.squeeze().detach().cpu().numpy()}\n')


if __name__ == '__main__':
    data_dir, model_path, num_cls, csv_path = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    main(data_dir, model_path, num_cls, csv_path)