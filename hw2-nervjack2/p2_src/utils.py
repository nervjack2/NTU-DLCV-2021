import random
import torch
import os 
import numpy as np
from PIL import Image
from os.path import join

def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_data(data_dir, phase='train'):
    data_paths, labels = [], []
    data_dir_path = os.path.join(data_dir, phase)
    label_csv = os.path.join(data_dir, f'{phase}.csv')
    with open(label_csv, 'r') as fp:
        fp.readline()
        label_dict = {x.strip().split(',')[0]: int(x.strip().split(',')[1]) for x in fp}
    data_names = os.listdir(data_dir_path)
    for x in data_names:
        data_paths.append(os.path.join(data_dir_path,x))
        labels.append(label_dict[x])
    return data_paths, labels

def gen_img(save_dir, vec_dim, model_G, device):
    for i in range(10):
        for j in range(100):
            noise = torch.randn(1, vec_dim).to(device)
            cls_label = torch.LongTensor([i]).to(device)
            img = model_G(noise, cls_label) 
            img = (img.data+1)/2.0*255
            img = img.squeeze().permute(1,2,0).detach().cpu().numpy()
            im = Image.fromarray(img.astype(np.uint8))
            im = im.resize((28,28))
            file_name = join(save_dir, f'{i}_{j}.jpg')
            im.save(file_name)
    return 