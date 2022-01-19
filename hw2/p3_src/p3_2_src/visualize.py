import torch
import torch.nn as nn
import numpy as np 
import os 
import math
import sys
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch import optim
from os import listdir
from os.path import join
from dataset import ImgDataset, SVHN_ImgDataset, SVHN_EvalImgDataset
from model import DANN, DANN_SVHN
from hyper import hyper as hp
from utils import same_seeds, get_data
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv

def generate_colormap(number_of_distinct_colors: int = 80):
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80

    number_of_shades = 7
    number_of_distinct_colors_with_multiply_of_shades = int(math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)

    # Create an array with uniformly drawn floats taken from <0, 1) partition
    linearly_distributed_nums = np.arange(number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades

    # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
    #     but each saw tooth is slightly higher than the one before
    # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades, number_of_distinct_colors_with_multiply_of_shades // number_of_shades)

    # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T

    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]

    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
    initial_cm = hsv(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half

    # Modify lower half in such way that colours towards beginning of partition are darker
    # First colours are affected more, colours closer to the middle are affected less
    lower_half = lower_partitions_half * number_of_shades
    for i in range(3):
        initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8/lower_half)

    # Modify second half in such way that colours towards end of partition are less intense and brighter
    # Colours closer to the middle are affected less, colours closer to the end are affected more
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier

    return ListedColormap(initial_cm)

def main(
    source_dir: str,
    target_dir: str,
    save_dir: str,
    model_path: str,
    target_domain: str
):
    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    s_data_paths, s_data_labels = get_data(source_dir, phase='test')
    t_data_paths, t_data_labels = get_data(target_dir, phase='test')
    if target_domain != 'svhn':
        s_dataset = ImgDataset(s_data_paths, s_data_labels)
        t_dataset = ImgDataset(t_data_paths, t_data_labels)
    else:
        s_dataset = SVHN_EvalImgDataset(s_data_paths, s_data_labels)
        t_dataset = SVHN_EvalImgDataset(t_data_paths, t_data_labels)
    s_dataloader = DataLoader(s_dataset, batch_size=hp.batch_size, shuffle=True)
    t_dataloader = DataLoader(t_dataset, batch_size=hp.batch_size, shuffle=True)
    if target_domain != 'svhn':
        model = DANN(hp.img_dim, hp.n_cls, 45).to(device)
        latent_dim = 810
    else:
        model = DANN_SVHN(hp.img_dim, hp.n_cls, 64).to(device)
        latent_dim = 800
    model.load_state_dict(torch.load(model_path))
    model.eval()

    same_seeds(0)
    alpha = 0 
    latent_array = np.empty((0,latent_dim))
    labels_array = np.empty((0,1))
    domains_array = np.empty((0,1))
    for i, (s_img, s_label) in enumerate(s_dataloader):
        print(f'{len(s_dataloader)}/{i+1}', end='\r')
        s_img, s_label = s_img.to(device), s_label.to(device)
        bs = s_img.size(0)
        if target_domain != 'svhn':
            latent_vec = model.conv(s_img).view(s_img.size(0),-1)
            latent_array = np.concatenate((latent_array, latent_vec.detach().cpu().numpy()), axis=0)
            labels_array = np.concatenate((labels_array, s_label.detach().cpu().numpy()), axis=0)
            domains_labels = np.full((bs, 1), 0)
            domains_array = np.concatenate((domains_array, domains_labels), axis=0)
        elif target_domain == 'svhn':
            latent_vec = model.resnet34(s_img)
            latent_vec = model.conv(latent_vec).view(latent_vec.size(0),-1)
            latent_array = np.concatenate((latent_array, latent_vec.detach().cpu().numpy()), axis=0)
            labels_array = np.concatenate((labels_array, s_label.detach().cpu().numpy()), axis=0)
            domains_labels = np.full((bs, 1), 0)
            domains_array = np.concatenate((domains_array, domains_labels), axis=0)
    for i, (t_img, t_label) in enumerate(t_dataloader):
        print(f'{len(t_dataloader)}/{i+1}', end='\r')
        t_img, t_label = t_img.to(device), t_label.to(device)
        bs = t_img.size(0)
        if target_domain != 'svhn':
            latent_vec = model.conv(t_img).view(t_img.size(0),-1)
            latent_array = np.concatenate((latent_array, latent_vec.detach().cpu().numpy()), axis=0)
            labels_array = np.concatenate((labels_array, t_label.detach().cpu().numpy()), axis=0)
            domains_labels = np.full((bs, 1), 1)
            domains_array = np.concatenate((domains_array, domains_labels), axis=0)
        elif target_domain == 'svhn':
            latent_vec = model.resnet34(t_img)
            latent_vec = model.conv(latent_vec).view(latent_vec.size(0),-1)
            latent_array = np.concatenate((latent_array, latent_vec.detach().cpu().numpy()), axis=0)
            labels_array = np.concatenate((labels_array, t_label.detach().cpu().numpy()), axis=0)
            domains_labels = np.full((bs, 1), 1)
            domains_array = np.concatenate((domains_array, domains_labels), axis=0)

    labels_array = labels_array.reshape(-1)
    domains_array = domains_array.reshape(-1)

    tsne_proj = TSNE(n_components=2).fit_transform(latent_array)
    fig, ax = plt.subplots(figsize=(8,8))
    num_categories = 10
    cmap = generate_colormap(num_categories)
    for lab in range(num_categories):
        indices = labels_array==lab
        ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label=lab ,alpha=0.5)
    ax.legend(fontsize='xx-small', markerscale=2)
    plt.savefig(join(save_dir, f'{target_domain}_class.jpg'))

    fig, ax = plt.subplots(figsize=(8,8))
    num_categories = 2
    cmap = ['red', 'blue']
    for lab in range(num_categories):
        indices = domains_array==lab
        ax.scatter(tsne_proj[indices,0], tsne_proj[indices,1], c=cmap[lab], label=lab ,alpha=0.5)
    ax.legend(fontsize='xx-small', markerscale=2)
    plt.savefig(join(save_dir, f'{target_domain}_domain.jpg'))

if __name__ == '__main__':
    source_dir, target_dir, save_dir, model_path, target_domain = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    main(source_dir, target_dir, save_dir, model_path, target_domain)





