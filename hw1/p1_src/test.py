import torch
import os 
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from os import listdir
from os.path import join
from dataset import ImgDataset
from hyper import *
from model import ImgClassifier
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import math
from matplotlib.colors import ListedColormap
from matplotlib.cm import hsv
import torchvision.models as models
        
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

phase = ['val']
features = torch.Tensor()

def fn(module,inputs,outputs):
    global features
    features = inputs[0].squeeze().detach().cpu().numpy()

def main(
    data_dir: str,
    model_path: str,
    num_cls: int
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data_dirs = {p: join(data_dir,f'{p}_50') for p in phase}
    data_names = {p: listdir(data_dirs[p]) for p in phase}
    data_paths = {p: [join(data_dirs[p], name) for name in data_names[p]] for p in phase}
    data_labels = {p: [int(name.split('_')[0]) for name in data_names[p]] for p in phase}
    dataset = {p: ImgDataset(data_paths[p], data_labels[p], train=(False if p == 'val' else True)) for p in phase}
    dataloader = {p: DataLoader(dataset[p], batch_size=1, shuffle=(True if p =='train' else False)) for p in phase}
    model = ImgClassifier(num_cls).to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(model)
    model.model.fc.register_forward_hook(fn)
    correct = 0
    logits_array = []
    labels_array = []
    for i, (inputs, labels) in enumerate(dataloader[phase[0]]):
        inputs, labels = inputs.to(device), labels.squeeze().to(device)
        logits = model(inputs)
        preds = logits.argmax(dim=1)
        logits_array.append(features)
        labels_array.append(preds.squeeze().cpu().detach().numpy())
        correct += (preds==labels).sum().float()/len(inputs)
    print(f'Accuracy: {correct/len(dataloader[phase[0]])}')
    logits_array = np.stack(logits_array, axis=0).astype(float)
    labels_array = np.stack(labels_array, axis=0).astype(int)
    print(logits_array.shape, labels_array.shape)
    #pca = PCA(n_components=10, whiten=False)
    #pca.fit(logits_array)
    #pca_result = np.dot(logits_array,pca.components_.T)
    #print(pca_result.shape)
    tsne_proj = TSNE(n_components=2).fit_transform(logits_array)
    cmap = generate_colormap(50)
    fig, ax = plt.subplots(figsize=(8,8))
    num_categories = 50
    for lab in range(num_categories):
        indices = labels_array==lab
        ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4), label=lab ,alpha=0.5)
    ax.legend(fontsize='xx-small', markerscale=2)
    plt.savefig('../tsne.png')

if __name__ == '__main__':
    data_dir, model_path, num_cls = sys.argv[1], sys.argv[2], int(sys.argv[3])
    main(data_dir, model_path, num_cls)