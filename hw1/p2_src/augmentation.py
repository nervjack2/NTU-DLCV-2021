import torch 
import numpy as np
import torchvision.transforms as T


def img_augment(img, label):
    return img, label