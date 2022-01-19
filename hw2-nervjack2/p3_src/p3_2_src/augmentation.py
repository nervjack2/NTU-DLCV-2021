import torch 
import numpy as np
import torchvision.transforms as T


img_augment = T.Compose([
    T.RandomRotation(20),
    T.ColorJitter(0.5,0.5,0.5,0.5)
])

