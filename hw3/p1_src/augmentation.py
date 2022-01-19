import torch 
import numpy as np
import torchvision.transforms as T


img_augment = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.2),
    T.RandomApply([
        T.RandomResizedCrop(32,scale=(0.5,1.0))
    ], p=0.5),
    T.RandomApply([
        T.RandomAffine(
            degrees=30,
            scale=(0.8,1.2),
            translate=(0.25,0.5),
            shear=13,
        )
    ], p=0.5),
    T.RandomApply([
        T.ColorJitter(brightness=(0.7,1.3), contrast=(0.9,1.5))
    ], p=0.5)
])

