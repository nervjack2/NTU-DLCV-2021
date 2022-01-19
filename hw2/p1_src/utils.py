import random
import torch
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

def gen_img(model_G, save_dir, hp, device):
    for i in range(1000):
        print(f'{1000}/{i+1}', end='\r')
        noise = torch.randn(1, hp.vec_dim).to(device)
        img = model_G(noise) 
        img = (img.data+1)/2.0*255
        img = img.squeeze().permute(1,2,0).detach().cpu().numpy()
        im = Image.fromarray(img.astype(np.uint8))
        file_name = join(save_dir, f'{i}.jpg')
        im.save(file_name)
    return 