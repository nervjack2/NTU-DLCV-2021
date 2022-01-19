import torch
import torchvision
import sys
import numpy as np 
import torchvision.transforms as transforms
from PIL import Image
from os.path import join
from model import Generator
from hyper import hyper as hp
from utils import same_seeds

transform = transforms.Compose([
                            transforms.Resize(28),
                        ])

def main(
    save_dir: str,
    model_path: str 
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_G = Generator(hp.vec_dim, hp.n_cls, hp.latent_dim).to(device)
    model_G.eval()
    model_G.load_state_dict(torch.load(model_path))
    same_seeds(0)
    for i in range(10):
        for j in range(100):
            noise = torch.randn(1, hp.vec_dim).to(device)
            cls_label = torch.LongTensor([i]).to(device)
            img = model_G(noise, cls_label) 
            img = (img.data+1)/2.0*255
            img = img.squeeze().permute(1,2,0).detach().cpu().numpy()
            im = Image.fromarray(img.astype(np.uint8))
            im = transform(im)
            file_name = join(save_dir, '{}_{:03d}.jpg'.format(i,j+1))
            im.save(file_name)

if __name__ == '__main__':
    save_dir, model_path = sys.argv[1], sys.argv[2]
    main(save_dir, model_path)



