import torch
import torchvision.models as models
import torch.nn as nn 

def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Upsampling(nn.Module):
    """
        Upsampling block.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.uconv = nn.Sequential(
            nn.ConvTranspose2d(in_dim, out_dim, 4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU()
        )
        self.apply(weights_init)
    def forward(self, x):
        x = self.uconv(x)
        return x

class Generator(nn.Module):
    """
        Generate a image of shape (64, 64).
    """
    def __init__(self, vec_dim, n_cls, latent_dim):
        super().__init__()
        self.fn = nn.Sequential(
            nn.ConvTranspose2d(vec_dim, latent_dim*8, 4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(latent_dim*8),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            Upsampling(latent_dim*8, latent_dim*4),
            Upsampling(latent_dim*4, latent_dim*2),
            Upsampling(latent_dim*2, latent_dim),
            nn.ConvTranspose2d(latent_dim, 3, 4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        self.cls_emb = nn.Embedding(n_cls, vec_dim)
        self.apply(weights_init)
    def forward(self, x, y):
        cls_emb = self.cls_emb(y)
        x = torch.mul(x, cls_emb)
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = self.fn(x)
        x = self.conv(x)
        return x 

class Downsampling(nn.Module):
    """
    Downsampling block.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dconv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.LeakyReLU(0.2)
        )
        self.apply(weights_init)
    def forward(self, x):
        x = self.dconv(x)
        return x 

class Discriminator(nn.Module):
    """
        Classify whether the image is real or generated.
    """
    def __init__(self, img_dim, n_cls, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, latent_dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),
            Downsampling(latent_dim, latent_dim*2),
            Downsampling(latent_dim*2, latent_dim*4),
            Downsampling(latent_dim*4, latent_dim*8),
            nn.Sigmoid()
        )
        self.real_cls = nn.Sequential(
            nn.Conv2d(latent_dim*8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        self.num_cls = nn.Sequential(
            nn.Conv2d(latent_dim*8, 10, 4, 1, 0, bias=False),
            nn.LogSoftmax(dim=1)
        )
        self.apply(weights_init)

    def forward(self, x):
        feature = self.conv(x)
        x_cls = self.num_cls(feature)
        x_real = self.real_cls(feature)
        return x_real.view(-1), x_cls.view(x_cls.size(0),-1)