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
        self.num_cls = nn.Sequential(
            nn.Conv2d(latent_dim*8, 10, 4, 1, 0, bias=False),
            nn.LogSoftmax(dim=1)
        )
        self.apply(weights_init)

    def forward(self, x):
        feature = self.conv(x)
        x_cls = self.num_cls(feature)
        return x_cls.view(x_cls.size(0),-1)