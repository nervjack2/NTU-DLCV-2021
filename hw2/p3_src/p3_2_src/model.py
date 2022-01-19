import torch
import torchvision.models as models
import torch.nn as nn 
from torch.autograd import Function

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

def weights_init(m):
    classname = m.__class__.__name__

    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class DANN(nn.Module):
    """
        Classify whether the image is real or generated.
    """
    def __init__(self, img_dim, n_cls, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, latent_dim, 5, 2, bias=False),
            nn.BatchNorm2d(latent_dim),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(latent_dim, latent_dim*2, 5, 2, bias=False),
            nn.BatchNorm2d(latent_dim*2),
            nn.Dropout(),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.num_cls = nn.Sequential(
            nn.Linear(810, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(100, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 10),
            nn.LogSoftmax(dim=1)
        )
        self.domain_cls = nn.Sequential(
            nn.Linear(810, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x, alpha):
        feature = self.conv(x)
        feature = feature.view(feature.size(0),-1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        x_cls = self.num_cls(feature)
        x_domain = self.domain_cls(reverse_feature)
        return x_cls.view(x_cls.size(0),-1), x_domain.view(-1)

class DANN_SVHN(nn.Module):
    """
        Classify whether the image is real or generated.
    """
    def __init__(self, img_dim, n_cls, latent_dim):
        super().__init__()
        self.resnet34 = models.resnet34(pretrained = False)
        self.resnet34 = nn.Sequential(*(list(self.resnet34.children())[:-2]))
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=512,
                                out_channels=256, 
                                kernel_size=4, 
                                stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=256,
                                out_channels=128, 
                                kernel_size=4, 
                                stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=128,
                                out_channels=64, 
                                kernel_size=4, 
                                stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 50, kernel_size=5),
            nn.BatchNorm2d(50),
            nn.Dropout2d(),
            nn.ReLU(True)
        )
        self.num_cls = nn.Sequential(
            nn.Linear(800,100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Dropout2d(),
            nn.Linear(100,100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100,10),
            nn.LogSoftmax(dim=1)
        )
        self.domain_cls = nn.Sequential(
            nn.Linear(800, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(True),
            nn.Linear(100, 2),
            nn.LogSoftmax(dim=1)
        )
        self.apply(weights_init)

    def forward(self, x, alpha):
        x = self.resnet34(x)
        feature = self.conv(x)
        feature = feature.view(feature.size(0),-1)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        x_cls = self.num_cls(feature)
        x_domain = self.domain_cls(reverse_feature)
        return x_cls.view(x_cls.size(0),-1), x_domain.view(x_domain.size(0),-1)