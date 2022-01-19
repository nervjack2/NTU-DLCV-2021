import torch
import torch.nn as nn


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ProtoNet(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(ProtoNet, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

class ParaDisMatrix(nn.Module):
    def __init__(self, emb_dim=1600):
        super(ParaDisMatrix, self).__init__()
        self.M =  nn.parameter.Parameter(torch.eye(emb_dim))
    def forward(self, x):
        return torch.matmul(x, self.M)

class ParaDis(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64, emb_dim=1600):
        super(ParaDis, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.para = ParaDisMatrix(emb_dim)

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)