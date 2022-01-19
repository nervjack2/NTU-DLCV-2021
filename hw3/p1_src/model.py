import torchvision.models as models
import torch.nn as nn 
from pytorch_pretrained_vit import ViT

class ViT_Classifier(nn.Module):
    def __init__(self, model_name, num_cls, pretrained=True):
        super().__init__()
        self.model = ViT(model_name, pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(in_features, 100, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(100, num_cls, bias=True),
        )
    
    def forward(self, x):
        return self.model(x)

        