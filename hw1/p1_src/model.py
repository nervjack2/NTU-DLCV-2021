import torchvision.models as models
import torch.nn as nn 


class ImgClassifier(nn.Module):
    def __init__(self, num_cls):
        super().__init__()
        self.model = models.inception_v3(pretrained=True)
        self.model.fc = nn.Sequential(
            nn.Linear(2048, 1000, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(1000, 50, bias=True),
        )
    
    def forward(self, x):
        return self.model(x)

        