
import torch
import torch.nn as nn
from torchvision import models

features = torch.Tensor()
def hook_fn(module,inputs,outputs):
    global features
    features = inputs[0]

class Classifier(nn.Module):
    def __init__(self, n_cls, in_dim, hidden_dim=4096):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, n_cls)
        )
    def forward(self, x):
        return self.net(x)

class ImgClassifier(nn.Module):
    def __init__(self, model_path, fix_backbone, n_cls, in_dim, hidden_dim=4096):
        super(ImgClassifier, self).__init__()
        self.feat_extractor = models.resnet50(pretrained=False)
        if model_path != 'None':
            self.feat_extractor.load_state_dict(torch.load(model_path))
        if fix_backbone:
            for param in self.feat_extractor.parameters():
                param.requires_grad = False
        self.feat_extractor.fc.register_forward_hook(hook_fn)
        self.classifier = Classifier(n_cls, in_dim, hidden_dim)

    def forward(self, x):
        _ = self.feat_extractor(x)
        pred = self.classifier(features)
        return pred 

class EvalImgClassifier(nn.Module):
    def __init__(self, n_cls, in_dim, hidden_dim=4096):
        super(EvalImgClassifier, self).__init__()
        self.feat_extractor = models.resnet50(pretrained=False)
        self.feat_extractor.fc.register_forward_hook(hook_fn)
        self.classifier = Classifier(n_cls, in_dim, hidden_dim)

    def forward(self, x):
        _ = self.feat_extractor(x)
        pred = self.classifier(features)
        return pred 