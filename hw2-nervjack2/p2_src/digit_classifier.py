import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from dataset import ImgDataset, EvalImgDataset
from torch.utils.data import DataLoader

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path, map_location = "cuda")
    model.load_state_dict(state['state_dict'])
    print('model loaded from %s' % checkpoint_path)

if __name__ != '__main__':
    net = Classifier()
    path = "../Classifier.pth"
    load_checkpoint(path, net)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if torch.cuda.is_available():
        net = net.to(device)


def log_acc(data_dir, epoch):
    data_names = os.listdir(data_dir)
    data_paths = [os.path.join(data_dir, x) for x in data_names]
    data_lables = [int(x.split('_')[0]) for x in data_names]
    dataset = EvalImgDataset(data_paths, data_lables)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)
    correct = 0
    for x,y in dataloader:
        x, y = x.to(device), y.squeeze().to(device)
        logits = net(x) 
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum() / len(x)
    return correct/len(dataloader)

if __name__ == '__main__':
    
    # load digit classifier
    net = Classifier()
    path = "../Classifier.pth"
    load_checkpoint(path, net)

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print('Device used:', device)
    if torch.cuda.is_available():
        net = net.to(device)
    print(net)
    data_dir = '../p2_result'
    data_names = os.listdir(data_dir)
    data_paths = [os.path.join(data_dir, x) for x in data_names]
    data_lables = [int(x.split('_')[0]) for x in data_names]
    dataset = ImgDataset(data_paths, data_lables)
    dataloader = DataLoader(dataset, batch_size=8)

    correct = 0
    for x,y in dataloader:
        x, y = x.to(device), y.squeeze().to(device)
        logits = net(x) 
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum() / len(x)
    print(f'Acc = {correct/len(dataloader)}')