import torchvision.models as models
import torch.nn as nn 


class VGG16_FCN32(nn.Module):
    def __init__(self, num_cls):
        super().__init__()
        self.model = models.vgg16(pretrained=True)
        self.fc = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=(2, 2), stride=(1, 1)),  
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1)), 
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_cls, kernel_size=(1, 1), stride=(1, 1)), 
            nn.ConvTranspose2d(num_cls, num_cls, 64 , 32 , 0, bias=False),
        )   
      
    def forward(self, x):
        x = self.model.features(x)
        x = self.fc(x)
        return x

class VGG16_FCN16(nn.Module):
    def __init__(self, num_cls, pretrained = True):
        super(VGG16_FCN16, self).__init__()
        self.vgg = models.vgg16(pretrained=True)
        self.start_pool4 = nn.Sequential(*list(self.vgg.features.children())[:24])
        self.pool4_pool5 = nn.Sequential(*list(self.vgg.features.children())[24:])
        self.vgg.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=(2, 2), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, 4096, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            
            nn.Conv2d(4096, num_cls, kernel_size=(1, 1), stride=(1, 1)),
            nn.ConvTranspose2d(num_cls, 512, 4 , 2 , 0, bias=False)
        )
        self.upsample16 = nn.ConvTranspose2d(512, num_cls, 16 , 16 , 0, bias=False)
        
    def  forward (self, x) :        
        pool4_output = self.start_pool4(x) #pool4 output size torch.Size([64, 512, 16, 16])
        x = self.pool4_pool5(pool4_output)
        x = self.vgg.classifier(x)    # 2xconv7 output size torch.Size([64, 512, 16, 16])
        x = self.upsample16(x+pool4_output)
        return x
