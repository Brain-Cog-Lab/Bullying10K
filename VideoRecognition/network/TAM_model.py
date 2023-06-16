import torch
import torch.nn as nn
from torchvision.models.video import r3d_18

class TemporalAdaptiveModule(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(TemporalAdaptiveModule, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.fc1 = nn.Conv3d(in_channels, in_channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv3d(in_channels // reduction, in_channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, t, h, w = x.size()
        y = self.avg_pool(x)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y.expand_as(x)

class TAM(nn.Module):
    def __init__(self, num_classes=10, pretrained=False):
        super(TAM, self).__init__()
        self.base_model = r3d_18(pretrained=pretrained)
        
        # 更改输入通道数为2
        self.base_model.stem[0] = nn.Conv3d(2, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)
        
        self.tam1 = TemporalAdaptiveModule(64)
        self.tam2 = TemporalAdaptiveModule(128)
        self.tam3 = TemporalAdaptiveModule(256)
        self.base_model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.base_model.stem(x)
        x = self.tam1(x)
        x = self.base_model.layer1(x)
        x = self.tam2(x)
        x = self.base_model.layer2(x)
        x = self.tam3(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)
        x = self.base_model.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.base_model.fc(x)
        return x

