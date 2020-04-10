# model.py

import torch
import torch.nn as nn

from torchvision import models

class MRNet(nn.Module):
    def __init__(self):
        super().__init__()
        #self.model = models.alexnet(pretrained=True)
        #self.model = models.googlenet(pretrained=True)
        #self.model = models.densenet169(pretrained=True)
        self.model = models.densenet121(pretrained=True)
        #self.model = models.densenet161(pretrained=True)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(1024, 1)
        #self.classifier = nn.Linear(256, 1)

    # change this to adapt to different networks
    def forward(self, x):
        x = torch.squeeze(x, dim=0) # only batch size 1 supported
        x = self.model.features(x)
        # make sure that gap returns size 256
        x = self.gap(x).view(x.size(0), -1)
        #print('x size', x.size())
        x = torch.max(x, 0, keepdim=True)[0]
        #print('x size max', x.size())
        x = self.classifier(x)
        return x

