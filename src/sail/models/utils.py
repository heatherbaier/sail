# src/simba/models/utils.py
import torch.nn as nn
from torchvision import models

class ResNet18Embedding(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        base = models.resnet18(pretrained=pretrained)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.out_dim = base.fc.in_features

    def forward(self, x):
        x = self.features(x)
        return x.view(x.size(0), -1)
