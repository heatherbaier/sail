# src/simba/models/resnet18.py
from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from ..core.base_model import BaseModelWrapper

class ResNet18Classifier(BaseModelWrapper):
    def __init__(self, num_classes: int = 10, pretrained: bool = True):
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.net: nn.Module | None = None

    def build(self) -> nn.Module:
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if self.pretrained else None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, 1)  # regression head
        self.net = backbone
        return self.net

    def forward(self, batch: Dict[str, Any]):
        # coords are present in batch for API consistency, but ignored here
        return self.net(batch["image"])

    def compute_loss(self, pred, batch):
        target = batch["label"].float().view(-1, 1)
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def predict(self, batch):
        return self.net(batch["image"]).view(-1)

    def save(self, path: str) -> None:
        assert self.net is not None
        torch.save({"state_dict": self.net.state_dict(),
                    "num_classes": self.num_classes}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.num_classes = ckpt["num_classes"]
        self.build()
        assert self.net is not None
        self.net.load_state_dict(ckpt["state_dict"])
