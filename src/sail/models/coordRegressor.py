# src/simba/models/coord_resnet.py
from __future__ import annotations
from typing import Any, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from ..core.base_model import BaseModelWrapper

def _get_resnet_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    """
    Returns (resnet_without_fc, feat_dim).
    Supports: resnet18, resnet34, resnet50.
    """
    name = name.lower()
    weights = None
    if pretrained:
        if name == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1
        elif name == "resnet34":
            weights = models.ResNet34_Weights.IMAGENET1K_V1
        elif name == "resnet50":
            weights = models.ResNet50_Weights.IMAGENET1K_V1
    if name == "resnet18":
        net = models.resnet18(weights=weights)
    elif name == "resnet34":
        net = models.resnet34(weights=weights)
    elif name == "resnet50":
        net = models.resnet50(weights=weights)
    else:
        raise ValueError(f"Unsupported resnet backbone: {name}")

    feat_dim = net.fc.in_features
    net.fc = nn.Identity()  # we’ll take pooled features directly
    return net, feat_dim

class CoordResNetRegressor(BaseModelWrapper):
    """
    Image through a chosen ResNet backbone; 2D coords through a small projection head.
    Concatenate and predict a single regression value.

    Expected batch keys:
      - 'image':  [B,3,H,W]
      - 'coords': [B,2]   (lat, lon or your chosen ordering)
      - (ignores any neighbor_* keys)
    """

    def __init__(
        self,
        num_outputs: int = 1,              # keep 1 for regression; can be >1 for multi-target
        backbone: str = "resnet18",
        coord_proj_dim: int = 32,
        pretrained: bool = True,
        dropout: float = 0.1,
        img_fc_dim: int | None = None,     # optional extra bottleneck on image feats
    ):
        self.num_outputs = num_outputs
        self.backbone_name = backbone
        self.coord_proj_dim = coord_proj_dim
        self.pretrained = pretrained
        self.dropout = dropout
        self.img_fc_dim = img_fc_dim
        self.net: nn.Module | None = None

    def build(self) -> nn.Module:
        img_backbone, feat_dim = _get_resnet_backbone(self.backbone_name, self.pretrained)

        coord_head = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(inplace=True),
            nn.Linear(64, self.coord_proj_dim), nn.ReLU(inplace=True),
            nn.Dropout(self.dropout),
        )

        if self.img_fc_dim is not None:
            img_head = nn.Sequential(
                nn.Linear(feat_dim, self.img_fc_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
            )
            fused_in = self.img_fc_dim + self.coord_proj_dim
        else:
            img_head = nn.Identity()
            fused_in = feat_dim + self.coord_proj_dim

        final_head = nn.Linear(fused_in, self.num_outputs)

        class _Model(nn.Module):
            def __init__(self, img_backbone, coord_head, img_head, final_head):
                super().__init__()
                self.backbone = img_backbone
                self.coord_head = coord_head
                self.img_head = img_head
                self.final = final_head

            def forward(self, image, coords):
                # image: [B,3,H,W], coords: [B,2]
                z_img = self.backbone(image)            # [B, feat_dim]
                z_img = self.img_head(z_img)            # [B, feat_dim or img_fc_dim]
                # normalize coords to roughly [-1,1] scale (optional, simple)
                z_coord = self.coord_head(coords)       # [B, coord_proj_dim]
                z = torch.cat([z_img, z_coord], dim=1)  # [B, fused_in]
                out = self.final(z)                     # [B, num_outputs]
                return out

        self.net = _Model(img_backbone, coord_head, img_head, final_head)
        return self.net

    def forward(self, batch: Dict[str, Any]):
        # Ignore neighbors if present
        return self.net(batch["image"], batch["coords"])

    def compute_loss(self, pred, batch: Dict[str, Any]):
        # Regression (L2). Targets can be [B] or [B,1].
        target = batch["label"].float().view(pred.size(0), -1)
        pred   = pred.view(pred.size(0), -1)
        return F.l1_loss(pred, target)

    @torch.no_grad()
    def predict(self, batch: Dict[str, Any]):
        out = self.forward(batch)  # [B, num_outputs]
        return out.view(out.size(0), -1)  # [B, num_outputs]

    def save(self, path: str) -> None:
        assert self.net is not None
        torch.save({
            "state_dict": self.net.state_dict(),
            "cfg": {
                "num_outputs": self.num_outputs,
                "backbone": self.backbone_name,
                "coord_proj_dim": self.coord_proj_dim,
                "pretrained": False,       # don’t re-download on load
                "dropout": self.dropout,
                "img_fc_dim": self.img_fc_dim,
            }
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt.get("cfg", {})
        self.num_outputs   = cfg.get("num_outputs", self.num_outputs)
        self.backbone_name = cfg.get("backbone", self.backbone_name)
        self.coord_proj_dim= cfg.get("coord_proj_dim", self.coord_proj_dim)
        self.pretrained    = cfg.get("pretrained", False)
        self.dropout       = cfg.get("dropout", self.dropout)
        self.img_fc_dim    = cfg.get("img_fc_dim", self.img_fc_dim)
        self.build()
        assert self.net is not None
        # Load only matching shapes (skip if you change heads)
        model_sd = self.net.state_dict()
        to_load = {k: v for k, v in ckpt["state_dict"].items()
                   if k in model_sd and model_sd[k].shape == v.shape}
        self.net.load_state_dict(to_load, strict=False)
