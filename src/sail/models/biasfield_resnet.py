# src/simba/models/biasfield_resnet.py
from __future__ import annotations
from typing import Any, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from ..core.base_model import BaseModelWrapper  # adjust path if needed


# ------------------------------
# Helpers
# ------------------------------
def _get_resnet_backbone(name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    """
    Return (resnet_without_fc, feat_dim). Supports resnet18/34/50.
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
    net.fc = nn.Identity()
    return net, feat_dim


def _encode_coords(coords: torch.Tensor,
                   encoding: str = "raw",    # {"raw","sincos"}
                   norm: str = "global"      # {"global","none"}
                   ) -> torch.Tensor:
    """
    coords: [B,2] = [lat, lon] in degrees.
    Returns [B,K] where K=2 (raw) or K=4 (sincos).
    """
    assert coords.ndim == 2 and coords.size(1) == 2, "coords must be [B,2]=(lat,lon)"
    lat, lon = coords[:, 0], coords[:, 1]

    if norm == "global":
        lat_n = torch.clamp(lat / 90.0, -1, 1)
        lon_n = torch.clamp(lon / 180.0, -1, 1)
    elif norm == "none":
        lat_n, lon_n = lat, lon
    else:
        raise ValueError("norm must be 'global' or 'none'")

    if encoding == "raw":
        return torch.stack([lat_n, lon_n], dim=1)  # [B,2]
    elif encoding == "sincos":
        lat_r = lat * torch.pi / 180.0
        lon_r = lon * torch.pi / 180.0
        return torch.stack(
            [torch.sin(lat_r), torch.cos(lat_r), torch.sin(lon_r), torch.cos(lon_r)],
            dim=1
        )  # [B,4]
    else:
        raise ValueError("encoding must be 'raw' or 'sincos'")
# ------------------------------


class BiasFieldResNetRegressor(BaseModelWrapper):
    """
    ResNet image encoder with coordinate-derived bias field:
      z_img = ResNet(image)            # [B, D]
      b     = MLP(coords_enc)          # [B, D]
      z     = z_img + alpha * b
      yhat  = Linear(z)                # regression

    Batch keys expected:
      - 'image':  [B,3,H,W]
      - 'coords': [B,2]  (lat, lon in degrees)
      - 'label':  [B] or [B,1]
    """

    def __init__(
        self,
        num_outputs: int = 1,
        backbone: str = "resnet18",
        pretrained: bool = True,
        coord_encoding: str = "raw",      # {"raw","sincos"}
        coord_norm: str = "global",       # {"global","none"}
        coord_hidden: int = 64,           # hidden width of the coord MLP
        dropout: float = 0.1,
        img_fc_dim: Optional[int] = None, # optional bottleneck on image feats
        unit_norm: bool = False,          # L2-normalize z_img and b before sum
        alpha_init: float = 1.0,          # initial scale for bias field
        learn_alpha: bool = True          # learnable scalar scale
    ):
        self.num_outputs = num_outputs
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.coord_encoding = coord_encoding
        self.coord_norm = coord_norm
        self.coord_hidden = coord_hidden
        self.dropout = dropout
        self.img_fc_dim = img_fc_dim
        self.unit_norm = unit_norm

        self.net: nn.Module | None = None
        self._enc_dim = 2 if coord_encoding == "raw" else 4
        self._alpha_init = alpha_init
        self._learn_alpha = learn_alpha

    def build(self) -> nn.Module:
        resnet, feat_dim = _get_resnet_backbone(self.backbone_name, self.pretrained)

        # Optional bottleneck on image features
        if self.img_fc_dim is not None:
            img_head = nn.Sequential(
                nn.Linear(feat_dim, self.img_fc_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
            )
            z_dim = self.img_fc_dim
        else:
            img_head = nn.Identity()
            z_dim = feat_dim

        # Coord MLP outputs a bias vector in the SAME space as z_img
        coord_head = nn.Sequential(
            nn.Linear(self._enc_dim, self.coord_hidden),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(self.dropout),
            nn.Linear(self.coord_hidden, z_dim)
        )

        final_head = nn.Linear(z_dim, self.num_outputs)

        # Learnable/global alpha
        alpha_param = nn.Parameter(torch.tensor(self._alpha_init, dtype=torch.float32)) \
                      if self._learn_alpha else None
        alpha_const = self._alpha_init

        class _Net(nn.Module):
            def __init__(self, backbone, img_head, coord_head, final_head,
                         parent: "BiasFieldResNetRegressor",
                         alpha_param: Optional[nn.Parameter],
                         alpha_const: float):
                super().__init__()
                self.backbone = backbone
                self.img_head = img_head
                self.coord_head = coord_head
                self.final = final_head
                self.parent = parent
                self.alpha_param = alpha_param
                self.alpha_const = alpha_const

            def forward(self, image: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
                # Encode image → z_img
                z_img = self.backbone(image)           # [B, feat_dim]
                z_img = self.img_head(z_img)           # [B, z_dim]

                # Encode coords → bias vector b
                enc = _encode_coords(coords,
                                     encoding=self.parent.coord_encoding,
                                     norm=self.parent.coord_norm)  # [B, K]
                b = self.coord_head(enc)                # [B, z_dim]

                if self.parent.unit_norm:
                    # Avoid division by zero
                    z_img = F.normalize(z_img, dim=1)
                    b = F.normalize(b, dim=1)

                # Scale bias
                if self.alpha_param is not None:
                    alpha = self.alpha_param
                else:
                    # Use constant alpha (broadcast)
                    alpha = torch.tensor(self.alpha_const, device=z_img.device)

                z = z_img + alpha * b                  # bias-field injection
                out = self.final(z)                    # [B, num_outputs]
                return out

        self.net = _Net(resnet, img_head, coord_head, final_head,
                        self, alpha_param, alpha_const)
        return self.net

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.net(batch["image"], batch["coords"])

    def compute_loss(self, pred: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        target = batch["label"].float().view(pred.size(0), -1)
        pred   = pred.view(pred.size(0), -1)
        return F.l1_loss(pred, target)

    @torch.no_grad()
    def predict(self, batch: Dict[str, Any]) -> torch.Tensor:
        out = self.forward(batch)
        return out.view(out.size(0), -1)

    def save(self, path: str) -> None:
        assert self.net is not None
        torch.save({
            "state_dict": self.net.state_dict(),
            "cfg": {
                "num_outputs": self.num_outputs,
                "backbone": self.backbone_name,
                "pretrained": False,
                "coord_encoding": self.coord_encoding,
                "coord_norm": self.coord_norm,
                "coord_hidden": self.coord_hidden,
                "dropout": self.dropout,
                "img_fc_dim": self.img_fc_dim,
                "unit_norm": self.unit_norm,
                "alpha_init": self._alpha_init,
                "learn_alpha": self._learn_alpha,
            }
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt.get("cfg", {})
        self.num_outputs    = cfg.get("num_outputs", self.num_outputs)
        self.backbone_name  = cfg.get("backbone", self.backbone_name)
        self.pretrained     = cfg.get("pretrained", False)
        self.coord_encoding = cfg.get("coord_encoding", self.coord_encoding)
        self.coord_norm     = cfg.get("coord_norm", self.coord_norm)
        self.coord_hidden   = cfg.get("coord_hidden", self.coord_hidden)
        self.dropout        = cfg.get("dropout", self.dropout)
        self.img_fc_dim     = cfg.get("img_fc_dim", self.img_fc_dim)
        self.unit_norm      = cfg.get("unit_norm", self.unit_norm)
        self._alpha_init    = cfg.get("alpha_init", self._alpha_init)
        self._learn_alpha   = cfg.get("learn_alpha", self._learn_alpha)

        self._enc_dim = 2 if self.coord_encoding == "raw" else 4
        self.build()
        assert self.net is not None
        self.net.load_state_dict(ckpt["state_dict"], strict=False)
