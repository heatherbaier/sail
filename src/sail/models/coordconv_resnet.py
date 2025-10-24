# src/simba/models/coordconv_resnet.py
from __future__ import annotations
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torchvision import models

from ..core.base_model import BaseModelWrapper  # keep consistent with your repo


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


def _patch_first_conv(resnet: nn.Module, extra_in_channels: int, init_mode: str = "avg") -> None:
    """
    Replace resnet.conv1 to accept (3 + extra_in_channels). Initialize new
    channel weights either as zeros or as the RGB-kernel average.
    """
    if extra_in_channels <= 0:
        return
    old: nn.Conv2d = resnet.conv1
    new_in = old.in_channels + extra_in_channels

    new = nn.Conv2d(
        in_channels=new_in,
        out_channels=old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        dilation=old.dilation,
        groups=old.groups,
        bias=(old.bias is not None),
        padding_mode=old.padding_mode,
    )

    with torch.no_grad():
        # copy RGB weights
        new.weight[:, :old.in_channels] = old.weight
        if init_mode == "avg":
            avg = old.weight.mean(dim=1, keepdim=True)  # [C_out,1,kh,kw]
            for k in range(extra_in_channels):
                new.weight[:, old.in_channels + k : old.in_channels + k + 1] = avg
        else:  # zeros
            new.weight[:, old.in_channels:] = 0.0
        if old.bias is not None:
            new.bias.copy_(old.bias)

    resnet.conv1 = new


def _coords_to_planes(
    coords: torch.Tensor,  # [B,2] = [lat, lon] in degrees
    H: int,
    W: int,
    encoding: str = "raw",      # {"raw","sincos"}
    norm: str = "global",       # {"global","none"}
) -> torch.Tensor:
    """
    Build per-sample full-resolution coordinate planes.
      - raw:   2 planes => [lat, lon] (optionally normalized)
      - sincos:4 planes => [sin(lat), cos(lat), sin(lon), cos(lon)]
    """
    assert coords.ndim == 2 and coords.size(1) == 2, "coords must be [B,2]=(lat,lon)"
    device = coords.device
    lat = coords[:, 0]
    lon = coords[:, 1]

    if norm == "global":
        lat_n = torch.clamp(lat / 90.0, -1, 1)
        lon_n = torch.clamp(lon / 180.0, -1, 1)
    elif norm == "none":
        lat_n, lon_n = lat, lon
    else:
        raise ValueError("norm must be 'global' or 'none'")

    if encoding == "raw":
        feats = torch.stack([lat_n, lon_n], dim=1)  # [B,2]
    elif encoding == "sincos":
        lat_r = lat * torch.pi / 180.0
        lon_r = lon * torch.pi / 180.0
        feats = torch.stack(
            [torch.sin(lat_r), torch.cos(lat_r), torch.sin(lon_r), torch.cos(lon_r)],
            dim=1,
        )  # [B,4]
    else:
        raise ValueError("encoding must be 'raw' or 'sincos'")

    B, K = feats.shape
    planes = feats.view(B, K, 1, 1).expand(B, K, H, W).to(device)
    return planes


# ------------------------------
# Model
# ------------------------------
class CoordConvResNetRegressor(BaseModelWrapper):
    """
    CoordConv-style regression:
      - Build per-sample coordinate planes.
      - Concatenate with RGB (=> 3+K channels).
      - Feed through a ResNet backbone whose conv1 has been patched to accept 3+K.
      - Final linear head for regression.

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
        coord_encoding: str = "raw",    # {"raw","sincos"}
        coord_norm: str = "global",     # {"global","none"}
        firstconv_init: str = "avg",    # {"avg","zeros"}
        dropout: float = 0.0,
        img_fc_dim: int | None = None,  # optional bottleneck
    ):
        self.num_outputs = num_outputs
        self.backbone_name = backbone
        self.pretrained = pretrained
        self.coord_encoding = coord_encoding
        self.coord_norm = coord_norm
        self.firstconv_init = firstconv_init
        self.dropout = dropout
        self.img_fc_dim = img_fc_dim

        self.net: nn.Module | None = None
        self._extra_in: int = 2 if coord_encoding == "raw" else 4

    def build(self) -> nn.Module:
        resnet, feat_dim = _get_resnet_backbone(self.backbone_name, self.pretrained)

        # Patch conv1 to accept extra coord channels
        _patch_first_conv(resnet, extra_in_channels=self._extra_in, init_mode=self.firstconv_init)

        if self.img_fc_dim is not None:
            head = nn.Sequential(
                nn.Linear(feat_dim, self.img_fc_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(self.img_fc_dim, self.num_outputs),
            )
        else:
            head = nn.Linear(feat_dim, self.num_outputs)

        class _Net(nn.Module):
            def __init__(self, backbone: nn.Module, head: nn.Module, parent: "CoordConvResNetRegressor"):
                super().__init__()
                self.backbone = backbone
                self.head = head
                self.parent = parent

            def forward(self, image: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
                # image: [B,3,H,W], coords: [B,2]
                B, _, H, W = image.shape
                coord_planes = _coords_to_planes(
                    coords, H, W,
                    encoding=self.parent.coord_encoding,
                    norm=self.parent.coord_norm
                )  # [B,K,H,W]
                x = torch.cat([image, coord_planes], dim=1)  # [B,3+K,H,W]
                z = self.backbone(x)                         # [B, feat_dim]
                out = self.head(z)                           # [B, num_outputs]
                return out

        self.net = _Net(resnet, head, self)
        return self.net

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        return self.net(batch["image"], batch["coords"])

    def compute_loss(self, pred: torch.Tensor, batch: Dict[str, Any]) -> torch.Tensor:
        target = batch["label"].float().view(pred.size(0), -1)
        pred   = pred.view(pred.size(0), -1)
        return nn.functional.l1_loss(pred, target)

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
                "firstconv_init": self.firstconv_init,
                "dropout": self.dropout,
                "img_fc_dim": self.img_fc_dim,
            }
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt.get("cfg", {})
        self.num_outputs     = cfg.get("num_outputs", self.num_outputs)
        self.backbone_name   = cfg.get("backbone", self.backbone_name)
        self.pretrained      = cfg.get("pretrained", False)
        self.coord_encoding  = cfg.get("coord_encoding", self.coord_encoding)
        self.coord_norm      = cfg.get("coord_norm", self.coord_norm)
        self.firstconv_init  = cfg.get("firstconv_init", self.firstconv_init)
        self.dropout         = cfg.get("dropout", self.dropout)
        self.img_fc_dim      = cfg.get("img_fc_dim", self.img_fc_dim)
        self._extra_in       = 2 if self.coord_encoding == "raw" else 4
        self.build()
        assert self.net is not None
        self.net.load_state_dict(ckpt["state_dict"], strict=False)
