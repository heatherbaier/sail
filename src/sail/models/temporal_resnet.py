# src/simba/models/geoconv_native.py
from __future__ import annotations
from typing import Any, Dict, Optional
import importlib
import importlib.util
import types
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from ..core.base_model import BaseModelWrapper
from .adp_r18 import *


def _import_from_module_path(module_path: str, class_name: str):
    """
    Import `class_name` from a python module path.

    Supports either:
      - dotted module path (e.g., 'myproj.models.geoconv_impl')
      - filesystem path to a .py file (e.g., '/abs/path/to/geoconv_impl.py')
    """
    if module_path.endswith(".py"):
        spec = importlib.util.spec_from_file_location("user_temporal_resnet_mod", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load module from {module_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    else:
        mod = importlib.import_module(module_path)

    cls = getattr(mod, class_name, None)
    if cls is None:
        raise ImportError(f"Class {class_name} not found in {module_path}")
    return cls




# 1. Temporal attention kernel
class TemporalLagKernel(nn.Module):
    
    def __init__(self, embed_dim):
        super().__init__()
        self.query = nn.Parameter(torch.randn(embed_dim))
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
    
    # def forward(self, feats):         # feats: [B, T, C]
    #     keys = self.key_proj(feats)
    #     values = self.value_proj(feats)
    #     # attention over time
    #     attn = torch.softmax((keys @ self.query) / np.sqrt(feats.size(-1)), dim=1)
    #     out = torch.sum(values * attn.unsqueeze(-1), dim=1)  # [B, C]
    #     return out, attn

    def forward(self, feats, mask=None):
        # feats: [B, T, C]
        keys = self.key_proj(feats)
        values = self.value_proj(feats)
    
        raw_attn = (keys @ self.query) / np.sqrt(feats.size(-1))  # [B, T]
    
        if mask is not None:
            raw_attn = raw_attn.masked_fill(mask == 0, -1e9)
    
        attn = torch.softmax(raw_attn, dim=1)
        out  = torch.sum(values * attn.unsqueeze(-1), dim=1)
    
        return out, attn



# ---- Temporal ResNet wrapper ----
class TemporalResNet(nn.Module):

    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        out_dim: int = 1,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """
        Args:
            backbone_name: torchvision backbone (e.g., 'resnet18', 'resnet50')
            pretrained: use ImageNet weights
            out_dim: output dimension of regression/classification head
            model_kwargs: optional dict of args for the backbone (e.g., {'norm_layer': nn.GroupNorm})
        """
        super().__init__()

        model_kwargs = model_kwargs or {}
        base_model = getattr(models, backbone_name)(pretrained=pretrained, **model_kwargs)

        # remove classification head
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        self.embed_dim = base_model.fc.in_features

        self.temporal_kernel = TemporalLagKernel(self.embed_dim)
        self.head = nn.Linear(self.embed_dim, out_dim)


    # def forward(self, x: torch.Tensor):
    #     # x: [B, T, 3, H, W]
    #     B, T, C, H, W = x.shape
    #     feats = []
    #     for t in range(T):
    #         f = self.backbone(x[:, t])  # [B, C, 1, 1]
    #         feats.append(f.squeeze(-1).squeeze(-1))
    #     feats = torch.stack(feats, dim=1)  # [B, T, C]
    #     agg, attn = self.temporal_kernel(feats)
    #     y = self.head(agg)
    #     return y, attn

    def forward(self, x, mask=None):
        # x: [B, T, 3, H, W]
        B, T, C, H, W = x.shape
        feats = []
        for t in range(T):
            f = self.backbone(x[:, t])
            feats.append(f.squeeze(-1).squeeze(-1))
        feats = torch.stack(feats, dim=1)
    
        agg, attn = self.temporal_kernel(feats, mask)  # <--- masked softmax!
        y = self.head(agg)
        return y, attn





import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv
from typing import Dict, Any, Optional
from .temporal_resnet import TemporalResNet   # import your TemporalResNet class

class TemporalResNetModule:
    def __init__(self, model_kwargs: Optional[Dict[str, Any]] = None):
        """
        Wrapper class for TemporalResNet that matches your existing model lifecycle.
        """
        self.model_kwargs = model_kwargs or {}
        self.net: Optional[nn.Module] = None
        self.num_classes = 1  # regression by default

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def build(self) -> nn.Module:
        """
        Construct the TemporalResNet model.
        """
        self.net = TemporalResNet(
            backbone_name=self.model_kwargs.get("backbone_name", "resnet18"),
            pretrained=self.model_kwargs.get("pretrained", True),
            out_dim=self.model_kwargs.get("out_dim", self.num_classes),
            model_kwargs=self.model_kwargs.get("model_kwargs", None),
        )

        if not isinstance(self.net, nn.Module):
            raise TypeError("TemporalResNet is not a torch.nn.Module")

        return self.net

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, batch: Dict[str, Any]):
        """
        batch should include:
            batch["image"] : Tensor [B, T, 3, H, W]
        """
        assert self.net is not None, "Model must be built before forward."
        out, attn = self.net(batch["image"])
        return out, attn

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    def compute_loss(self, pred, batch: Dict[str, Any]):
        """
        Default regression loss (L1).
        """
        target = batch["label"].float().view(pred.size(0), -1)
        pred = pred.view(pred.size(0), -1)
        return F.l1_loss(pred, target)

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, batch: Dict[str, Any]):
        assert self.net is not None, "Model must be built before predict."
        out = self.forward(batch)
        return out.view(out.size(0), -1)

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        assert self.net is not None
        torch.save({
            "state_dict": self.net.state_dict(),
            "num_classes": self.num_classes
        }, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.num_classes = ckpt.get("num_classes", 1)
        self.build()
        assert self.net is not None
        self.net.load_state_dict(ckpt["state_dict"])


