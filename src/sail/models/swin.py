# src/simba/models/geoconv_native.py
from __future__ import annotations
from typing import Any, Dict, Optional
import importlib
import importlib.util
import types
import torch
import torchvision as tv 
import torch.nn as nn
import torch.nn.functional as F

from ..core.base_model import BaseModelWrapper
# from .adp_r18 import *


def _import_from_module_path(module_path: str, class_name: str):
    """
    Import `class_name` from a python module path.

    Supports either:
      - dotted module path (e.g., 'myproj.models.geoconv_impl')
      - filesystem path to a .py file (e.g., '/abs/path/to/geoconv_impl.py')
    """
    if module_path.endswith(".py"):
        spec = importlib.util.spec_from_file_location("user_swim_mod", module_path)
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


def _expand_patch_embed_conv(old_conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """
    Rebuild Swin's stem/patch-embedding conv for a different band count,
    transplanting the pretrained RGB weights instead of discarding them.

    `old_conv` is torchvision Swin's net.features[0][0]:
    Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size), which
    turns the image into the first sequence of patch embeddings -- the
    transformer-stem analogue of a ResNet's conv1. Its weight shape
    depends on in_channels, so unlike every other pretrained layer, it
    can't be reused as-is for N != 3 bands.

    New channels beyond the original 3 are initialized to the mean of the
    pretrained RGB kernels -- a standard warm start for adapting an RGB
    backbone to extra spectral bands (better than random init for the
    first few epochs), not a claim that it's meaningful for e.g. NIR/SWIR
    on its own. The model gets fully retrained afterward regardless.
    """
    new_conv = nn.Conv2d(
        in_channels,
        old_conv.out_channels,
        kernel_size=old_conv.kernel_size,
        stride=old_conv.stride,
        padding=old_conv.padding,
        bias=old_conv.bias is not None,
    )
    with torch.no_grad():
        old_w = old_conv.weight  # (out_channels, 3, kh, kw)
        n_copy = min(in_channels, old_w.shape[1])
        new_conv.weight[:, :n_copy] = old_w[:, :n_copy]
        if in_channels > old_w.shape[1]:
            mean_w = old_w.mean(dim=1, keepdim=True)  # (out, 1, kh, kw)
            new_conv.weight[:, old_w.shape[1]:] = mean_w.expand(-1, in_channels - old_w.shape[1], -1, -1)
        if old_conv.bias is not None:
            new_conv.bias.copy_(old_conv.bias)
    return new_conv


class SwinRegressor(BaseModelWrapper):
    """
    Adapter that wraps *your* unmodified GeoConv model so it plugs into SIMBA.

    Expectations:
      - Your model's forward signature is: forward(image, coords) -> [B, 1] (or [B])
      - image:  [B,in_channels,H,W]
      - coords: [B,2] (whatever scale/order your model expects; we do NOT normalize)

    Config you pass at init:
      - module_path: dotted import or .py file path to your model definition
      - class_name:  class name of your GeoConv
      - model_kwargs: dict of kwargs to construct your GeoConv exactly as you trained it
      - in_channels: number of input image bands (default 3 = plain RGB,
        matching every existing config). For any other value, the
        pretrained stem conv is rebuilt -- see _expand_patch_embed_conv.
    """

    def __init__(
        self,
        # module_path: str,
        # class_name: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
        num_outputs: int = 1,   # kept for symmetry; your model should already output 1-dim for regression
        weights: str = "IMAGENET1K_V1",
        in_channels: int = 3,
    ):
        # self.module_path = module_path
        # self.class_name = class_name
        self.model_kwargs = model_kwargs or {}
        self.num_outputs = num_outputs
        self.in_channels = in_channels
        self.net: nn.Module | None = None

        print(self.model_kwargs)

    def build(self) -> nn.Module:

        self.net = tv.models.swin_v2_t(weights="IMAGENET1K_V1")
        if self.in_channels != 3:
            old_conv = self.net.features[0][0]
            self.net.features[0][0] = _expand_patch_embed_conv(old_conv, self.in_channels)
        self.net.head = nn.Sequential(
            nn.Linear(self.net.head.in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

        return self.net

        # self.net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes = 1, normalize = False, use_means = False)
        # # self.net = GeoConvCls(**self.model_kwargs)
        # if not isinstance(self.net, nn.Module):
        #     raise TypeError(f"{self.class_name} is not a torch.nn.Module")
        # return self.net

    def forward(self, batch: Dict[str, Any]):
        # Keep your original calling convention: (image, coords)
        # Do NOT normalize or reorder coordinates here.
        out = self.net(batch["image"])#, batch["coords"])
        return out, None

    def compute_loss(self, pred, batch: Dict[str, Any]):
        # L1 regression by default; adjust if your training used MSE etc.
        target = batch["label"].float().view(pred.size(0), -1)
        pred   = pred.view(pred.size(0), -1)
        return F.l1_loss(pred, target)

    @torch.no_grad()
    def predict(self, batch: Dict[str, Any]):
        out = self.forward(batch)
        return out.view(out.size(0), -1)

    def save(self, path: str) -> None:
        assert self.net is not None
        torch.save({"state_dict": self.net.state_dict(),
                    "num_classes": 1,
                    "in_channels": self.in_channels}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.num_classes = ckpt["num_classes"]
        # .get() with the current default: checkpoints saved before this
        # option existed have no "in_channels" key, and were always 3-band.
        self.in_channels = ckpt.get("in_channels", self.in_channels)
        self.build()
        assert self.net is not None
        self.net.load_state_dict(ckpt["state_dict"])


        
        # ckpt = torch.load(path, map_location="cpu")
        # cfg = ckpt.get("cfg", {})
        # self.module_path = cfg.get("module_path", self.module_path)
        # self.class_name = cfg.get("class_name", self.class_name)
        # self.model_kwargs = cfg.get("model_kwargs", self.model_kwargs)
        # self.num_outputs = cfg.get("num_outputs", self.num_outputs)

        # self.build()
        # assert self.net is not None
        # # load matching keys (in case you changed anything later)
        # model_sd = self.net.state_dict()
        # to_load = {k: v for k, v in ckpt["state_dict"].items()
        #            if k in model_sd and model_sd[k].shape == v.shape}
        # self.net.load_state_dict(to_load, strict=False)
