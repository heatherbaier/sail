# src/simba/models/geoconv_native.py
from __future__ import annotations
from typing import Any, Dict, Optional
import importlib
import importlib.util
import types
import torch
import torch.nn as nn
import torch.nn.functional as F

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
        spec = importlib.util.spec_from_file_location("user_geoconv_mod", module_path)
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


class GeoConvNativeRegressor(BaseModelWrapper):
    """
    Adapter that wraps *your* unmodified GeoConv model so it plugs into SIMBA.

    Expectations:
      - Your model's forward signature is: forward(image, coords) -> [B, 1] (or [B])
      - image:  [B,3,H,W]
      - coords: [B,2] (whatever scale/order your model expects; we do NOT normalize)

    Config you pass at init:
      - module_path: dotted import or .py file path to your model definition
      - class_name:  class name of your GeoConv
      - model_kwargs: dict of kwargs to construct your GeoConv exactly as you trained it
    """

    def __init__(
        self,
        # module_path: str,
        # class_name: str,
        model_kwargs: Optional[Dict[str, Any]] = None,
        num_outputs: int = 1,   # kept for symmetry; your model should already output 1-dim for regression
    ):
        # self.module_path = module_path
        # self.class_name = class_name
        self.model_kwargs = model_kwargs or {}
        self.num_outputs = num_outputs
        self.net: nn.Module | None = None

    def build(self) -> nn.Module:
        self.net = ResNet(BasicBlock, [2, 2, 2, 2], num_classes = 1, normalize = False, use_means = False)
        # self.net = GeoConvCls(**self.model_kwargs)
        if not isinstance(self.net, nn.Module):
            raise TypeError(f"{self.class_name} is not a torch.nn.Module")
        return self.net

    def forward(self, batch: Dict[str, Any]):
        # Keep your original calling convention: (image, coords)
        # Do NOT normalize or reorder coordinates here.
        out = self.net(batch["image"], batch["coords"])
        return out

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
                    "num_classes": 1}, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        self.num_classes = ckpt["num_classes"]
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
