# src/sail/core/builders.py

from .registries import ModelRegistry, DatasetRegistry, ExplainerRegistry
from ..data.adapters import JSONGeoAdapter, resolve_json_paths
from ..explain.simba import Simba  # (later: also SIAExplainer, etc.)
from ..training.trainer import Trainer, TrainConfig

import os
import torch

def build_dataset(cfg_dataset: dict):
    from ..data.adapters import JSONGeoAdapter, resolve_json_paths

    ys_path, coords_path, dup_path = resolve_json_paths(
        data_root=cfg_dataset["data_root"],
        prefix=cfg_dataset["prefix"],
        with_neighbors=False,   # explicitly off
    )

    ds = JSONGeoAdapter(
        root_dir=cfg_dataset["data_root"],
        ys_path=ys_path,
        coords_path=coords_path,
        dup_path=dup_path,
        batch_size=cfg_dataset["batch_size"],
        img_size=tuple(cfg_dataset.get("img_size", [224, 224])),
        num_workers=cfg_dataset.get("num_workers", 0),
        ckpt_dir=cfg_dataset.get("output_dir", "."),
    )

    return ds


def build_model(cfg_model: dict):
    name = cfg_model["name"]
    params = cfg_model.get("params", {})
    model_wrapper_cls = ModelRegistry.get(name)
    # print(params)
    model_wrapper = model_wrapper_cls(**params)
    model = model_wrapper.build()
    return model_wrapper, model


def build_trainer(cfg_trainer: dict, model_wrapper, dataset, model_name: str, batch_size: int, ckpt_dir: str):
    train_cfg = TrainConfig(
        epochs=cfg_trainer["epochs"],
        lr=cfg_trainer["lr"],
        ckpt_dir = ckpt_dir,
        device=cfg_trainer.get("device", "cuda"),
        # save_every=cfg_trainer.get("save_every", 1),
        # eval_every=cfg_trainer.get("eval_every", 1),
    )
    return Trainer(model_wrapper, dataset, train_cfg, model_name, batch_size)

def build_explainer(cfg_exp: dict, net):
    # For now default to Simba. Later you can branch on cfg_exp["method"]
    return Simba(
        model=net,
        ckpt_dir=cfg_exp["ckpt_dir"],
        device="cuda",  # could expose in cfg_exp later
    )
