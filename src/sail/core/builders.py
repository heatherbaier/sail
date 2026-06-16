# src/sail/core/builders.py

from .registries import ModelRegistry, DatasetRegistry, ExplainerRegistry
from ..data.adapters import SimbaJSONDataset, JSONGeoAdapter, resolve_json_paths
from ..data.temporal_adapters import TemporalGeoAdapter, TemporalSchoolDataset
from ..explain.simba import Simba  # (later: also SIAExplainer, etc.)
from ..training.trainer import Trainer, TrainConfig


import os
import re
import torch
from pathlib import Path



def highest_epoch(dir_path=".", max_epoch=None):
    pat = re.compile(r"^model_epoch(\d+)\.torch$")
    best = max(
        (
            (int(m.group(1)), p)
            for p in Path(dir_path).iterdir()
            if (m := pat.match(p.name))
            and (max_epoch is None or int(m.group(1)) <= max_epoch)
        ),
        default=(None, None),
    )
    return best  # (epoch_number, Path)



def build_validation_dataset(cfg_dataset: dict, cfg: dict, ckpt_dir: str, temporal: bool = False):

    from ..data.adapters import JSONGeoAdapter, resolve_json_paths

    ys_path, coords_path, dup_path = resolve_json_paths(
        data_root=cfg_dataset["data_root"],
        prefix=cfg_dataset["prefix"],
        with_neighbors=False,   # explicitly off
    )


    if temporal:
        print("TEMPORAL")
        ds = TemporalSchoolDataset(
            json_path = ys_path,
            img_dir = ckpt_dir,
            validate = True,
        )
        


    else:
        print("NOT TEMPORAL")
        ds = SimbaJSONDataset(
            root_dir = cfg["dataset"]["data_root"],
            ys_path = ys_path,
            coords_path = coords_path,
            dup_path = dup_path,
            ckpt_dir = ckpt_dir,
            validate = True,
            new = cfg_dataset.get("new", False),
        )

    return ds



def build_temporal_dataset(cfg_dataset: dict, cfg: dict, batch_size: int, ckpt_dir: str):

    print(cfg)

    ys_path, coords_path, _ = resolve_json_paths(
        data_root=cfg_dataset["data_root"],
        prefix=cfg_dataset["prefix"],
        with_neighbors=False,   # explicitly off
    )

    ds = TemporalGeoAdapter(
        root_dir=cfg_dataset["data_root"],
        ys_path=ys_path,
        coords_path=coords_path,
        batch_size = batch_size,
        img_size=tuple(cfg_dataset.get("img_size", (256, 256))),
        num_workers=cfg_dataset.get("num_workers", 0),
        ckpt_dir = ckpt_dir
    )

    return ds



    

def build_dataset(cfg_dataset: dict, cfg: dict, batch_size: int, ckpt_dir: str):

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
        batch_size = batch_size,
        img_size=tuple(cfg_dataset.get("img_size", [224, 224])),
        num_workers = cfg_dataset.get("num_workers", 0),
        ckpt_dir = ckpt_dir,
        seed = cfg_dataset.get("seed", 1337),
        write_files = cfg_dataset.get("write_files", True),
    )

    return ds


def build_model(cfg_model: dict, cfg = None, continue_training = False, ckpt_dir = None):

    name = cfg_model["name"]
    params = cfg_model.get("params", {})
    model_wrapper_cls = ModelRegistry.get(name)
    print(params)
    
    if continue_training:
        epoch, path = highest_epoch(ckpt_dir)
        print("continue_training: ", continue_training, path)
        model_wrapper = model_wrapper_cls(**params)
        model_wrapper.load(path)
        model = model_wrapper.net
        start_epoch = epoch + 1
        return_vals = [model_wrapper, model, start_epoch]
    else:
        model_wrapper = model_wrapper_cls(**params)
        model = model_wrapper.build()
        start_epoch = 0

        print(model)

        return_vals = [model_wrapper, model, start_epoch]

    # gafs
    
    return return_vals


def build_trainer(cfg_trainer: dict, model_wrapper, dataset, model_name: str, batch_size: int, ckpt_dir: str, start_epoch = 0):
    train_cfg = TrainConfig(
        epochs = cfg_trainer["epochs"],
        lr = cfg_trainer["lr"],
        ckpt_dir = ckpt_dir,
        device = cfg_trainer.get("device", "cuda"),
        # model_name = model_name
        # save_every=cfg_trainer.get("save_every", 1),
        # eval_every=cfg_trainer.get("eval_every", 1),
    )
    return Trainer(model_wrapper, dataset, train_cfg, model_name, batch_size, start_epoch)

def build_explainer(cfg_exp: dict, net):
    # For now default to Simba. Later you can branch on cfg_exp["method"]
    return Simba(
        model=net,
        ckpt_dir=cfg_exp["ckpt_dir"],
        device="cuda",  # could expose in cfg_exp later
    )
