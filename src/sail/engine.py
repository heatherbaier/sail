# src/sail/engine.py

import yaml
import os
import torch
import re
from pathlib import Path
from .core.builders import build_dataset, build_validation_dataset, build_model, build_trainer, build_explainer
from .utils.io import save_metrics_csv  # imaginary helper you’ll add
# from .training import loops            # validation loops etc.

import typer
import tqdm
import pandas as pd

def load_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    # simple interpolation for things like "${output_dir}"
    def _resolve(obj):
        if isinstance(obj, dict):
            return {k: _resolve(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_resolve(v) for v in obj]
        if isinstance(obj, str) and "${output_dir}" in obj:
            return obj.replace("${output_dir}", cfg["output_dir"])
        return obj
    return _resolve(cfg)

def run_training(cfg):
    ds = build_dataset(cfg["dataset"], cfg)
    model_wrapper, net = build_model(cfg["model"])

    # print(net)

    trainer = build_trainer(
        cfg_trainer=cfg["trainer"],
        model_wrapper=model_wrapper,
        dataset=ds,
        model_name=cfg["model"]["name"],
        batch_size=cfg["dataset"]["batch_size"],
        ckpt_dir = cfg["output_dir"]
    )
    trainer.fit()  # you already have Trainer.fit() in your CLI train path. :contentReference[oaicite:8]{index=8}
    return ds, model_wrapper, net  # return for downstream steps


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


def run_validation(cfg):
    


    ds = build_validation_dataset(cfg["dataset"], cfg)
    
    ckpt_dir = cfg["output_dir"]#.get("ckpt_dir", cfg["output_dir"])

    # load best / latest checkpoint
    # model_wrapper.load(/* path to best/last ckpt in ckpt_dir */)

    model_wrapper, net = build_model(cfg["model"])


    # mw = ModelRegistry.get(model)()#.build()
    epoch, path = highest_epoch(ckpt_dir)
    print(epoch, path)
    model_wrapper.load(path)

    device = cfg["validator"]["device"]

    model_wrapper.net = model_wrapper.net.to(device).eval()
    

    imnames, preds, labels = [], [], []
    for c, batch in tqdm.tqdm(enumerate(ds), desc = "Validating"):

        batch = {k: (v.to(device).unsqueeze(0) if hasattr(v, "to") else v) for k,v in batch.items()}

        out = model_wrapper.forward(batch)
        
        # print(, batch["label"])

        imnames.append(batch["image_name"])
        preds.append(out.item())
        labels.append(batch["label"].item())

        # print(imnames, preds, labels)
        
        
        if c % 10:
            df = pd.DataFrame()
            df["name"], df["pred"], df["label"] = imnames, preds, labels
            df.to_csv(os.path.join(ckpt_dir, f"epoch{epoch}_preds.csv"))

        df = pd.DataFrame()
        df["name"], df["pred"], df["label"] = imnames, preds, labels
        df.to_csv(os.path.join(ckpt_dir, f"epoch{epoch}_preds.csv"))  

    


    # gdfgsg

    

    # metrics = loops.validate_loop(
    #     model_wrapper,
    #     ds,                   # or ds.val_loader()
    #     device=cfg["trainer"].get("device", "cuda"),
    #     metrics_to_compute=cfg["validate"].get("metrics", []),
    #     ckpt_dir=ckpt_dir,
    # )

    # save_metrics_csv(metrics, os.path.join(ckpt_dir, "val_metrics.csv"))
    # return metrics

def run_explain(cfg, ds, net):
    exp_cfg = cfg.get("explain", {})
    if not exp_cfg.get("enabled", False):
        return None

    explainer = build_explainer(exp_cfg, net)
    distances = exp_cfg["distances_km"]
    max_instances = exp_cfg.get("max_instances", None)

    # global SIMBA across dataset
    explainer.explain_global_from_list(
        ds,
        distances_km=distances,
        max_instances=max_instances,
    )
    # explainer already writes plots/CSVs in ckpt_dir. :contentReference[oaicite:9]{index=9}

def run(cfg_path):
    cfg = load_config(cfg_path)
    task = cfg.get("task", "train")

    if task == "train":
        ds, mw, net = run_training(cfg)
    elif task == "validate":
        # ds, mw, net = run_training(cfg)  # or load ckpt
        run_validation(cfg)
    elif task == "explain":
        ds, mw, net = run_training(cfg)  # or load ckpt
        run_explain(cfg, ds, net)
    else:
        raise ValueError(f"Unknown task: {task}")

