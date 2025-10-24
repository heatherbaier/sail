# src/sail/engine.py

import yaml
import os
import torch
from .core.builders import build_dataset, build_model, build_trainer, build_explainer
from .utils.io import save_metrics_csv  # imaginary helper you’ll add
from .training import loops            # validation loops etc.

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
    ds = build_dataset(cfg["dataset"])
    model_wrapper, net = build_model(cfg["model"])

    trainer = build_trainer(
        cfg_trainer=cfg["trainer"],
        model_wrapper=model_wrapper,
        dataset=ds,
        model_name=cfg["model"]["name"],
        batch_size=cfg["dataset"]["batch_size"],
    )
    trainer.fit()  # you already have Trainer.fit() in your CLI train path. :contentReference[oaicite:8]{index=8}
    return ds, model_wrapper, net  # return for downstream steps

def run_validation(cfg, ds, model_wrapper):
    if not cfg.get("validate", {}).get("enabled", False):
        return None
    ckpt_dir = cfg["validate"].get("ckpt_dir", cfg["output_dir"])

    # load best / latest checkpoint
    model_wrapper.load(/* path to best/last ckpt in ckpt_dir */)

    metrics = loops.validate_loop(
        model_wrapper,
        ds,                   # or ds.val_loader()
        device=cfg["trainer"].get("device", "cuda"),
        metrics_to_compute=cfg["validate"].get("metrics", []),
        ckpt_dir=ckpt_dir,
    )

    save_metrics_csv(metrics, os.path.join(ckpt_dir, "val_metrics.csv"))
    return metrics

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
        ds, mw, net = run_training(cfg)  # or load ckpt
        run_validation(cfg, ds, mw)
    elif task == "explain":
        ds, mw, net = run_training(cfg)  # or load ckpt
        run_explain(cfg, ds, net)
    else:
        raise ValueError(f"Unknown task: {task}")

