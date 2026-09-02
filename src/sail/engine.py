# src/sail/engine.py

import yaml
import os
import torch
import re
from pathlib import Path
from .core.builders import build_dataset, build_temporal_dataset, build_validation_dataset, build_model, build_trainer, build_explainer
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

    output_dir = os.path.join(cfg["output_dir"], cfg["experiment_name"])
    os.mkdir(output_dir)

    if cfg["model"]["name"] == "geoconv":
        print("geoconv!!")
        bs = 1
    else:
        bs = cfg["dataset"]["batch_size"]

    if cfg["dataset"]["temporal"]:
        ds = build_temporal_dataset(cfg["dataset"], cfg, bs, output_dir)
    else:
        ds = build_dataset(cfg["dataset"], cfg, bs, output_dir)
        
    model_wrapper, net, _ = build_model(cfg["model"])

    trainer = build_trainer(
        cfg_trainer = cfg["trainer"],
        model_wrapper = model_wrapper,
        dataset = ds,
        model_name = cfg["model"]["name"],
        batch_size = cfg["dataset"]["batch_size"],
        ckpt_dir = output_dir
    )
    trainer.fit()  # you already have Trainer.fit() in your CLI train path. :contentReference[oaicite:8]{index=8}

    return ds, model_wrapper, net  # return for downstream steps




def continue_training(cfg):

    output_dir = os.path.join(cfg["output_dir"], cfg["experiment_name"])

    # set batch size correctly
    if cfg["model"]["name"] == "geoconv":
        print("geoconv!!")
        bs = 1
    else:
        bs = cfg["dataset"]["batch_size"]

    # actually build the dataset
    if cfg["dataset"]["temporal"]:
        ds = build_temporal_dataset(cfg["dataset"], cfg, bs)
    else:
        ds = build_dataset(cfg["dataset"], cfg, bs)
        
    model_wrapper, net, start_epoch = build_model(cfg["model"], cfg = cfg, continue_training = True, ckpt_dir = output_dir)

    # print(net)

    trainer = build_trainer(
        cfg_trainer = cfg["trainer"],
        model_wrapper = model_wrapper,
        dataset = ds,
        model_name = cfg["model"]["name"],
        batch_size = cfg["dataset"]["batch_size"],
        ckpt_dir = output_dir,
        start_epoch = start_epoch
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


def unpack_outputs(out):
    """
    Standardizes model output into (pred, extras) form.
    pred is always a tensor.
    extras can be None, a tensor, tuple, dict, etc.
    """

    # Case 1: dict
    if isinstance(out, dict):
        pred = out["pred"]
        extras = {k: v for k, v in out.items() if k != "pred"}
        return pred, extras if extras else None

    # Case 2: tuple or list
    elif isinstance(out, (tuple, list)):
        pred = out[0]
        extras = out[1:] if len(out) > 1 else None
        return pred, extras

    # Case 3: just a tensor
    else:
        return out, None



def run_validation(cfg):
    
    ckpt_dir = os.path.join(cfg["output_dir"], cfg["experiment_name"])
    device = cfg["validator"]["device"]
    
    # ds = build_validation_dataset(cfg["dataset"], cfg)

    if cfg["dataset"]["temporal"]:
        ds = build_validation_dataset(cfg["dataset"], cfg, ckpt_dir, temporal = True)
        # ds = ds.test_loader()
    else:
        ds = build_validation_dataset(cfg["dataset"], cfg, ckpt_dir, temporal = False)


    if cfg["dataset"]["new"]:
        append = "full"
    else:
        append = "valset"
    
    
    model_wrapper, net, _ = build_model(cfg["model"])
    epoch, path = highest_epoch(ckpt_dir)
    print(epoch, path)
    model_wrapper.load(path)
    model_wrapper.net = model_wrapper.net.to(device).eval()
    
    imnames, preds, labels, all_extras = [], [], [], []
    for c, batch in tqdm.tqdm(enumerate(ds), desc = "Validating"):
        batch = {k: (v.to(device).unsqueeze(0) if hasattr(v, "to") else v) for k,v in batch.items()}
        out = model_wrapper.forward(batch)
        pred, extras = unpack_outputs(out)
        # print(pred, extras[0][0])
        # jdkjaga
        imnames.append(batch["image_name"])
        preds.append(pred.item())
        labels.append(batch["label"].item())

        if extras[0] is not None:
            all_extras.append(extras[0].detach().cpu().numpy().flatten())
        else:
            all_extras.append(0)

        # print("EXTRAS: ", extras)

        if c % 10:
            df = pd.DataFrame()
            df["name"], df["pred"], df["label"], df["extra"] = imnames, preds, labels, all_extras
            
            # if extras[0] is not None:
                # df["extra"] = df["extra"].apply(lambda x: x.detach().cpu().numpy().flatten())
                
            df.to_csv(os.path.join(ckpt_dir, f"epoch{epoch}_{append}_preds.csv"))  

    df = pd.DataFrame()
    df["name"], df["pred"], df["label"], df["extra"] = imnames, preds, labels, all_extras
    
    # if extras[0] is not None:
        # df["extra"] = df["extra"].apply(lambda x: x.detach().cpu().numpy().flatten())
    
    df.to_csv(os.path.join(ckpt_dir, f"epoch{epoch}_{append}_preds.csv"))  

    # metrics = loops.validate_loop(
    #     model_wrapper,
    #     ds,                   # or ds.val_loader()
    #     device=cfg["trainer"].get("device", "cuda"),
    #     metrics_to_compute=cfg["validate"].get("metrics", []),
    #     ckpt_dir=ckpt_dir,
    # )

    # save_metrics_csv(metrics, os.path.join(ckpt_dir, "val_metrics.csv"))
    # return metrics


def run_band_importance(cfg):
    """
    task: band_importance -- band attribution for an already-trained
    checkpoint (see explain/band_importance.py for the methods: global
    permutation importance per band and per named group, plus optional
    per-tract ablation importance). Reuses the same dataset/model/
    output_dir/experiment_name/validator config fields as `task:
    validate`, plus an optional `band_importance:` section (n_repeats,
    seed, band_names, groups, per_image).
    """
    from .explain.band_importance import (
        compute_band_importance, compute_group_importance, compute_per_image_importance,
    )

    ckpt_dir = os.path.join(cfg["output_dir"], cfg["experiment_name"])
    device = cfg["validator"]["device"]

    if cfg["dataset"]["temporal"]:
        raise NotImplementedError("Band importance is not implemented for temporal datasets.")
    ds = build_validation_dataset(cfg["dataset"], cfg, ckpt_dir, temporal=False)

    model_wrapper, net, _ = build_model(cfg["model"])
    epoch, path = highest_epoch(ckpt_dir)
    if path is None:
        raise FileNotFoundError(
            f"No model_epoch*.torch checkpoint found in {ckpt_dir} -- "
            f"band importance needs a trained model to evaluate."
        )
    print(f"Loading checkpoint: epoch {epoch}, {path}")
    model_wrapper.load(path)
    model_wrapper.net = model_wrapper.net.to(device).eval()

    bi_cfg = cfg.get("band_importance", {})
    band_names = bi_cfg.get("band_names")
    groups = bi_cfg.get("groups")
    n_repeats = bi_cfg.get("n_repeats", 5)
    seed = bi_cfg.get("seed", 1337)
    batch_size = cfg["dataset"].get("batch_size", 32)

    print("\n=== Per-band global permutation importance ===")
    df = compute_band_importance(
        model_wrapper=model_wrapper, dataset=ds, device=device,
        band_names=band_names, n_repeats=n_repeats, seed=seed, batch_size=batch_size,
    )
    out_path = os.path.join(ckpt_dir, "band_importance.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved {out_path}")

    if groups:
        print("\n=== Per-group global permutation importance ===")
        df_grouped = compute_group_importance(
            model_wrapper=model_wrapper, dataset=ds, device=device, groups=groups,
            band_names=band_names, n_repeats=n_repeats, seed=seed, batch_size=batch_size,
        )
        grouped_path = os.path.join(ckpt_dir, "band_importance_grouped.csv")
        df_grouped.to_csv(grouped_path, index=False)
        print(f"Saved {grouped_path}")

    if bi_cfg.get("per_image"):
        print("\n=== Per-tract ablation importance ===")
        df_per_image = compute_per_image_importance(
            model_wrapper=model_wrapper, dataset=ds, device=device,
            band_names=band_names, groups=groups, batch_size=batch_size,
        )
        per_image_path = os.path.join(ckpt_dir, "band_importance_per_image.csv")
        df_per_image.to_csv(per_image_path, index=False)
        print(f"Saved {per_image_path}")

    return df


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
    if task == "continue_training":
        ds, mw, net = continue_training(cfg)
    elif task == "validate":
        # ds, mw, net = run_training(cfg)  # or load ckpt
        run_validation(cfg)
    elif task == "band_importance":
        run_band_importance(cfg)
    elif task == "explain":
        ds, mw, net = run_training(cfg)  # or load ckpt
        run_explain(cfg, ds, net)
    else:
        raise ValueError(f"Unknown task: {task}")

