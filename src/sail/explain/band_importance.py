"""
Band attribution: how much does prediction accuracy depend on each input
band (or a named group of bands), both in aggregate and per tract.

Two distinct mechanisms are used here, deliberately different from each
other, because they answer different questions:

  GLOBAL (permutation) -- compute_band_importance / compute_group_importance
      Shuffle a band's (or a group's) values across the validation set,
      breaking that band's per-sample association with the label while
      leaving its marginal distribution intact, and measure the resulting
      MSE increase over several repeats. Answers "how much does the
      dataset-level accuracy depend on this band", and because it's
      randomized and repeated, it comes with a noise estimate
      (delta_mse_std) -- see the correlated-bands and repeat-count
      caveats in the module-level docs/PR for #9.

  PER-IMAGE (ablation to the band's mean) -- compute_per_image_importance
      Permutation's random donor-swap makes a poor per-instance measure:
      which OTHER sample gets swapped in dominates a single tract's delta,
      so it doesn't settle down without many repeats per tract (expensive
      at N-tracts x N-bands). Ablation is a different, deterministic
      mechanism instead -- replace a band's values with its OWN validation-
      set mean (a "neutral"/uninformative value for that band) and measure
      how much THIS tract's own prediction shifts. One forward pass per
      band/group, same interpretation ("what would this prediction do
      without this band's information") but computed per tract rather
      than as one dataset-wide number. This is what feeds urban/rural
      moderation and spatial-autocorrelation tests downstream (see
      analyze_band_importance_spatial.py in uswealth-geoai) -- those need
      one importance value per tract with its own GEOID and coordinates.

Run per quarter's trained checkpoint via `task: band_importance` in a
config -- the same dataset/model/output_dir/experiment_name/validator
fields as a `task: validate` config, plus an optional
`band_importance:` section:

    band_importance:
        n_repeats: 5          # permutations per band/group, averaged for stability
        seed: 1337
        band_names: ["B01", "B02", ..., "B12"]   # must match input channel count
        groups:                                   # optional, for grouped permutation
            vegetation: ["B05", "B06", "B07", "B08", "B8A"]
            swir: ["B11", "B12"]
            visible: ["B02", "B03", "B04"]
        per_image: true       # also run the per-tract ablation test

Writes, under <ckpt_dir>:
    band_importance.csv              per-band global permutation importance
    band_importance_grouped.csv      per-group global permutation importance (if groups given)
    band_importance_per_image.csv    per-tract ablation importance (if per_image: true)
"""

from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def permute_band(images: torch.Tensor, band_idx: Union[int, List[int]], perm: torch.Tensor) -> torch.Tensor:
    """
    Copy of `images` (N,C,H,W) with band(s) `band_idx` shuffled across the
    sample dimension according to `perm`. A single index shuffles one
    band; a list shuffles all of them together with the SAME permutation
    (i.e. as a group -- each sample's donor is the same across every band
    in the group, preserving whatever correlation structure exists
    between them in the donor sample).
    """
    out = images.clone()
    idx = [band_idx] if isinstance(band_idx, int) else list(band_idx)
    for b in idx:
        out[:, b] = images[perm, b]
    return out


@torch.no_grad()
def _predict_all(model_wrapper, images: torch.Tensor, coords: torch.Tensor,
                  device: str, batch_size: int) -> torch.Tensor:
    preds = []
    for start in range(0, images.shape[0], batch_size):
        batch = {
            "image": images[start:start + batch_size].to(device),
            "coords": coords[start:start + batch_size].to(device),
        }
        pred, _ = model_wrapper.forward(batch)
        preds.append(pred.view(-1).cpu())
    return torch.cat(preds)


def _load_full_dataset(dataset, batch_size: int):
    """
    Load every item in `dataset` into memory once: images, coords,
    labels, and image_name (used downstream to recover GEOID -- chip
    filename stem IS the GEOID, see compute_pixel_diversity.py /
    geoetl's pipeline.py in uswealth-geoai for why). A validate-task
    split is typically a few hundred chips, so this is cheap, and both
    the global and per-image tests below need the whole set in memory
    anyway (permutation shuffles across it; ablation compares every
    tract against the same per-band means).
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    images_parts, coords_parts, labels_parts, names_parts = [], [], [], []
    for batch in loader:
        images_parts.append(batch["image"])
        coords_parts.append(batch["coords"])
        labels_parts.append(batch["label"])
        names_parts.append(batch["image_name"])
    images = torch.cat(images_parts)
    coords = torch.cat(coords_parts)
    labels = torch.cat(labels_parts)
    names = [n for part in names_parts for n in part]
    return images, coords, labels, names


def _resolve_band_names(n_bands: int, band_names: Optional[List[str]]) -> List[str]:
    if band_names is None:
        return [f"band{i}" for i in range(n_bands)]
    if len(band_names) != n_bands:
        raise ValueError(
            f"band_importance.band_names has {len(band_names)} entries "
            f"but this dataset's images have {n_bands} channels."
        )
    return list(band_names)


def _resolve_group_indices(groups: Dict[str, List[str]], band_names: List[str]) -> Dict[str, List[int]]:
    name_to_idx = {name: i for i, name in enumerate(band_names)}
    resolved = {}
    for group_name, members in groups.items():
        missing = [m for m in members if m not in name_to_idx]
        if missing:
            raise ValueError(
                f"band_importance.groups['{group_name}'] references bands "
                f"not in band_names: {missing}"
            )
        resolved[group_name] = [name_to_idx[m] for m in members]
    return resolved


# -------------------------------------------------------------------------
# Global (permutation) importance -- per band and per group
# -------------------------------------------------------------------------
def compute_band_importance(
    model_wrapper,
    dataset,
    device: str,
    band_names: Optional[List[str]] = None,
    n_repeats: int = 5,
    seed: int = 1337,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Permutation importance over every individual input band."""
    images, coords, labels, _ = _load_full_dataset(dataset, batch_size)
    labels = labels.to(device)
    n_bands = images.shape[1]
    band_names = _resolve_band_names(n_bands, band_names)

    baseline_preds = _predict_all(model_wrapper, images, coords, device, batch_size).to(device)
    baseline_mse = torch.mean((baseline_preds - labels) ** 2).item()
    print(f"Baseline MSE ({images.shape[0]} samples, {n_bands} bands): {baseline_mse:.6f}")

    rng = np.random.RandomState(seed)
    rows = []
    for b in range(n_bands):
        deltas = []
        for _ in range(n_repeats):
            perm = torch.from_numpy(rng.permutation(images.shape[0]))
            permuted = permute_band(images, b, perm)
            preds = _predict_all(model_wrapper, permuted, coords, device, batch_size).to(device)
            mse = torch.mean((preds - labels) ** 2).item()
            deltas.append(mse - baseline_mse)
        mean_delta = float(np.mean(deltas))
        pct = 100 * mean_delta / baseline_mse if baseline_mse > 0 else float("nan")
        rows.append({
            "band_index": b, "band_name": band_names[b], "baseline_mse": baseline_mse,
            "delta_mse": mean_delta, "delta_mse_std": float(np.std(deltas)), "pct_increase": pct,
        })
        print(f"  {band_names[b]}: delta_mse={mean_delta:+.6f} ({pct:+.2f}%)")

    return pd.DataFrame(rows).sort_values("delta_mse", ascending=False).reset_index(drop=True)


def compute_group_importance(
    model_wrapper,
    dataset,
    device: str,
    groups: Dict[str, List[str]],
    band_names: Optional[List[str]] = None,
    n_repeats: int = 5,
    seed: int = 1337,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Same mechanism as compute_band_importance, but shuffling a whole
    named group of bands together (same permutation across every band in
    the group) rather than one band at a time.

    Why this matters: correlated bands (e.g. the red-edge/NIR cluster
    B05/B06/B07/B08/B8A, which all measure overlapping vegetation-
    chlorophyll signal) can make single-band permutation understate each
    member's importance individually -- the model can partly fall back on
    a correlated band that's still intact. Shuffling the whole cluster at
    once removes that fallback, so a group's importance here can be
    meaningfully larger than the sum of its members' individual
    single-band importances in compute_band_importance's output.
    """
    images, coords, labels, _ = _load_full_dataset(dataset, batch_size)
    labels = labels.to(device)
    n_bands = images.shape[1]
    band_names = _resolve_band_names(n_bands, band_names)
    group_indices = _resolve_group_indices(groups, band_names)

    baseline_preds = _predict_all(model_wrapper, images, coords, device, batch_size).to(device)
    baseline_mse = torch.mean((baseline_preds - labels) ** 2).item()
    print(f"Baseline MSE ({images.shape[0]} samples): {baseline_mse:.6f}")

    rng = np.random.RandomState(seed)
    rows = []
    for group_name, idxs in group_indices.items():
        deltas = []
        for _ in range(n_repeats):
            perm = torch.from_numpy(rng.permutation(images.shape[0]))
            permuted = permute_band(images, idxs, perm)
            preds = _predict_all(model_wrapper, permuted, coords, device, batch_size).to(device)
            mse = torch.mean((preds - labels) ** 2).item()
            deltas.append(mse - baseline_mse)
        mean_delta = float(np.mean(deltas))
        pct = 100 * mean_delta / baseline_mse if baseline_mse > 0 else float("nan")
        rows.append({
            "group": group_name, "members": ",".join(groups[group_name]),
            "n_bands": len(idxs), "baseline_mse": baseline_mse,
            "delta_mse": mean_delta, "delta_mse_std": float(np.std(deltas)), "pct_increase": pct,
        })
        print(f"  [{group_name}] ({groups[group_name]}): delta_mse={mean_delta:+.6f} ({pct:+.2f}%)")

    return pd.DataFrame(rows).sort_values("delta_mse", ascending=False).reset_index(drop=True)


# -------------------------------------------------------------------------
# Per-image (ablation-to-mean) importance -- per band and per group
# -------------------------------------------------------------------------
def compute_per_image_importance(
    model_wrapper,
    dataset,
    device: str,
    band_names: Optional[List[str]] = None,
    groups: Optional[Dict[str, List[str]]] = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Per-tract band/group importance via ablation to the band's own
    validation-set mean -- see the module docstring for why this (not
    permutation) is the right mechanism for a per-instance number.

    One row per tract: GEOID (from the chip filename -- geoetl names
    every chip "{GEOID}.tif", so the filename stem IS the GEOID), lat,
    lon (from the dataset's coords, for the spatial-autocorrelation test
    downstream), label, pred_baseline, then for every band and group:
        delta_pred_<name>    pred_ablated - pred_baseline (signed: which
                              direction does removing this band's
                              information shift THIS tract's prediction)
        delta_sqerr_<name>   (pred_ablated - label)^2 - (pred_baseline - label)^2
                              (signed: does removing it make this one
                              tract's prediction better [negative] or
                              worse [positive])
    """
    images, coords, labels, names = _load_full_dataset(dataset, batch_size)
    labels_dev = labels.to(device)
    n_bands = images.shape[1]
    band_names = _resolve_band_names(n_bands, band_names)
    group_indices = _resolve_group_indices(groups, band_names) if groups else {}

    from pathlib import Path
    geoids = [Path(n).stem for n in names]
    lat = coords[:, 0].numpy()
    lon = coords[:, 1].numpy()

    baseline_preds = _predict_all(model_wrapper, images, coords, device, batch_size)
    baseline_sqerr = (baseline_preds - labels) ** 2

    out = pd.DataFrame({
        "GEOID": geoids, "lat": lat, "lon": lon,
        "label": labels.numpy(), "pred_baseline": baseline_preds.numpy(),
    })

    band_means = images.mean(dim=0, keepdim=True)  # (1, C, H, W), per-band spatial mean

    def _ablate_and_score(idxs: List[int], col_suffix: str):
        ablated = images.clone()
        for b in idxs:
            ablated[:, b] = band_means[:, b]
        preds = _predict_all(model_wrapper, ablated, coords, device, batch_size)
        sqerr = (preds - labels) ** 2
        out[f"delta_pred_{col_suffix}"] = (preds - baseline_preds).numpy()
        out[f"delta_sqerr_{col_suffix}"] = (sqerr - baseline_sqerr).numpy()

    for b, name in enumerate(band_names):
        _ablate_and_score([b], name)
        print(f"  ablated {name}")
    for group_name, idxs in group_indices.items():
        _ablate_and_score(idxs, f"grp_{group_name}")
        print(f"  ablated group [{group_name}]")

    return out
