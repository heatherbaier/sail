"""
Permutation-importance band attribution.

For a trained model, how much does prediction accuracy degrade if one
input band's values are shuffled across the validation set? Shuffling
breaks that band's per-sample association with the label while leaving
its marginal distribution -- and every other band -- untouched, so the
resulting MSE increase is attributable to that band specifically.

Run per quarter's trained checkpoint via `task: band_importance` in a
config (see engine.run_band_importance) -- the same dataset/model/
output_dir/experiment_name/validator fields as a `task: validate`
config, plus an optional `band_importance:` section:

    band_importance:
        n_repeats: 5          # permutations per band, averaged for stability
        seed: 1337
        band_names: ["B01", "B02", "B03", "B04", "B05", "B06", "B07",
                     "B08", "B8A", "B09", "B11", "B12"]   # must match input channel count

Writes <ckpt_dir>/band_importance.csv with one row per band:
    band_index, band_name, baseline_mse, delta_mse, delta_mse_std, pct_increase

Compare pct_increase (100 * delta_mse / baseline_mse) ACROSS quarters,
not raw delta_mse -- a quarter that's just harder overall (higher
baseline_mse) will show bigger raw deltas for every band regardless of
which bands actually matter more, so pct_increase is what isolates a
band's relative contribution from a quarter's overall difficulty.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def permute_band(images: torch.Tensor, band_idx: int, perm: torch.Tensor) -> torch.Tensor:
    """
    Copy of `images` (N,C,H,W) with band `band_idx` shuffled across the
    sample dimension according to `perm`.
    """
    out = images.clone()
    out[:, band_idx] = images[perm, band_idx]
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


def compute_band_importance(
    model_wrapper,
    dataset,
    device: str,
    band_names: Optional[List[str]] = None,
    n_repeats: int = 5,
    seed: int = 1337,
    batch_size: int = 32,
) -> pd.DataFrame:
    """
    Permutation importance over every input band of `dataset`'s images.

    Loads the full dataset into memory once (a validate-task split is
    typically a few hundred chips, so this is cheap) rather than
    streaming per-batch, since permutation shuffles a band's values
    across the WHOLE sample set, not within one batch.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    images_parts, coords_parts, labels_parts = [], [], []
    for batch in loader:
        images_parts.append(batch["image"])
        coords_parts.append(batch["coords"])
        labels_parts.append(batch["label"])
    images = torch.cat(images_parts)
    coords = torch.cat(coords_parts)
    labels = torch.cat(labels_parts).to(device)

    n_bands = images.shape[1]
    if band_names is None:
        band_names = [f"band{i}" for i in range(n_bands)]
    elif len(band_names) != n_bands:
        raise ValueError(
            f"band_importance.band_names has {len(band_names)} entries "
            f"but this dataset's images have {n_bands} channels."
        )

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
            "band_index": b,
            "band_name": band_names[b],
            "baseline_mse": baseline_mse,
            "delta_mse": mean_delta,
            "delta_mse_std": float(np.std(deltas)),
            "pct_increase": pct,
        })
        print(f"  {band_names[b]}: delta_mse={mean_delta:+.6f} ({pct:+.2f}%)")

    return pd.DataFrame(rows).sort_values("delta_mse", ascending=False).reset_index(drop=True)
