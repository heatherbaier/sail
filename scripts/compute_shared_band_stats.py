"""
Compute ONE band_mean/band_std, pooled across multiple quarters' chips,
instead of the per-quarter auto-computed stats SimbaJSONDataset normally
falls back to (see adapters.py::_compute_tiff_band_stats /
_resolve_or_compute_band_stats).

Why this exists: per-quarter normalization divides each quarter's chips by
that quarter's OWN std, which forces every quarter's normalized pixel
variance to look the same (std=1) regardless of how different the raw
quarters actually were -- that's standardizing WITHIN a group, which
mathematically removes the BETWEEN-group variance by construction. If part
of what you're testing is whether the model can pick up on absolute-scale
differences across quarters (e.g. a genuinely flatter winter chip vs. a
higher-contrast summer chip), per-quarter normalization erases exactly
that signal before the model ever sees a pixel. Using one shared mean/std
across all quarters keeps normalization's usual training-stability
benefits (inputs still roughly zero-mean/unit-variance overall) while
letting a chip that's actually flatter than the yearly norm come out with
visibly smaller normalized variance than a chip that isn't.

Reuses the exact same accumulation math as adapters.py::_compute_tiff_band_stats
(same nodata convention: 0 = nodata, matches geoetl's fillna(0)), just
pooled across multiple quarters' chip populations instead of one, so the
numbers this produces are apples-to-apples with what per-quarter
auto-compute would have used -- only the population being averaged over
is different.

Usage:
    python compute_shared_band_stats.py \
        --data-roots /data/hbaier/new_data/tlag/az_imagery/q1_2016_s2_allbands/ \
                     /data/hbaier/new_data/tlag/az_imagery/q2_2016_s2_allbands/ \
                     /data/hbaier/new_data/tlag/az_imagery/q3_2016_s2_allbands/ \
                     /data/hbaier/new_data/tlag/az_imagery/q4_2016_s2_allbands/ \
        --prefixes az_2016_q1_s2_allbands az_2016_q2_s2_allbands \
                   az_2016_q3_s2_allbands az_2016_q4_s2_allbands \
        --out ./az_2016_shared_band_stats.json

Then paste the printed band_mean/band_std into EVERY quarter's train
config for that state (dataset.band_mean / dataset.band_std) -- setting
those explicitly makes SimbaJSONDataset skip its own per-quarter
auto-compute and use these instead. Same discipline as seed/split/
split_strategy: it has to be pasted identically into every quarter's
config, or you're back to inconsistent normalization across quarters.
"""

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import rasterio


def load_tiff_paths(data_root: str, prefix: str) -> list:
    """Same item universe SimbaJSONDataset builds: intersection of ys.json
    and coords.json keys, restricted to .tif/.tiff paths."""
    ys_path = os.path.join(data_root, f"{prefix}_ys.json")
    coords_path = os.path.join(data_root, f"{prefix}_coords.json")
    with open(ys_path) as f:
        ys = json.load(f)
    with open(coords_path) as f:
        coords = json.load(f)
    keys = set(ys) & set(coords)
    paths = []
    for k in keys:
        full = k if os.path.isabs(k) else os.path.join(data_root, k)
        if os.path.splitext(full)[1].lower() in (".tif", ".tiff"):
            paths.append(full)
    return paths


def accumulate_band_sums(paths: list, scale_divisor: float, totals: dict):
    """Reads each chip, adds its per-band valid-pixel sum/sum-of-squares/
    count into the running totals dict (mutated in place). Mirrors
    adapters.py::_compute_tiff_band_stats's inner loop exactly."""
    for p in paths:
        with rasterio.open(p) as src:
            arr = src.read().astype("float64") / scale_divisor  # (C, H, W)
        if totals["n_bands"] is None:
            n_bands = arr.shape[0]
            totals["n_bands"] = n_bands
            totals["total"] = np.zeros(n_bands)
            totals["total_sq"] = np.zeros(n_bands)
            totals["count"] = np.zeros(n_bands)
        for b in range(totals["n_bands"]):
            valid = arr[b][arr[b] > 0]  # 0 = nodata, matches geoetl's convention
            totals["total"][b] += valid.sum()
            totals["total_sq"][b] += (valid ** 2).sum()
            totals["count"][b] += valid.size


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-roots", nargs="+", required=True, help="One dataset.data_root per quarter")
    p.add_argument("--prefixes", nargs="+", required=True, help="One dataset.prefix per quarter, same order as --data-roots")
    p.add_argument("--sample-size", type=int, default=200,
                    help="Chips sampled PER QUARTER (not total) -- keeps each quarter equally represented "
                         "in the pooled stats regardless of how many tracts each quarter happens to have "
                         "(default: 200, matches sail's own per-run default)")
    p.add_argument("--scale-divisor", type=float, default=10000.0,
                    help="Matches geoetl's uint16 'reflectance x 10000' convention (default: 10000.0)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out", type=Path, default=Path("./shared_band_stats.json"))
    args = p.parse_args()

    if len(args.data_roots) != len(args.prefixes):
        raise SystemExit(f"--data-roots ({len(args.data_roots)}) and --prefixes "
                          f"({len(args.prefixes)}) must have the same length, one per quarter")

    rng = random.Random(args.seed)
    totals = {"n_bands": None, "total": None, "total_sq": None, "count": None}
    n_sampled_total = 0

    for data_root, prefix in zip(args.data_roots, args.prefixes):
        paths = load_tiff_paths(data_root, prefix)
        sample = paths if len(paths) <= args.sample_size else rng.sample(paths, args.sample_size)
        print(f"[{prefix}] {len(paths)} chips available, sampling {len(sample)}")
        accumulate_band_sums(sample, args.scale_divisor, totals)
        n_sampled_total += len(sample)

    if totals["n_bands"] is None:
        raise SystemExit("No tiff chips found across any of the given quarters -- check --data-roots/--prefixes.")

    count = np.maximum(totals["count"], 1)  # avoid div-by-zero for a degenerate/all-nodata band
    mean = totals["total"] / count
    var = np.maximum(totals["total_sq"] / count - mean ** 2, 1e-8)  # numerical floor
    std = np.sqrt(var)

    print(f"\nPooled across {n_sampled_total} chips, {totals['n_bands']} bands:")
    print(f"  band_mean: {mean.round(6).tolist()}")
    print(f"  band_std:  {std.round(6).tolist()}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({
            "band_mean": mean.tolist(),
            "band_std": std.tolist(),
            "n_sampled_total": n_sampled_total,
            "n_bands": totals["n_bands"],
            "sample_size_per_quarter": args.sample_size,
            "seed": args.seed,
            "scale_divisor": args.scale_divisor,
            "source_prefixes": args.prefixes,
        }, f, indent=2)
    print(f"\nWrote {args.out}")
    print("\nPaste into EVERY quarter's train config for this state:")
    print(f"  band_mean: {mean.round(6).tolist()}")
    print(f"  band_std: {std.round(6).tolist()}")


if __name__ == "__main__":
    main()
