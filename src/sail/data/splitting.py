"""
Deterministic, dataset-independent train/val/test splitting.

The original split (still the default -- see split_strategy="random" in
JSONGeoAdapter) is `random.Random(seed).shuffle(range(n))` then contiguous
slicing. That's index-based: which bucket an item lands in depends on
shuffling the *whole* dataset's index range, so it depends on `n` and each
item's position in the sorted item list. Build a family of datasets that
share items across runs -- e.g. one dataset per quarter/year, where a
location's chip is sometimes missing that particular quarter -- and the
same seed gives a DIFFERENT split each time: every item after the first
missing one shifts position, so a location can land in train for one
quarter and val for another purely because of which OTHER locations
happened to be present that quarter, not anything about that location
itself. Comparing per-quarter validation metrics (accuracy, bias, ...)
across such datasets is then comparing different populations each time,
which confounds any real trend with which tracts happened to be evaluated.

It's also not spatially aware: a purely random split typically puts
spatially adjacent locations on both sides of the train/val boundary.
Since geospatial data is autocorrelated (neighboring tracts tend to have
similar imagery and similar labels), that inflates validation performance
relative to a genuinely held-out region.

split_strategy="stable" here fixes both with the same mechanism: make each
item's bucket a pure, seeded hash of the item's own identity (or, for
spatial mode, its location) -- NOT of its position in whatever dataset it
happens to be part of.
  - Stability across differently-composed datasets falls straight out of
    that: the same key always hashes to the same bucket, whether or not
    any other key is present alongside it.
  - Spatial awareness is the same mechanism with a coarser key: hash a
    lat/lon grid cell instead of the item's own id, so every location in
    that cell lands in the same bucket together instead of being
    scattered independently.
"""
from __future__ import annotations

import hashlib
import math
import os
from typing import Dict, Optional, Sequence, Tuple

Split = Tuple[float, float, float]


def stable_unit_interval(key: str, seed: int) -> float:
    """
    Deterministic float in [0, 1) from a string key + seed. Stable across
    Python processes/machines (unlike the builtin hash(), which is
    randomized per-process unless PYTHONHASHSEED is pinned) and, crucially,
    independent of anything except `key` and `seed` -- no dependence on how
    many other keys exist or what order they're seen in.
    """
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def assign_bucket(u: float, split: Split) -> str:
    train_frac, val_frac, _test_frac = split
    if u < train_frac:
        return "train"
    if u < train_frac + val_frac:
        return "val"
    return "test"


def spatial_block_id(lon: float, lat: float, block_deg: float) -> str:
    """
    Coarse grid-cell id for spatial blocking: every location within the
    same block_deg x block_deg cell gets the same id, so they're all
    assigned to the same split together instead of being scattered across
    train/val/test independently. Degrees, not meters -- fine at census
    tract scale for a single state/region; pick block_deg meaningfully
    larger than typical inter-tract spacing (start around 0.1-0.3 for
    census tracts and tune by checking how many distinct blocks you get --
    too small approaches the unblocked per-item split, too large starves
    val/test of any items in some regions).
    """
    return f"{math.floor(lon / block_deg)}_{math.floor(lat / block_deg)}"


def item_key(path_or_name: str) -> str:
    """
    Normalize an item to a stable split key: the filename stem, not the
    full path. In this project chips are named "{GEOID}.tif" (see
    uswealth-geoai's compute_pixel_diversity.py docstring for the same
    convention), but different quarters/years point `data_root` at
    different directories, so the same location's full item path differs
    by dataset even though the location itself doesn't. Hashing the full
    path would silently break the "same location, same split, every
    dataset" guarantee this module exists for -- use the basename stem
    instead, which is dataset-independent.
    """
    return os.path.splitext(os.path.basename(path_or_name))[0]


def compute_stable_split(
    items: Sequence[str],
    coords: Optional[Dict[str, Tuple[float, float]]],
    seed: int,
    split: Split = (0.8, 0.1, 0.1),
    spatial_block_deg: Optional[float] = None,
) -> Dict[str, str]:
    """
    Return {item: "train"|"val"|"test"}. Deterministic given (items,
    coords, seed, split, spatial_block_deg), but -- the whole point --
    each item's own bucket does not depend on which OTHER items are
    present. Build this dataset again later with a different (sub)set of
    items (e.g. a different quarter with a few tracts missing) and every
    item present in both calls gets the same bucket both times.

    spatial_block_deg=None: split by each item's own identity (still
    dataset-independent/stable, just not spatially blocked -- items right
    next to each other can still land in different splits).
    spatial_block_deg=<float>: split by spatial block instead (see
    spatial_block_id), so whole neighborhoods of nearby items land in the
    same split together -- use this to avoid spatial-autocorrelation
    leakage between train and val.
    """
    if abs(sum(split) - 1.0) > 1e-6:
        raise ValueError(f"split must sum to 1.0, got {split} (sums to {sum(split)})")

    out: Dict[str, str] = {}
    for it in items:
        if spatial_block_deg is not None:
            if coords is None or it not in coords:
                raise ValueError(
                    f"spatial_block_deg is set but no coords entry for {it!r} -- "
                    f"spatial splitting needs coordinates for every item."
                )
            lon, lat = coords[it]
            key = spatial_block_id(float(lon), float(lat), spatial_block_deg)
        else:
            key = item_key(it)
        out[it] = assign_bucket(stable_unit_interval(key, seed), split)
    return out
