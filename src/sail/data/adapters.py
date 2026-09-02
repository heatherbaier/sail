from __future__ import annotations
import os, json, random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import math
import re

from ..core.base_dataset import BaseDatasetAdapter


# ---------- helpers ----------

# _SUFFIX_RE = re.compile(r"^(?P<root>.+)_(?P<idx>\d+)(?P<ext>\.[^.]+)$")

# import os, re
# from typing import Dict, List, Tuple, Optional

_SUFFIX_RE = re.compile(r"^(?P<root>.+)_(?P<idx>\d+)(?P<ext>\.[^.]+)$")




# Add near the top with the other helpers
import os, re
_SUFFIX_RE = re.compile(r"^(?P<root>.+)_(?P<idx>\d+)(?P<ext>\.[^.]+)$")

def _basename(p: str) -> str:
    return os.path.basename(p)

def _split_suffix_basename(path: str):
    base = os.path.basename(path)
    m = _SUFFIX_RE.match(base)
    if m:
        return m.group("root"), int(m.group("idx")), m.group("ext")
    root, ext = os.path.splitext(base)
    return root, None, ext

def _build_dups_from_coords(root_dir: str, ys_keys, coords_keys):
    """
    Synthesize base->neighbors from coords list by grouping clusterid_*.tiff.
    Picks base as clusterid_1.ext if present, else clusterid.ext.
    Returns dict[full_base_path] = [full_neighbor_paths...]
    """
    # Map basename -> full path for all coords keys
    by_base = {_basename(k): k for k in coords_keys}
    # Group neighbors by (root, ext)
    groups = {}
    for full in coords_keys:
        root, idx, ext = _split_suffix_basename(full)
        groups.setdefault((root, ext), []).append(full)

    out = {}
    ys_set = set(ys_keys)
    for (root, ext), paths in groups.items():
        paths.sort()  # ensures _1,_2... order
        cand1 = f"{root}_1{ext}"
        cand2 = f"{root}{ext}"
        base_full = by_base.get(cand1) or by_base.get(cand2)
        if not base_full:
            continue
        # Optionally restrict to bases that exist in ys
        if base_full not in ys_set:
            # if ys uses clusterid.ext as base, try that
            base_no_idx = by_base.get(f"{root}{ext}")
            if base_no_idx and base_no_idx in ys_set:
                base_full = base_no_idx
            else:
                continue
        out[base_full] = paths
    return out


# def _split_suffix_basename(path: str) -> Tuple[str, Optional[int], str]:
#     base = os.path.basename(path)
#     m = _SUFFIX_RE.match(base)
#     if m:
#         return m.group("root"), int(m.group("idx")), m.group("ext")
#     root, ext = os.path.splitext(base)
#     return root, None, ext

def _abs_or_join(root_dir: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(root_dir, p)

def _build_dups_index_by_basename(
    root_dir: str,
    ys_keys: List[str],
    coords_keys: List[str],
    dups_raw: Optional[Dict[str, object]],
) -> Optional[Dict[str, List[str]]]:
    if dups_raw is None:
        return None

    # Map basename -> full path for bases that exist in BOTH ys and coords
    yc = set(ys_keys) & set(coords_keys)
    base_by_name: Dict[str, str] = {}
    for full in yc:
        base_by_name[os.path.basename(full)] = full

    # If dup is already "base -> list/int", normalize to full paths and return
    sample_keys = list(dups_raw.keys())
    case_a = any(k in base_by_name for k in map(os.path.basename, sample_keys))
    if case_a:
        out: Dict[str, List[str]] = {}
        for base_key, entry in dups_raw.items():
            base_name = os.path.basename(base_key)
            if base_name not in base_by_name:
                # allow clusterid_1 fallback → clusterid
                root, idx, ext = _split_suffix_basename(base_key)
                cand1 = f"{root}{ext}"
                cand2 = f"{root}_1{ext}"
                base_full = base_by_name.get(cand1) or base_by_name.get(cand2)
                if base_full is None:
                    continue
            else:
                base_full = base_by_name[base_name]

            if isinstance(entry, list):
                neigh_full = [_abs_or_join(root_dir, p) for p in entry]
            elif isinstance(entry, int):
                root, _, ext = _split_suffix_basename(base_key)
                stem = os.path.splitext(os.path.basename(base_key))[0]
                # if base was 'clusterid.tiff', stem should be 'clusterid'
                if stem.endswith("_1"):
                    stem = stem[:-2]
                neigh_full = [
                    _abs_or_join(root_dir, f"{stem}_{i}{ext}") for i in range(1, entry+1)
                ]
            else:
                continue
            out[base_full] = neigh_full
        return out

    # Case B: dup JSON uses neighbor filenames as keys (e.g., clusterid_7.tiff)
    grouped: Dict[Tuple[str,str], List[str]] = {}
    for neigh_key in dups_raw.keys():
        root, _, ext = _split_suffix_basename(neigh_key)
        grouped.setdefault((root, ext), []).append(neigh_key)

    out: Dict[str, List[str]] = {}
    for (root, ext), neigh_list in grouped.items():
        neigh_list.sort()
        # prefer base 'root.ext', else 'root_1.ext'
        cand1 = f"{root}{ext}"
        cand2 = f"{root}_1{ext}"
        base_full = base_by_name.get(cand1) or base_by_name.get(cand2)
        if base_full is None:
            continue
        out[base_full] = [_abs_or_join(root_dir, n) for n in neigh_list]
    return out

    
    
def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)

def _ensure_rgb(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


# ---------- multiband TIFF support ----------
#
# PIL images can't represent more than ~4 bands (its modes top out around
# RGBA/CMYK), so multiband GeoTIFF chips (e.g. from geoetl's MPC pipeline,
# which can carry NIR/SWIR/red-edge alongside RGB) can't go through the
# PIL-based `_ensure_rgb` + torchvision v1 `transforms.Compose` pipeline
# below at all -- that pipeline stays exactly as-is and is only used for
# non-tiff images (PNG/JPEG from older 3-band projects), so existing
# configs/checkpoints keep behaving identically. TIFFs get a parallel,
# tensor-native pipeline instead.

_TIFF_EXTS = {".tif", ".tiff"}


def _is_tiff(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in _TIFF_EXTS


def _load_tiff_tensor(path: str, scale_divisor: float) -> torch.Tensor:
    """
    Read a (possibly multiband) GeoTIFF chip into a (C,H,W) float32 tensor.

    geoetl's MPC pipeline writes chips as uint16 "reflectance x 10000"
    (see geoetl/io/mpc.py _build_composite, scale_divisor default matches
    that convention) -- dividing back out puts pixel values on an
    approximate [0,1] reflectance scale, the same numeric range ToTensor()
    produces for 8-bit PNG/JPEG chips in the legacy path below.
    """
    with rasterio.open(path) as src:
        arr = src.read().astype("float32")  # (C, H, W)
    return torch.from_numpy(arr) / scale_divisor


def _compute_tiff_band_stats(
    paths: List[str],
    scale_divisor: float,
    sample_size: int = 200,
    seed: int = 1337,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Auto-compute per-band normalization stats from a random sample of
    TIFF chips, instead of requiring them precomputed offline. Cheap next
    to an actual training run (a few hundred small chip reads, once) --
    see _resolve_or_compute_band_stats for how this gets shared/cached
    across a dataset's train/val/test splits instead of run per-split.

    Returns (None, None) if `paths` is empty (e.g. a pure-PNG dataset).
    """
    if not paths:
        return None, None

    rng = random.Random(seed)
    sample = paths if len(paths) <= sample_size else rng.sample(paths, sample_size)

    n_bands = None
    total = total_sq = count = None  # per-band accumulators

    for p in sample:
        with rasterio.open(p) as src:
            arr = src.read().astype("float64") / scale_divisor  # (C, H, W)
        if n_bands is None:
            n_bands = arr.shape[0]
            total = np.zeros(n_bands)
            total_sq = np.zeros(n_bands)
            count = np.zeros(n_bands)
        for b in range(n_bands):
            valid = arr[b][arr[b] > 0]  # 0 = nodata, matches geoetl's convention
            total[b] += valid.sum()
            total_sq[b] += (valid ** 2).sum()
            count[b] += valid.size

    count = np.maximum(count, 1)  # avoid div-by-zero for a degenerate/all-nodata band
    mean = total / count
    var = np.maximum(total_sq / count - mean ** 2, 1e-8)  # numerical floor
    std = np.sqrt(var)
    print(f"Computed TIFF band stats from {len(sample)} sampled chips "
          f"({n_bands} bands): mean={mean.round(4).tolist()} std={std.round(4).tolist()}")
    return mean.tolist(), std.tolist()


def _resolve_or_compute_band_stats(
    tiff_paths: List[str],
    scale_divisor: float,
    ckpt_dir: Optional[str],
    sample_size: int = 200,
    seed: int = 1337,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Return (band_mean, band_std) for a TIFF dataset, computing them only
    once and reusing that value everywhere after -- critical because
    JSONGeoAdapter builds separate SimbaJSONDataset instances for
    train/val/test (and a `full` one to compute the split itself); each
    independently auto-computing stats from its own item subset would
    give the model slightly different normalization for train vs. val,
    which would be a real (silent) bug. Whichever of those instances gets
    constructed first computes and caches to
    <ckpt_dir>/band_stats.json; the rest just load that file.

    This also makes evaluation reuse the exact stats a checkpoint was
    trained with (rather than recomputing from whatever data you're now
    evaluating on) whenever ckpt_dir already has a cached file -- which is
    what you want even when validating on a different dataset than the
    one trained on, since the model's weights were tuned for that specific
    input distribution.
    """
    cache_path = os.path.join(ckpt_dir, "band_stats.json") if ckpt_dir else None
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"Loaded cached TIFF band stats from {cache_path}")
        return cached["band_mean"], cached["band_std"]

    mean, std = _compute_tiff_band_stats(tiff_paths, scale_divisor, sample_size=sample_size, seed=seed)

    if cache_path and mean is not None:
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"band_mean": mean, "band_std": std,
                       "n_sampled": min(sample_size, len(tiff_paths))}, f, indent=2)
        print(f"Cached TIFF band stats to {cache_path}")

    return mean, std


# ---------- auto-computed PNG/JPEG normalization stats ----------
#
# Deliberately separate functions/cache file from the TIFF ones above,
# rather than refactored to share code with them, even though the
# accumulation logic is nearly identical: an existing training run's
# cached <ckpt_dir>/band_stats.json (TIFF) is depended on as-is by
# _resolve_or_compute_band_stats's exact schema, and this shouldn't risk
# changing that.

def _load_png_array(path: str) -> np.ndarray:
    """
    PIL-decode a PNG/JPEG chip into a (3,H,W) float64 array scaled to
    [0,1] -- the same numeric range ToTensor() produces, since Normalize()
    is applied after ToTensor() in the pipeline below.
    """
    arr = np.asarray(_ensure_rgb(path), dtype="float64") / 255.0  # (H, W, 3)
    return arr.transpose(2, 0, 1)  # (3, H, W)


def _compute_png_band_stats(
    paths: List[str],
    sample_size: int = 200,
    seed: int = 1337,
) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Auto-compute per-channel RGB normalization stats from a random sample
    of PNG/JPEG chips, the same idea as _compute_tiff_band_stats: measured
    from this dataset's own imagery instead of reusing the hardcoded
    ImageNet RGB stats (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    when dataset.compute_png_stats is set. 0 is treated as nodata (matches
    geoetl's PNG output, which also fillna(0)s masked pixels), same
    convention as the TIFF path.

    Returns (None, None) if `paths` is empty (e.g. a pure-TIFF dataset).
    """
    if not paths:
        return None, None

    rng = random.Random(seed)
    sample = paths if len(paths) <= sample_size else rng.sample(paths, sample_size)

    n_bands = None
    total = total_sq = count = None

    for p in sample:
        arr = _load_png_array(p)
        if n_bands is None:
            n_bands = arr.shape[0]
            total = np.zeros(n_bands)
            total_sq = np.zeros(n_bands)
            count = np.zeros(n_bands)
        for b in range(n_bands):
            valid = arr[b][arr[b] > 0]  # 0 = nodata, matches geoetl's convention
            total[b] += valid.sum()
            total_sq[b] += (valid ** 2).sum()
            count[b] += valid.size

    count = np.maximum(count, 1)
    mean = total / count
    var = np.maximum(total_sq / count - mean ** 2, 1e-8)
    std = np.sqrt(var)
    print(f"Computed PNG band stats from {len(sample)} sampled chips "
          f"({n_bands} channels): mean={mean.round(4).tolist()} std={std.round(4).tolist()}")
    return mean.tolist(), std.tolist()


def _resolve_or_compute_png_stats(
    png_paths: List[str],
    ckpt_dir: Optional[str],
    sample_size: int = 200,
    seed: int = 1337,
) -> Tuple[List[float], List[float]]:
    """
    Same cache-once-and-share reasoning as _resolve_or_compute_band_stats
    (see its docstring), for PNG/JPEG's per-channel RGB stats instead of
    TIFF's per-band ones. Separate cache file (png_band_stats.json) --
    the two are independent measurements of different data.

    Unlike the TIFF resolver, this cannot return (None, None): it's only
    called when dataset.compute_png_stats is explicitly set, at which
    point Normalize() unconditionally needs a mean/std pair (whereas the
    TIFF path's fallback-to-ImageNet-stats question doesn't exist here --
    there IS no default for computed PNG stats to fall back to besides
    the ImageNet ones the caller already has and uses when this isn't
    requested at all).
    """
    cache_path = os.path.join(ckpt_dir, "png_band_stats.json") if ckpt_dir else None
    if cache_path and os.path.exists(cache_path):
        with open(cache_path) as f:
            cached = json.load(f)
        print(f"Loaded cached PNG band stats from {cache_path}")
        return cached["band_mean"], cached["band_std"]

    mean, std = _compute_png_band_stats(png_paths, sample_size=sample_size, seed=seed)
    if mean is None:
        raise ValueError(
            "dataset.compute_png_stats is set but this dataset has no "
            "PNG/JPEG items to compute stats from."
        )

    if cache_path:
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({"band_mean": mean, "band_std": std,
                       "n_sampled": min(sample_size, len(png_paths))}, f, indent=2)
        print(f"Cached PNG band stats to {cache_path}")

    return mean, std


def _adjust_contrast_nd(img: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Contrast adjustment that works for any number of channels.

    torchvision's TF.adjust_contrast only supports 1 or 3 channels (it
    raises TypeError for anything else, since it tries to compute a
    grayscale reference), so it can't be reused directly for N-band
    imagery. This is the same blend-toward-mean definition, generalized:
    mean + factor * (img - mean), with the mean taken over the whole
    image (not clamped -- tif reflectance values aren't guaranteed to sit
    in [0,1], unlike 8-bit-derived tensors, so clamping there would be
    wrong).
    """
    mean = img.mean(dim=(-3, -2, -1), keepdim=True)
    return mean + factor * (img - mean)


def _adjust_brightness_nd(img: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Brightness adjustment that works for any number of channels.

    TF.adjust_brightness ALSO only supports 1 or 3 channels -- it was
    wrongly assumed safe here (unlike adjust_contrast, its actual
    definition -- blend toward zero, i.e. a plain scalar multiply -- has
    no real channel-count dependency), and that assumption broke a real
    12-band training run with the same _assert_channels TypeError as
    adjust_contrast. Fixed the same way: reimplemented without the
    assertion. img * factor, unclamped, same reasoning as
    _adjust_contrast_nd above.
    """
    return img * factor


class _TiffAugment:
    """
    Tensor-native augmentation pipeline for multiband TIFF chips. Mirrors
    the geometric/blur/erasing steps of SimbaJSONDataset's PNG pipeline as
    closely as a tensor-first pipeline reasonably can, with two deliberate
    differences:

      - No saturation/hue jitter: those are RGB-color-space operations
        (they convert to HSV) and aren't well-defined for arbitrary band
        counts, so they're dropped rather than silently doing something
        wrong on, say, a 5-band NIR+SWIR chip. Brightness/contrast are
        kept, via _adjust_brightness_nd/_adjust_contrast_nd above rather
        than TF.adjust_brightness/adjust_contrast -- both of the latter
        assert 1-or-3 channels despite neither's actual definition
        depending on channel count, which cost a real training run before
        being caught.
      - Normalization uses per-band mean/std (band_mean/band_std) instead
        of ImageNet RGB stats, since those obviously don't apply here.
        Pass both or neither; if normalize was requested but no stats
        were given, this raises rather than guessing.

    Sizing mirrors the PNG train pipeline's fix (see the resize_or_crop
    comment above it): img_size actually controls training resolution
    now (via Resize, or RandomResizedCrop if use_random_resized_crop),
    matching eval's Resize(img_size) in the default case -- this used to
    be a hardcoded 256 center crop regardless of img_size here too.
    """

    def __init__(
        self,
        img_size: Tuple[int, int],
        train: bool,
        augment: bool,
        normalize: bool,
        band_mean: Optional[List[float]] = None,
        band_std: Optional[List[float]] = None,
        use_random_resized_crop: bool = False,
    ):
        self.img_size = img_size
        self.train = train and augment
        # NOT checked eagerly here: this object is constructed for every
        # dataset regardless of whether it ever actually loads a tiff (a
        # pure-PNG project never calls __call__ on it), and normalize
        # defaults to True -- raising in __init__ would break every
        # existing PNG-only config that doesn't pass band stats. The check
        # happens in __call__ instead, where it's only reached once a
        # tiff item is actually being loaded.
        self.band_mean = band_mean
        self.band_std = band_std
        self.normalize = normalize
        # RandomResizedCrop is a torchvision v1 transform class, same as
        # RandomErasing below -- both dispatch through functional kernels
        # that support tensor input directly (crop + resize, no
        # colorimetric assertion), so it's safe to reuse as-is here.
        self._resize_or_crop = (
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1))
            if use_random_resized_crop else None
        )
        self._erase = transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3))

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if self.train:
            img = self._resize_or_crop(img) if self._resize_or_crop is not None else TF.resize(img, list(self.img_size))
            if random.random() < 0.5:
                img = TF.hflip(img)
            if random.random() < 0.5:
                img = TF.vflip(img)
            img = TF.rotate(img, random.uniform(-20, 20))
            img = _adjust_brightness_nd(img, 1.0 + random.uniform(-0.2, 0.2))
            img = _adjust_contrast_nd(img, 1.0 + random.uniform(-0.2, 0.2))
            if random.random() < 0.2:
                img = TF.gaussian_blur(img, kernel_size=3, sigma=random.uniform(0.1, 1.5))
        else:
            img = TF.resize(img, list(self.img_size))

        if self.normalize:
            if self.band_mean is None or self.band_std is None:
                raise ValueError(
                    "normalize=True but this dataset has no band_mean/"
                    "band_std -- there's no generic default the way "
                    "ImageNet RGB stats are for 3-band photos. Pass "
                    "dataset.band_mean / dataset.band_std (one value per "
                    "band) in the config for a TIFF dataset."
                )
            img = TF.normalize(img, mean=self.band_mean, std=self.band_std)

        if self.train:
            img = self._erase(img)

        return img

def resolve_json_paths(data_root: str, prefix: str, with_neighbors: bool = True):
    ys = os.path.join(data_root, f"{prefix}_ys.json")
    coords = os.path.join(data_root, f"{prefix}_coords.json")
    dup = os.path.join(data_root, f"{prefix}_dup_ys.json") if with_neighbors else None

    missing = [p for p in [ys, coords] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")
    if with_neighbors and (dup is None or not os.path.exists(dup)):
        raise FileNotFoundError(f"Neighbors requested but missing file: {dup}")
    return ys, coords, dup

# ---------- core dataset ----------

class SimbaJSONDataset(Dataset):
    """
    Expects:
      - ys_path:     {image_path: y}
      - coords_path: {image_path: [lon, lat]}
      - dup_path:    {image_path: list|int}  # optional; neighbors
    Returns dict with keys: image, coords([lat,lon]), label,
                            optional neighbor_images [Nmax,3,H,W], neighbor_mask [Nmax]
    """
    def __init__(
        self,
        root_dir: str,
        ys_path: str,
        coords_path: str,
        dup_path: Optional[str] = None,
        split_indices: Optional[List[int]] = None,
        max_neighbors: int = 10,
        img_size: Tuple[int,int] = (224,224),
        normalize: bool = True,
        seed: int = 1337,
        # NEW: control augmentation strength & mode
        train: bool = True,
        augment: bool = True,
        use_random_resized_crop: bool = False,  # set True only if changing FOV is OK
        validate: bool = False,
        ckpt_dir: Optional[str] = None,
        new = True,
        # NEW: multiband TIFF support. Only consulted for items whose path
        # ends in .tif/.tiff -- PNG/JPEG items keep using the pipeline
        # above unchanged. See _TiffAugment for why these can't share
        # ImageNet's mean/std or ColorJitter's saturation/hue. Leave
        # band_mean/band_std as None to auto-compute them from a sample
        # of this dataset's own tiff chips instead of supplying them by
        # hand -- see _resolve_or_compute_band_stats.
        band_mean: Optional[List[float]] = None,
        band_std: Optional[List[float]] = None,
        tif_scale_divisor: float = 10000.0,
        band_stats_sample_size: int = 200,
        # NEW: opt-in auto-computed PNG/JPEG normalization stats, instead
        # of the hardcoded ImageNet RGB stats below. Default False keeps
        # every existing PNG config/checkpoint's behavior bit-for-bit
        # unchanged -- an already-trained model expects ImageNet-stats-
        # normalized input, so this can't default to True the way the
        # TIFF path's auto-compute could (there was no prior TIFF
        # behavior to preserve).
        compute_png_stats: bool = False,
    ):
        if (band_mean is None) != (band_std is None):
            raise ValueError(
                "band_mean and band_std must be given together, or both "
                "left as None to auto-compute them -- got only one."
            )
        super().__init__()
        self.root = root_dir
        self.max_neighbors = max_neighbors
        self.ys = _load_json(ys_path)
        self.ys = {k: v for k, v in self.ys.items() if not math.isnan(v)}

        print("NUM YS: ", len(self.ys), ys_path)

        # print(list(self.ys.keys()))

        # dasag

        self.coords = _load_json(coords_path)
        # self.dups_raw = _load_json(dup_path) if dup_path is not None else None

        # ys_keys = list(self.ys.keys())

        # coords_keys = list(self.coords.keys())
        # self.dups_index = _build_dups_index_by_basename(self.root, ys_keys, coords_keys, self.dups_raw)
        
        # # NEW: fallback if empty (handles your BF/PHL case where coords already has _1.._10)
        # if self.dups_index is not None and len(self.dups_index) == 0:
        #     self.dups_index = _build_dups_from_coords(self.root, ys_keys, coords_keys)
        
        # intersect keys
        keys = set(self.ys) & set(self.coords)
        # if self.dups_index is not None:
        #     keys &= set(self.dups_index)
        self.items = sorted(keys)

        print(self.items[0:5])

        print(len(self.items))

        # gjhgk

        if len(self.items) == 0:
            raise ValueError(
                "No overlapping base keys across ys/coords (and dups). "
                # f"ys={len(self.ys)} coords={len(self.coords)} dups={'None' if self.dups_index is None else len(self.dups_index)}\n"
                f"ys={len(self.ys)} coords={len(self.coords)}\n"
                f"Example ys key: {next(iter(self.ys)) if self.ys else 'EMPTY'}\n"
                f"Example coords key: {next(iter(self.coords)) if self.coords else 'EMPTY'}\n"
                "Hint: We now fall back to grouping neighbors from coords by basename root "
                "(clusterid_*.tiff). Ensure ys uses either clusterid_1.tiff or clusterid.tiff for the base."
            )
        
        elif validate:
            
            print("here in validate!!")

            # If validating the validation set used in training
            if not new:

                p = os.path.join(ckpt_dir, "test_indices.txt")
                with open(p, "r") as f:
                    test_names = f.read().splitlines()
                    self.items = list(set(self.items) & set(test_names))
    
                print(len(self.items))

            # If validating on another dataset
            else:
                # set the save path correctly (now it's rewriting)
                pass


             

        # -------------------------------
        # Transforms (train vs. val/test)
        # -------------------------------
        if compute_png_stats:
            png_paths = [
                p if os.path.isabs(p) else os.path.join(self.root, p)
                for p in self.items if not _is_tiff(p)
            ]
            png_mean, png_std = _resolve_or_compute_png_stats(
                png_paths, ckpt_dir, sample_size=band_stats_sample_size, seed=seed,
            )
        else:
            png_mean, png_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        # Base resize or crop
        if train and augment:
            print("IN TRAIN AND AUGMENT!!")
            resize_or_crop = (
                transforms.RandomResizedCrop(
                    img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)
                ) if use_random_resized_crop else transforms.Resize(img_size)
            )
            tf_list = [
                # Previously a hardcoded CenterCrop(256) sat here instead,
                # with resize_or_crop computed above but never used --
                # img_size had no effect on training at all, and train saw
                # a center crop at native resolution while eval (Resize)
                # saw the whole chip rescaled: not just a different size
                # but a different operation, on top of ignoring
                # use_random_resized_crop entirely. This now actually
                # sizes training to img_size, matching eval's Resize()
                # when use_random_resized_crop=False (the default).
                resize_or_crop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),  # small rotations, keep content
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
                transforms.ToTensor(),
            ]
            if normalize:
                tf_list.append(transforms.Normalize(mean=png_mean, std=png_std))
            # tensor-only erasing last
            tf_list.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3)))
        else:
            # deterministic/eval pipeline
            tf_list = [
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
            if normalize:
                tf_list.append(transforms.Normalize(mean=png_mean, std=png_std))

        self.tf = transforms.Compose(tf_list)

        self.tif_scale_divisor = tif_scale_divisor
        if normalize and band_mean is None:
            tiff_paths = [
                p if os.path.isabs(p) else os.path.join(self.root, p)
                for p in self.items if _is_tiff(p)
            ]
            band_mean, band_std = _resolve_or_compute_band_stats(
                tiff_paths, tif_scale_divisor, ckpt_dir, sample_size=band_stats_sample_size, seed=seed,
            )
        self.tf_tiff = _TiffAugment(
            img_size=img_size,
            train=train,
            augment=augment,
            normalize=normalize,
            band_mean=band_mean,
            band_std=band_std,
            use_random_resized_crop=use_random_resized_crop,
        )

        if split_indices is not None:
            self.items = [self.items[i] for i in split_indices]

        random.seed(seed)
        # Optional: for reproducibility of tensor-level ops like RandomErasing
        try:
            import torch
            torch.manual_seed(seed)
        except Exception:
            pass

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rel = self.items[idx]
        img_path = rel if os.path.isabs(rel) else os.path.join(self.root, rel)
        if _is_tiff(img_path):
            img = self.tf_tiff(_load_tiff_tensor(img_path, self.tif_scale_divisor))
        else:
            img = self.tf(_ensure_rgb(img_path))

        lon, lat = self.coords[rel]
        coords = torch.tensor([float(lat), float(lon)], dtype=torch.float32)
    
        y = self.ys[rel]
        try:
            y_float = float(y)
        except Exception:
            raise ValueError(f"Label for {rel} must be a float, got {y!r}")
        label = torch.tensor(y_float, dtype=torch.float32)
            
        out: Dict[str, Any] = {"image": img, "coords": coords, "label": label}

        out["image_name"] = img_path
    
        # if self.dups_index is not None:
        #     neigh_full = self.dups_index.get(rel, [])[: self.max_neighbors]
        #     n_imgs = []
        #     for p in neigh_full:
        #         p_use = p if os.path.isabs(p) else os.path.join(self.root, p)
        #         if os.path.exists(p_use):
        #             n_imgs.append(self.tf(_ensure_rgb(p_use)))
        #     n = len(n_imgs)
        #     if n == 0:
        #         pad = torch.zeros_like(img)
        #         n_imgs = [pad for _ in range(self.max_neighbors)]
        #         mask = torch.zeros(self.max_neighbors, dtype=torch.float32)
        #     else:
        #         pad = torch.zeros_like(n_imgs[0])
        #         if n < self.max_neighbors:
        #             n_imgs += [pad for _ in range(self.max_neighbors - n)]
        #         mask = torch.cat([
        #             torch.ones(n, dtype=torch.float32),
        #             torch.zeros(self.max_neighbors - n, dtype=torch.float32)
        #         ])
        #     out["neighbor_images"] = torch.stack(n_imgs, dim=0)
        #     out["neighbor_mask"] = mask
    
        return out

# ---------- adapter (build loaders) ----------

class JSONGeoAdapter(BaseDatasetAdapter):
    def __init__(
        self,
        root_dir: str,
        ys_path: str,
        coords_path: str,
        ckpt_dir: str,
        dup_path: Optional[str] = None,
        batch_size: int = 16,
        max_neighbors: int = 10,
        img_size: Tuple[int,int] = (224,224),
        normalize: bool = True,
        split: Tuple[float,float,float] = (0.8, 0.1, 0.1),
        shuffle_train: bool = True,
        num_workers: int = 0,
        seed: int = 1337, # need to handle if input is None I think
        write_files = True,
        # NEW: multiband TIFF support -- see SimbaJSONDataset/_TiffAugment.
        # No-ops for PNG/JPEG-only datasets.
        band_mean: Optional[List[float]] = None,
        band_std: Optional[List[float]] = None,
        tif_scale_divisor: float = 10000.0,
        band_stats_sample_size: int = 200,
        compute_png_stats: bool = False,
    ):
        # Build an index over the full set to split once. Constructed
        # first, so if band_mean/band_std aren't given, this is the
        # instance that actually auto-computes them (from the full,
        # pre-split item set) and caches to ckpt_dir/band_stats.json --
        # _train/_val/_test below just load that cache rather than each
        # recomputing their own from their own split. See
        # _resolve_or_compute_band_stats for why that sharing matters.
        full = SimbaJSONDataset(root_dir, ys_path, coords_path, dup_path,
                                split_indices=None, max_neighbors=max_neighbors,
                                img_size=img_size, normalize=normalize, seed=seed,
                                ckpt_dir=ckpt_dir,
                                band_mean=band_mean, band_std=band_std,
                                tif_scale_divisor=tif_scale_divisor,
                                band_stats_sample_size=band_stats_sample_size,
                                compute_png_stats=compute_png_stats)
        n = len(full)
        idxs = list(range(n))
        # idxs = list(range(52))
        random.Random(seed).shuffle(idxs)
        n_train = int(split[0]*n)
        n_val   = int(split[1]*n)
        # n_train = 32
        # n_val = 20
        train_idx = idxs[:n_train]
        val_idx   = idxs[n_train:n_train+n_val]
        test_idx  = idxs[n_train+n_val:]

        self._train = SimbaJSONDataset(root_dir, ys_path, coords_path, dup_path,
                                       split_indices=train_idx, max_neighbors=max_neighbors,
                                       img_size=img_size, normalize=normalize, seed=seed,
                                       ckpt_dir=ckpt_dir,
                                       band_mean=band_mean, band_std=band_std,
                                       tif_scale_divisor=tif_scale_divisor,
                                       band_stats_sample_size=band_stats_sample_size,
                                       compute_png_stats=compute_png_stats)
        # train=False/augment=False: val and test must use the
        # deterministic eval transform (resize + normalize only), not the
        # training augmentation pipeline. Both previously defaulted to
        # True (unset here), so the val_loss Trainer.fit() reports and
        # logs every epoch -- and would use for early stopping -- was
        # being computed on randomly cropped/flipped/rotated/jittered/
        # blurred/erased images instead of a stable, comparable eval
        # signal.
        self._val   = SimbaJSONDataset(root_dir, ys_path, coords_path, dup_path,
                                       split_indices=val_idx, max_neighbors=max_neighbors,
                                       img_size=img_size, normalize=normalize, seed=seed,
                                       train=False, augment=False,
                                       ckpt_dir=ckpt_dir,
                                       band_mean=band_mean, band_std=band_std,
                                       tif_scale_divisor=tif_scale_divisor,
                                       band_stats_sample_size=band_stats_sample_size,
                                       compute_png_stats=compute_png_stats)
        self._test  = SimbaJSONDataset(root_dir, ys_path, coords_path, dup_path,
                                       split_indices=test_idx, max_neighbors=max_neighbors,
                                       img_size=img_size, normalize=normalize, seed=seed,
                                       train=False, augment=False,
                                       ckpt_dir=ckpt_dir,
                                       band_mean=band_mean, band_std=band_std,
                                       tif_scale_divisor=tif_scale_divisor,
                                       band_stats_sample_size=band_stats_sample_size,
                                       compute_png_stats=compute_png_stats)

        print("Seed: ", seed)
        print("Write Files: ", write_files)

        # jdkajgaklj

        if write_files:

            # Write validation indices to file
            with open(f"{ckpt_dir}/val_indices.txt", "w") as val_file:
                val_file.write('\n'.join(map(str, self._val.items)))           
    
            # Write validation indices to file
            with open(f"{ckpt_dir}/test_indices.txt", "w") as test_file:
                test_file.write('\n'.join(map(str, self._test.items)))     

        
        self.bs = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers

    def train_loader(self) -> DataLoader:
        return DataLoader(self._train, batch_size=self.bs, shuffle=self.shuffle_train, num_workers=self.num_workers)

    def val_loader(self) -> DataLoader:
        return DataLoader(self._val, batch_size=self.bs, shuffle=False, num_workers=self.num_workers)

    def test_loader(self) -> DataLoader:
        return DataLoader(self._test, batch_size=self.bs, shuffle=False, num_workers=self.num_workers)

    @property
    def spatial_crs(self) -> str:
        return "EPSG:4326"




# from __future__ import annotations
# import torch
# from torch.utils.data import Dataset, DataLoader
# from typing import Tuple, Dict, Any
# from ..core.base_dataset import BaseDatasetAdapter
# from ..core.registries import DatasetRegistry

# import os


# # --- Toy dataset that returns image, coords, neighbor_images, label ---
# class _ToyGeoDataset(Dataset):
#     def __init__(self, n: int = 512, n_neighbors: int = 4, img_hw: Tuple[int,int]=(224,224), n_classes: int = 10):
#         self.n = n
#         self.n_neighbors = n_neighbors
#         self.H, self.W = img_hw
#         self.n_classes = n_classes

#         # fixed lat/lon-ish
#         self.coords = torch.empty(n, 2).uniform_(-60, 60)
#         self.images = torch.randn(n, 3, self.H, self.W)
#         self.labels = torch.randint(0, n_classes, (n,))
#         # pre-generate neighbor images per sample for simplicity
#         self.neighbors = torch.randn(n, n_neighbors, 3, self.H, self.W)

#     def __len__(self) -> int:
#         return self.n

#     def __getitem__(self, i: int) -> Dict[str, Any]:
#         return {
#             "image": self.images[i],
#             "coords": self.coords[i],
#             "neighbor_images": self.neighbors[i],
#             "label": self.labels[i],
#         }

# class ToyGeoAdapter(BaseDatasetAdapter):
#     def __init__(self, batch_size: int = 16, n_neighbors: int = 4, img_hw=(224,224), n_classes: int = 10):
#         self.bs = batch_size
#         self.n_neighbors = n_neighbors
#         self.img_hw = img_hw
#         self.n_classes = n_classes
#         self._train = _ToyGeoDataset(512, n_neighbors, img_hw, n_classes)
#         self._val   = _ToyGeoDataset(128, n_neighbors, img_hw, n_classes)
#         self._test  = _ToyGeoDataset(128, n_neighbors, img_hw, n_classes)

#     def train_loader(self) -> DataLoader:
#         return DataLoader(self._train, batch_size=self.bs, shuffle=True, num_workers=0)
#     def val_loader(self) -> DataLoader:
#         return DataLoader(self._val, batch_size=self.bs, shuffle=False, num_workers=0)
#     def test_loader(self) -> DataLoader:
#         return DataLoader(self._test, batch_size=self.bs, shuffle=False, num_workers=0)

#     @property
#     def spatial_crs(self) -> str:
#         return "EPSG:4326"




# def resolve_json_paths(data_root: str, prefix: str, with_neighbors: bool = True):
#     """
#     Given a root directory and dataset prefix, resolve JSON file paths.

#     Args:
#         data_root: Directory containing the JSON files & images.
#         prefix: Dataset prefix, e.g. "phl" or "western_africa".
#         with_neighbors: Whether to expect *_dup_ys.json.

#     Returns:
#         Tuple of (ys_path, coords_path, dup_path or None).
#     """
#     ys = os.path.join(data_root, f"{prefix}_ys.json")
#     coords = os.path.join(data_root, f"{prefix}_coords.json")
#     dup = os.path.join(data_root, f"{prefix}_dup_ys.json") if with_neighbors else None

#     missing = [p for p in [ys, coords] if not os.path.exists(p)]
#     if missing:
#         raise FileNotFoundError(f"Missing required files: {missing}")

#     if with_neighbors and not os.path.exists(dup):
#         raise FileNotFoundError(f"Neighbors requested but missing file: {dup}")

#     return ys, coords, dup






# @DatasetRegistry.register("toygeo")
# def _make_toygeo():
#     return ToyGeoAdapter()
