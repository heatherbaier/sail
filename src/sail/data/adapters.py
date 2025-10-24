from __future__ import annotations
import os, json, random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
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
    ):
        super().__init__()
        self.root = root_dir
        self.max_neighbors = max_neighbors
        self.ys = _load_json(ys_path)
        print("NUM YS: ", len(self.ys), ys_path)
        self.coords = _load_json(coords_path)
        self.dups_raw = _load_json(dup_path) if dup_path is not None else None

        ys_keys = list(self.ys.keys())

        coords_keys = list(self.coords.keys())
        self.dups_index = _build_dups_index_by_basename(self.root, ys_keys, coords_keys, self.dups_raw)
        
        # NEW: fallback if empty (handles your BF/PHL case where coords already has _1.._10)
        if self.dups_index is not None and len(self.dups_index) == 0:
            self.dups_index = _build_dups_from_coords(self.root, ys_keys, coords_keys)
        
        # intersect keys
        keys = set(self.ys) & set(self.coords)
        if self.dups_index is not None:
            keys &= set(self.dups_index)
        self.items = sorted(keys)

        print(self.items[0:5])

        print(len(self.items))

        # gjhgk

        if len(self.items) == 0:
            raise ValueError(
                "No overlapping base keys across ys/coords (and dups). "
                f"ys={len(self.ys)} coords={len(self.coords)} dups={'None' if self.dups_index is None else len(self.dups_index)}\n"
                f"Example ys key: {next(iter(self.ys)) if self.ys else 'EMPTY'}\n"
                f"Example coords key: {next(iter(self.coords)) if self.coords else 'EMPTY'}\n"
                "Hint: We now fall back to grouping neighbors from coords by basename root "
                "(clusterid_*.tiff). Ensure ys uses either clusterid_1.tiff or clusterid.tiff for the base."
            )
        elif validate:
            print("here in validate!!")

            p = os.path.join(ckpt_dir, "test_indices.txt")
            with open(p, "r") as f:
                test_names = f.read().splitlines()
                self.items = list(set(self.items) & set(test_names))

            print(len(self.items))

        # -------------------------------
        # Transforms (train vs. val/test)
        # -------------------------------
        # Base resize or crop
        if train and augment:
            resize_or_crop = (
                transforms.RandomResizedCrop(
                    img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)
                ) if use_random_resized_crop else transforms.Resize(img_size)
            )
            tf_list = [
                resize_or_crop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),  # small rotations, keep content
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
                transforms.ToTensor(),
            ]
            if normalize:
                tf_list.append(transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))
            # tensor-only erasing last
            tf_list.append(transforms.RandomErasing(p=0.25, scale=(0.02, 0.08), ratio=(0.3, 3.3)))
        else:
            # deterministic/eval pipeline
            tf_list = [
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ]
            if normalize:
                tf_list.append(transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]))

        self.tf = transforms.Compose(tf_list)

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
    
        if self.dups_index is not None:
            neigh_full = self.dups_index.get(rel, [])[: self.max_neighbors]
            n_imgs = []
            for p in neigh_full:
                p_use = p if os.path.isabs(p) else os.path.join(self.root, p)
                if os.path.exists(p_use):
                    n_imgs.append(self.tf(_ensure_rgb(p_use)))
            n = len(n_imgs)
            if n == 0:
                pad = torch.zeros_like(img)
                n_imgs = [pad for _ in range(self.max_neighbors)]
                mask = torch.zeros(self.max_neighbors, dtype=torch.float32)
            else:
                pad = torch.zeros_like(n_imgs[0])
                if n < self.max_neighbors:
                    n_imgs += [pad for _ in range(self.max_neighbors - n)]
                mask = torch.cat([
                    torch.ones(n, dtype=torch.float32),
                    torch.zeros(self.max_neighbors - n, dtype=torch.float32)
                ])
            out["neighbor_images"] = torch.stack(n_imgs, dim=0)
            out["neighbor_mask"] = mask
    
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
        seed: int = 1337,
    ):
        # Build an index over the full set to split once
        full = SimbaJSONDataset(root_dir, ys_path, coords_path, dup_path,
                                split_indices=None, max_neighbors=max_neighbors,
                                img_size=img_size, normalize=normalize, seed=seed)
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
                                       img_size=img_size, normalize=normalize, seed=seed)
        self._val   = SimbaJSONDataset(root_dir, ys_path, coords_path, dup_path,
                                       split_indices=val_idx, max_neighbors=max_neighbors,
                                       img_size=img_size, normalize=normalize, seed=seed)
        self._test  = SimbaJSONDataset(root_dir, ys_path, coords_path, dup_path,
                                       split_indices=test_idx, max_neighbors=max_neighbors,
                                       img_size=img_size, normalize=normalize, seed=seed)

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
