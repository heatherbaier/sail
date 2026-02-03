from __future__ import annotations
import os, json, random
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import re

from ..core.base_dataset import BaseDatasetAdapter


import json
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from typing import Dict, Any, List

from torch.nn.utils.rnn import pad_sequence

  
# def temporal_collate(batch):
#     imgs = [b["image"] for b in batch]
#     y = torch.stack([b["label"] for b in batch])
#     ids = [b["image_name"] for b in batch]
#     # pad to max T in this batch
#     Tmax = max(x.size(0) for x in imgs)
#     padded = []
#     masks = []
#     for x in imgs:
#         T = x.size(0)
#         pad = torch.zeros((Tmax - T, 3, x.size(2), x.size(3)))
#         padded.append(torch.cat([x, pad], dim=0))
#         m = torch.zeros(Tmax)
#         m[:T] = 1
#         masks.append(m)

#     imgs = torch.stack(padded)
#     masks = torch.stack(masks)
#     return {"image": imgs, "mask": masks, "label": y, "id": ids}


# class TemporalSchoolDataset(Dataset):
#     def __init__(
#         self,
#         json_path: str,
#         img_dir: str,
#         transform: transforms.Compose = None,
#         img_ext: str = ".jpg",
#         validate: bool = False
#     ):
#         """
#         Args:
#             json_path: path to JSON with structure:
#                 {
#                     "school_001": {
#                         "images": ["img_001_2014_01.jpg", "img_001_2014_02.jpg", ...],
#                         "y": 0.72
#                     },
#                     "school_002": {
#                         "images": [...],
#                         "y": 0.53
#                     },
#                     ...
#                 }
#             img_dir: directory where images are stored
#             transform: torchvision transform(s)
#             img_ext: optional override for extension handling
#         """
#         with open(json_path, "r") as f:
#             self.data = json.load(f)

#         self.ids = list(self.data.keys())
#         self.img_dir = img_dir
#         self.transform = transform or transforms.Compose([
#             transforms.Resize((224, 224)),
#             transforms.ToTensor(),
#         ])
#         self.img_ext = img_ext

#         if validate:
            
#             print("here in validate!!")

#             p = os.path.join(self.img_dir, "test_indices.txt")
#             with open(p, "r") as f:
#                 self.ids = f.read().splitlines()
#                 # self.items = list(set(self.items) & set(test_names))

#             print(len(self.ids))


        

#     def __len__(self):
#         return len(self.ids)

#     def __getitem__(self, idx: int):
#         uid = self.ids[idx]
#         record = self.data[uid]
#         img_names = record["chips"]
#         y = torch.tensor(record["label"], dtype=torch.float32)

#         imgs = []
#         for name in img_names:
#             path = name
#             img = Image.open(path).convert("RGB")
#             img = self.transform(img)
#             imgs.append(img)

#         # Stack along time dimension → [T, 3, H, W]
#         imgs = torch.stack(imgs, dim=0)

#         out: Dict[str, Any] = {"image": imgs, "coords": torch.tensor([[1.0]]), "label": y, "image_name": uid}

#         return out


import os, re
import torch
from PIL import Image
from torchvision import transforms
import numpy as np

TIME_RE = re.compile(r'(?P<y>20\d{2})[_-]?m?(?P<m>0[1-9]|1[0-2])')  # matches 2023_m08 or 2023_08 or 202308

def parse_ym(fname: str):
    m = TIME_RE.search(fname)
    if not m:
        raise ValueError(f"Cannot parse year/month from {fname}")
    return int(m.group('y')), int(m.group('m'))

class TemporalSchoolDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 json_path, 
                 img_dir, 
                 normalize=True, 
                 T=12, 
                 pad_fill="zeros", 
                 start_from=None, 
                 img_size = [256, 256],
                 validate = False):
        """
        T: global #frames to return
        pad_fill: "zeros" or "mean"
        start_from: optional (year, month) to align index 0; else earliest found
        """
        import json
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        self.ids = list(self.data.keys())
        self.img_dir = img_dir
        self.T = T
        self.pad_fill = pad_fill
        self.start_from = start_from
        self.pad_value = 0
        self.img_size = img_size

        self.tx = transforms.Compose([
                
                transforms.CenterCrop(img_size),
                
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),  # small rotations, keep content
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
                
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                
            ])
        
        # self.tx = transforms.Compose([
        #     transforms.CenterCrop(img_size),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        # ])
        # self.pad_value = torch.zeros(3, *img_size)  # normalized space (≈ 0) is fine
        # 
        # else:
        #     self.tx = transforms.Compose([transforms.Resize(img_size),
        #                                   transforms.ToTensor()])
        #     self.pad_value = torch.zeros(3, *img_size)

        # optional: precompute dataset mean for padding
        self.dataset_mean = None
        if pad_fill == "mean":
            self.dataset_mean = self.pad_value  # TODO: compute if you want

        if validate:
            print("here in validate!!")
            p = os.path.join(img_dir, "test_indices.txt")
            with open(p, "r") as f:
                test_names = f.read().splitlines()
                self.ids = list(set(self.ids) & set(test_names))
            print(len(self.ids))            

    def __len__(self): return len(self.ids)

    def _sequence_index(self, pairs, T):
        """
        pairs: list of ( (year,month), filepath )
        Returns ordered list length T of filepaths or None, and mask [T].
        """
        # sort pairs by time
        pairs = sorted(pairs, key=lambda p: (p[0][0], p[0][1]))  # (y,m)

        # choose start anchor
        if self.start_from is None:
            y0, m0 = pairs[0][0]  # earliest
        else:
            y0, m0 = self.start_from

        # map (y,m) → index 0..T-1
        def ym_to_idx(y, m, y0, m0):
            return (y - y0) * 12 + (m - m0)

        slots = [None] * T
        mask = torch.zeros(T)
        for (y,m), fp in pairs:
            idx = ym_to_idx(y, m, y0, m0)
            if 0 <= idx < T and slots[idx] is None:
                slots[idx] = fp
                mask[idx] = 1.0
        return slots, mask

    def __getitem__(self, idx):
        uid = self.ids[idx]
        rec = self.data[uid]  # {"images": [...], "y": ...}

        # parse times for each filename
        pairs = []
        for name in rec["chips"]:
            y,m = parse_ym(name)
            pairs.append(((y,m), os.path.join(self.img_dir, name)))

        slots, mask = self._sequence_index(pairs, self.T)

        imgs = []
        for fp in slots:
            if fp is None:
                # padded frame
                # img_t = self.dataset_mean if (self.dataset_mean is not None) else self.pad_value
                img_t = torch.ones((3, *self.img_size)) * self.pad_value
            else:
                img = Image.open(fp).convert("RGB")
                img_t = self.tx(img)
            imgs.append(img_t)

        imgs = torch.stack(imgs, dim=0)  # [T,3,H,W]
        y = torch.tensor(rec["label"], dtype=torch.float32)

        return {"image": imgs, "coords": torch.tensor([[1.0]]), "mask": mask, "label": y, "image_name": uid}


import os
import json
import random
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from torchvision import transforms
# from .temporal_school_dataset import TemporalSchoolDataset  # import your dataset class

import os, json, random
from typing import Optional, Tuple
from torch.utils.data import DataLoader
from torchvision import transforms
# from .temporal_school_dataset import TemporalSchoolDataset


class TemporalGeoAdapter:
    def __init__(
        self,
        root_dir: str,
        ys_path: str,
        coords_path: Optional[str],
        ckpt_dir: str,
        batch_size: int = 16,
        img_size: Tuple[int, int] = (224, 224),
        normalize: bool = True,
        split: Tuple[float, float, float] = (0.8, 0.1, 0.1),
        shuffle_train: bool = True,
        num_workers: int = 0,
        seed: int = 1337,
    ):
        """
        Args:
            root_dir: imagery directory (holds all monthly images)
            ys_path: path to temporal-JSON file mapping school_id → {"images": [...], "y": val}
            coords_path: optional coordinate file (kept for interface parity)
            ckpt_dir: output directory for split index files
        """
        self.bs = batch_size
        self.shuffle_train = shuffle_train
        self.num_workers = num_workers
        os.makedirs(ckpt_dir, exist_ok=True)

        # --- split once so train/val/test are consistent ---
        with open(ys_path, "r") as f:
            full_data = json.load(f)
        ids = list(full_data.keys())
        random.Random(seed).shuffle(ids)
        n = len(ids)
        n_train = int(split[0] * n)
        n_val = int(split[1] * n)
        train_ids = ids[:n_train]
        val_ids = ids[n_train:n_train + n_val]
        test_ids = ids[n_train + n_val:]

        print("IMG_SIZE: ", img_size)

        # --- transforms ---
        if normalize:
            transform = transforms.Compose([
                
                transforms.CenterCrop(img_size),
                
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(degrees=20),  # small rotations, keep content
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.2),
                
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
                
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(img_size),
                transforms.ToTensor(),
            ])

        # --- build datasets ---
        self._train = TemporalSchoolDataset(ys_path, root_dir, transform)
        self._val   = TemporalSchoolDataset(ys_path, root_dir, transform)
        self._test  = TemporalSchoolDataset(ys_path, root_dir, transform)
        self._train.ids = train_ids
        self._val.ids   = val_ids
        self._test.ids  = test_ids

        # --- write split files ---
        with open(os.path.join(ckpt_dir, "train_indices.txt"), "w") as f:
            f.write("\n".join(train_ids))
        with open(os.path.join(ckpt_dir, "val_indices.txt"), "w") as f:
            f.write("\n".join(val_ids))
        with open(os.path.join(ckpt_dir, "test_indices.txt"), "w") as f:
            f.write("\n".join(test_ids))

    # ----------------------------------------------------------------
    # DataLoader accessors
    # ----------------------------------------------------------------
    def train_loader(self):
        return DataLoader(self._train, batch_size=self.bs,
                          shuffle=self.shuffle_train, num_workers=self.num_workers)#, collate_fn=temporal_collate)

    def val_loader(self):
        return DataLoader(self._val, batch_size=self.bs,
                          shuffle=False, num_workers=self.num_workers)#, collate_fn=temporal_collate)

    def test_loader(self):
        return DataLoader(self._test, batch_size=self.bs,
                          shuffle=False, num_workers=self.num_workers)#, collate_fn=temporal_collate)

    @property
    def spatial_crs(self) -> str:
        return "EPSG:4326"

