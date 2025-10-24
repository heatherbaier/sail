# src/simba/models/tang2015.py
from typing import Any, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.base_model import BaseModelWrapper
from .utils import ResNet18Embedding

def coords_to_grid_onehot(coords, grid_hw=(60,120), lat_bounds=(-90,90), lon_bounds=(-180,180)):
    B = coords.size(0)
    H, W = grid_hw
    lat = coords[:,0].clamp(*lat_bounds)
    lon = coords[:,1].clamp(*lon_bounds)
    lat_idx = ((lat - lat_bounds[0]) / (lat_bounds[1]-lat_bounds[0]) * H).floor().long().clamp(0, H-1)
    lon_idx = ((lon - lon_bounds[0]) / (lon_bounds[1]-lon_bounds[0]) * W).floor().long().clamp(0, W-1)
    cell = lat_idx * W + lon_idx
    return F.one_hot(cell, num_classes=H*W).float()

class CoordMLP(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.ReLU(inplace=True),
            nn.Linear(64, out_dim), nn.ReLU(inplace=True),
        )
    def forward(self, coords):
        x = torch.stack([coords[:,0]/90.0, coords[:,1]/180.0], dim=1)
        return self.net(x)

class VisualContextFixed(nn.Module):
    def __init__(self, out_proj = 256, pretrained = True):
        super().__init__()
        self.enc = ResNet18Embedding(pretrained=pretrained)
        d = self.enc.out_dim
        if out_proj is None:
            self.proj = nn.Identity(); self.out_dim = d
        else:
            self.proj = nn.Sequential(nn.Linear(d, out_proj), nn.ReLU(inplace=True), nn.Dropout(0.2))
            self.out_dim = out_proj
    def forward(self, neighbor_images, neighbor_mask=None):
        B, N, C, H, W = neighbor_images.shape
        x = neighbor_images.view(B*N, C, H, W)
        z = self.enc(x).view(B, N, -1)  # [B,N,512]
        if neighbor_mask is not None:
            m = neighbor_mask.view(B, N, 1)
            z = (z * m).sum(1) / (m.sum(1).clamp_min(1e-6))
        else:
            z = z.mean(1)
        return self.proj(z)  # [B,out_proj]

class ImageGPSContextNet(nn.Module):
    def __init__(self, num_classes: int, grid_hw=(60,120), coord_mlp_dim=64, pretrained=True):
        super().__init__()
        self.img = ResNet18Embedding(pretrained=pretrained)              # 512
        self.pre_img = nn.Sequential(nn.Linear(self.img.out_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.ctx = VisualContextFixed(out_proj=256, pretrained=pretrained)  # 256
        self.grid_hw = grid_hw
        self.coord_mlp = CoordMLP(coord_mlp_dim)                         # 64

        loc_in = (grid_hw[0]*grid_hw[1]) + 256 + coord_mlp_dim           # H*W + 256 + 64
        loc_h  = 512
        self.loc_mlp = nn.Sequential(nn.Linear(loc_in, loc_h), nn.ReLU(inplace=True), nn.Dropout(0.5))

        self.classifier = nn.Linear(256 + loc_h, 1)   # <- 1 output for regression

    def forward(self, image, coords, neighbor_images, neighbor_mask=None):
        z_img = self.pre_img(self.img(image))                                 # [B,256]
        z_ctx = self.ctx(neighbor_images, neighbor_mask)                      # [B,256]
        z_onehot = coords_to_grid_onehot(coords, grid_hw=self.grid_hw)        # [B,H*W]
        z_coord = self.coord_mlp(coords)                                      # [B,64]
        z_loc = self.loc_mlp(torch.cat([z_ctx, z_onehot, z_coord], dim=1))    # [B,512]
        fused = torch.cat([z_img, z_loc], dim=1)                              # [B,768]
        return self.classifier(fused)                                         # [B,num_classes]

class Tang2015Classifier(BaseModelWrapper):
    def __init__(self, num_classes: int = 1):
        self.num_classes = num_classes
        self.net: nn.Module | None = None

    def build(self) -> nn.Module:
        self.net = ImageGPSContextNet(num_classes=self.num_classes)
        return self.net

    def forward(self, batch: Dict[str, Any]):
        # Strict: requires neighbors
        if "neighbor_images" not in batch:
            raise ValueError("Tang2015 requires neighbor_images (and optionally neighbor_mask) in the batch.")
        return self.net(batch["image"], batch["coords"], batch["neighbor_images"], batch.get("neighbor_mask", None))

    def compute_loss(self, pred, batch):
        # pred: [B,1], label: [B]
        target = batch["label"].float().view(-1, 1)
        return F.l1_loss(pred, target)

    @torch.no_grad()
    def predict(self, batch):
        out = self.forward(batch)  # [B,1]
        return out.view(-1)        # [B]

    def save(self, path: str) -> None:
        assert self.net is not None
        # torch.save({"state_dict": self.net.state_dict(),
        #             "num_classes": self.num_classes}, path)


        # Save the most current epoch
        torch.save({
                    # 'epoch': epoch,
                    # 'loss': criterion,
                    'model_state_dict': self.net.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict(),
                    # 'scheduler': scheduler.state_dict(),
                }, path)   
        

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        # print(sd.keys())

        # print(self.net.state_dict().keys())


        # # Strip 'ctx.' from start of any keys
        # new_state_dict = {}
        # for k, v in sd.items():
        #     if k.startswith("ctx."):
        #         new_state_dict[k[len("ctx."):]] = v
        #     else:
        #         new_state_dict[k] = v

        # Build fresh net with current output size (e.g., regression: 1)
        if self.net is None:
            self.net = ImageGPSContextNet(num_classes=self.num_classes)


        # print(self.net.state_dict().keys())


        # dasgfsa

        
        # # Filter: only load params that exist AND have the same shape
        # model_sd = self.net.state_dict()
        # filtered = {}
        # skipped = []
        # for k, v in new_state_dict.items():
        #     if k in model_sd and model_sd[k].shape == v.shape:
        #         filtered[k] = v
        #     else:
        #         skipped.append(k)
    
        # Load matching weights; leave the rest (like classifier.*) randomly initialized
        self.net.load_state_dict(sd, strict=False)
    
        # # (Optional) print a tiny report so you know what got skipped
        # if skipped:
        #     print(f"[SIMBA] Skipped {len(skipped)} keys (shape/name mismatch), e.g.: {skipped[:5]}")
