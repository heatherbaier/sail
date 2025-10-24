from __future__ import annotations
from typing import Dict, Any, Iterable
import torch
from ..core.base_explainer import BaseExplainer
from ..core.base_model import BaseModelWrapper
from ..core.base_dataset import BaseDatasetAdapter
from .perturbations import lat_shift

class SimbaExplainer(BaseExplainer):
    def __init__(self, model_wrapper: BaseModelWrapper, dataset_adapter: BaseDatasetAdapter, device: str | None = None):
        self.mw = model_wrapper
        self.ds = dataset_adapter
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.net = self.mw.build().to(self.device)
        self.net.eval()

    @torch.no_grad()
    def global_sensitivity(self, deltas_m: Iterable[int] = (250, 1000), n_batches: int = 5) -> Dict[str, Any]:
        out: Dict[str, list[float]] = {f"delta_{d}m_mean_abs": [] for d in deltas_m}
        loader = self.ds.test_loader()
        it = iter(loader)
        for _ in range(n_batches):
            try:
                batch = next(it)
            except StopIteration:
                break
            batch = {k: (v.to(self.device) if hasattr(v, "to") else v) for k,v in batch.items()}
            base = self.mw.predict(batch)  # [B,C]
            for d in deltas_m:
                pert = dict(batch)
                pert["coords"] = lat_shift(batch["coords"], d)
                alt = self.mw.predict(pert)
                # mean absolute change in predicted probability vector
                delta = (alt - base).abs().mean().item()
                out[f"delta_{d}m_mean_abs"].append(delta)
        return {k: float(torch.tensor(v).mean()) if len(v)>0 else 0.0 for k,v in out.items()}

    @torch.no_grad()
    def local_sensitivity(self, batch, deltas_m: Iterable[int] = (0, 250, 1000)) -> Dict[str, Any]:
        batch = {k: (v.to(self.device) if hasattr(v, "to") else v) for k,v in batch.items()}
        result: Dict[str, torch.Tensor] = {}
        for d in deltas_m:
            pert = dict(batch)
            if d == 0:
                pert_coords = batch["coords"]
            else:
                pert_coords = lat_shift(batch["coords"], d)
            pert["coords"] = pert_coords
            result[f"{d}m"] = self.mw.predict(pert)  # [B,C]
        return result
