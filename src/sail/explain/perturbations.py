import torch

def lat_shift(coords: torch.Tensor, meters: float) -> torch.Tensor:
    meters_per_deg_lat = 111_000.0
    dlat = meters / meters_per_deg_lat
    out = coords.clone()
    out[:, 0] = out[:, 0] + dlat
    return out
