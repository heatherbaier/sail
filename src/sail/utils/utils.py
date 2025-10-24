from pathlib import Path
import re

import math, random
import numpy as np
import torch

def highest_epoch(dir_path=".", max_epoch=None):
    pat = re.compile(r"^model_epoch(\d+)\.torch$")
    best = max(
        (
            (int(m.group(1)), p)
            for p in Path(dir_path).iterdir()
            if (m := pat.match(p.name))
            and (max_epoch is None or int(m.group(1)) <= max_epoch)
        ),
        default=(None, None),
    )
    return best  # (epoch_number, Path)






EARTH_R_KM = 6371.0088

# --- Distance helpers ---
def km_to_deg_lat(km: float) -> float:
    return km / 110.574

def km_to_deg_lon(km: float, lat_deg: float) -> float:
    return km / (111.320 * math.cos(math.radians(lat_deg)))

def great_circle_km(lat1, lon1, lat2, lon2):
    phi1, lam1, phi2, lam2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * EARTH_R_KM * math.asin(math.sqrt(a))

# --- Perturbation sampler ---
def sample_coord_at_distance(lat0, lon0, h_km, n_samples, directional=None, rng=None):
    rng = rng or random
    out = []
    for _ in range(n_samples):
        if directional is None:
            bearing = rng.uniform(0, 2*math.pi)
        else:
            base = {'E':0, 'N':math.pi/2, 'W':math.pi, 'S':3*math.pi/2}[directional]
            bearing = base + rng.uniform(-math.radians(10), math.radians(10))
        dlat = km_to_deg_lat(h_km * math.sin(bearing))
        dlon = km_to_deg_lon(h_km * math.cos(bearing), lat0)
        out.append((lat0 + dlat, lon0 + dlon))
    return out



def sample_bearing(rng=None):
    rng = rng or random
    return rng.uniform(0, 2*math.pi)

def move_from(lat0, lon0, dist_km, bearing_rad):
    """Return (lat, lon) moved by dist_km along bearing from (lat0, lon0)."""
    dlat = km_to_deg_lat(dist_km * math.sin(bearing_rad))
    dlon = km_to_deg_lon(dist_km * math.cos(bearing_rad), lat0)
    return (lat0 + dlat, lon0 + dlon)

def rayleigh_truncated(max_h, sigma, size, rng=None):
    """
    Draw Rayleigh distances truncated at max_h (simple rejection sampler).
    sigma in km; max_h in km.
    """
    rng = rng or np.random
    out = []
    # Cap attempts for safety
    attempts = 0
    need = size
    while need > 0 and attempts < 10000:
        # oversample a bit to reduce loops
        draws = rng.rayleigh(scale=sigma, size=need*3)
        kept = draws[draws <= max_h]
        if kept.size > 0:
            take = kept[:need]
            out.append(take)
            need -= take.size
        attempts += 1
    if need > 0:
        # fall back to uniform(0,max_h) for any remaining
        out.append(np.random.uniform(0, max_h, size=need))
    return np.concatenate(out)



import numpy as np
import pandas as pd

def gamma_matrix_from_table(
    df: pd.DataFrame,
    distance_col: str = "distance_km",
    pred_col: str = "y_pred",
    y0: float = None,
    drop_incomplete: bool = True
):
    """
    Build a (N_permutations, N_distances) matrix of semivariances gamma_i(h)
    from a long table with columns [distance_km, y_pred, ...].

    - N_permutations = number of samples per distance (rows). If the counts
      differ across distances, rows with missing values can be dropped or kept as NaN.

    Returns:
        gamma_mat : np.ndarray of shape (N_permutations, N_distances)
        distances : np.ndarray of sorted unique distances
    """
    df = df.copy()

    # pick baseline prediction y0
    if y0 is None:
        # prefer an explicit 0-distance row if present
        zero_mask = (df[distance_col] == 0)
        if zero_mask.any():
            y0 = df.loc[zero_mask, pred_col].iloc[0]
        else:
            # fallback: first row's prediction (adjust if you store base pred elsewhere)
            y0 = float(df.iloc[0][pred_col])

    # per-sample semivariance
    df["gamma_item"] = 0.5 * (df[pred_col] - y0) ** 2

    # index samples within each distance bin
    df["perm_idx"] = df.groupby(distance_col).cumcount()

    # pivot to matrix: rows = permutation index, cols = distances
    wide = (
        df.pivot(index="perm_idx", columns=distance_col, values="gamma_item")
          .sort_index(axis=1)
    )

    distances = np.array(wide.columns.tolist(), dtype=float)
    gamma_mat = wide.to_numpy()  # shape (max_samples_across_bins, N_distances)

    if drop_incomplete:
        # keep only rows that have all distances present (no NaNs)
        mask = ~np.isnan(gamma_mat).any(axis=1)
        gamma_mat = gamma_mat[mask]

    return gamma_mat, distances




def pvalues_null_vs_obs(null_mat, gamma_obs, reduce_over_offsets="mean"):
    """
    Compare observed semivariances (gamma_obs: K x D) to a null matrix (null_mat: B x D).

    Returns:
        p_per_offset: (K, D) p-values (one per offset, per distance)
        p_agg:        (D,) aggregated across offsets (mean / min / max)
    """
    null = np.asarray(null_mat, dtype=float)    # (B, D)
    obs  = np.asarray(gamma_obs, dtype=float)   # (K, D)

    # Broadcast to (B, K, D): compare each bootstrap to each observed offset at each distance
    comp = null[:, None, :] >= obs[None, :, :]  # (B, K, D)
    p_per_offset = comp.mean(axis=0)            # average over bootstraps -> (K, D)

    # Aggregate across offsets to get one p per distance
    if   reduce_over_offsets == "mean": p_agg = p_per_offset.mean(axis=0)  # (D,)
    elif reduce_over_offsets == "min":  p_agg = p_per_offset.min(axis=0)   # conservative (worst-case offset)
    elif reduce_over_offsets == "max":  p_agg = p_per_offset.max(axis=0)   # liberal  (best-case offset)
    else:
        raise ValueError("reduce_over_offsets must be one of {'mean','min','max'}")

    return p_per_offset, p_agg



def _effective_range(hs, gamma_obs, eff_level=0.95):
    """Linear interpolate distance where gamma reaches eff_level * sill."""
    sill = float(np.nanmax(gamma_obs))
    if sill <= 0 or np.all(np.isnan(gamma_obs)):
        return float("nan")
    target = eff_level * sill
    idxs = np.where(gamma_obs >= target)[0]
    if idxs.size == 0:
        return float("nan")
    if idxs[0] == 0:
        return float(hs[0])
    i = idxs[0]
    h0, g0 = hs[i-1], gamma_obs[i-1]
    h1, g1 = hs[i],   gamma_obs[i]
    if g1 == g0:
        return float(h1)
    t = (target - g0) / (g1 - g0)
    return float(h0 + t * (h1 - h0))

# def _pvalues_null_vs_obs(null_mat, gamma_obs, reduce_over_offsets="mean"):
#     """
#     null_mat: (B, D)
#     gamma_obs: (K, D)
#     Returns:
#       p_per_offset: (K, D)
#       p_agg: (D,)
#     """
#     null = np.asarray(null_mat, dtype=float)   # (B, D)
#     obs  = np.asarray(gamma_obs, dtype=float)  # (K, D)
#     comp = null[:, None, :] >= obs[None, :, :] # (B, K, D)
#     p_per_offset = comp.mean(axis=0)           # (K, D)
#     if   reduce_over_offsets == "mean": p_agg = p_per_offset.mean(axis=0)
#     elif reduce_over_offsets == "min":  p_agg = p_per_offset.min(axis=0)
#     elif reduce_over_offsets == "max":  p_agg = p_per_offset.max(axis=0)
#     else:
#         raise ValueError("reduce_over_offsets must be one of {'mean','min','max'}")
#     return p_per_offset, p_agg

