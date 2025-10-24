import numpy as np
import pandas as pd

def summarize_variogram_from_df(df: pd.DataFrame, gamma_col="gamma", h_col="distance_km",
                                eff_level=0.95):
    """
    df must have columns:
      - h_col: distances in km
      - gamma_col: semivariance γ(h)

    Returns dict with nugget, slope0, sill, eff_range_km (+ a few extras).
    """
    # 1) Clean & sort
    d = df[[h_col, gamma_col]].dropna().drop_duplicates().sort_values(h_col).to_numpy()
    hs = d[:, 0].astype(float)
    gs = d[:, 1].astype(float)

    if len(hs) < 1:
        raise ValueError("No data to summarize.")
    if len(hs) == 1:
        nugget = gs[0]
        return {
            "nugget": nugget,
            "slope0": np.nan,
            "sill": gs[0],
            "eff_range_km": np.nan,
            "distances": hs,
            "gammas": gs
        }

    # 2) Core summaries
    nugget = gs[0]
    slope0 = (gs[1] - gs[0]) / (hs[1] - hs[0])  # finite-difference near origin
    sill = float(np.max(gs))

    # 3) Effective range: first h where γ(h) reaches eff_level * sill, with interpolation
    if sill <= 0:
        eff_range = np.nan
    else:
        target = eff_level * sill
        # find first index where gamma >= target
        idx = np.argmax(gs >= target)  # returns 0 if first already >= target
        if gs[idx] < target:  # never reaches target
            eff_range = np.nan
        elif idx == 0:
            eff_range = hs[0]
        else:
            # linear interpolate between (idx-1, idx)
            h0, g0 = hs[idx-1], gs[idx-1]
            h1, g1 = hs[idx],   gs[idx]
            if g1 == g0:
                eff_range = h1
            else:
                t = (target - g0) / (g1 - g0)
                eff_range = h0 + t * (h1 - h0)

    return {
        "nugget": float(nugget),
        "slope0": float(slope0),
        "sill": sill,
        "eff_range_km": float(eff_range) if not np.isnan(eff_range) else np.nan,
        "distances": hs,
        "gammas": gs
    }


import matplotlib.pyplot as plt

def plot_variogram_summary(metrics, title=None):
    hs = metrics["distances"]; gs = metrics["gammas"]
    plt.figure(figsize=(5,3.5))
    plt.plot(hs, gs, marker='o')
    plt.xlabel("Perturbation distance h (km)")
    plt.ylabel("Semivariance γ(h)")
    if title: plt.title(title)

    # annotate nugget / sill / effective range
    plt.axhline(metrics["sill"], linestyle="--", alpha=0.5)
    if not np.isnan(metrics["eff_range_km"]):
        plt.axvline(metrics["eff_range_km"], linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("./test1.png")
    plt.show()


