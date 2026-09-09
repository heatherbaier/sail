"""
Sweep candidate spatial_block_deg values for
JSONGeoAdapter(split_strategy="stable", spatial_block_deg=...) and report
diagnostics to help pick a defensible one.

This is NOT an optimizer that hands you a single "correct" number --
there isn't one, it's a tradeoff. Too small and blocks approach
individual-tract granularity, giving no real protection against a val
tract having a train tract right next door (spatial-autocorrelation
leakage) -- same as not blocking at all. Too large and blocks swallow
whole regions: fewer, bigger blocks means the realized train/val/test
proportions drift further from what you asked for (fewer independent
draws = more variance around the target ratio), and val/test can end up
geographically lopsided (e.g. entirely rural, or clustered in one corner
of the state) instead of representative of the whole state. This script
makes that tradeoff visible across a range of candidates so you can pick
where it stabilizes, rather than guessing.

Metrics reported per candidate spatial_block_deg:
  1. n_blocks -- how many distinct spatial blocks this size produces
     across the tracts given.
  2. tracts-per-block distribution (min/median/mean/max) -- a block with
     1 tract gives that tract zero protection; a block with hundreds is
     extremely coarse (and if one block holds a large share of all
     tracts, that single hash draw disproportionately swings the whole
     split).
  3. Realized train/val/test proportions (by tract count) vs. requested
     --split.
  4. Neighbor cross-split fraction -- for a sample of tracts, the
     fraction of each one's K nearest OTHER tracts (by real haversine
     distance on lon/lat centroids) that land in a DIFFERENT split
     bucket. This is the actual leakage-exposure metric: with NO
     blocking, two independent hash draws agree by pure chance on
     0.8^2+0.1^2+0.1^2 = 66% of the time for a (0.8,0.1,0.1) split, so
     ~34% of any tract's neighbors land in a different split purely by
     chance -- that's the "no protection" baseline this should start
     near at very small block_deg, then fall toward 0% as blocks grow.
     Look for the elbow where it flattens rather than picking the
     largest candidate -- past that point you're trading away split
     representativeness (see metrics 2-3) for protection you've already
     gotten.

Also prints the overall nearest-tract-distance distribution (km) up
front, purely as a reference for picking a sensible candidate range --
there's no point sweeping block sizes far smaller than typical
inter-tract spacing.

Usage:
    python find_spatial_block_deg.py \
        --data-roots /data/hbaier/new_data/tlag/az_imagery/q1_2016_s2_allbands/ \
                     /data/hbaier/new_data/tlag/az_imagery/q2_2016_s2_allbands/ \
        --prefixes az_2016_q1_s2_allbands az_2016_q2_s2_allbands \
        --block-degs 0.02 0.05 0.1 0.15 0.2 0.3 0.5

(Pass every quarter you have downloaded for --data-roots/--prefixes --
this only reads coords.json, so more quarters just means a more complete
picture of the state's tract layout, unioned across them.)
"""

import argparse
import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict


# ---------------------------------------------------------------------
# Same hashing/blocking primitives as sail/data/splitting.py, inlined so
# this script has zero dependency on the sail package (or torch/rasterio)
# and can run anywhere with just python3's stdlib.
# ---------------------------------------------------------------------

def stable_unit_interval(key: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def assign_bucket(u: float, split) -> str:
    if u < split[0]:
        return "train"
    if u < split[0] + split[1]:
        return "val"
    return "test"


def spatial_block_id(lon: float, lat: float, block_deg: float) -> str:
    return f"{math.floor(lon / block_deg)}_{math.floor(lat / block_deg)}"


def item_key(path_or_name: str) -> str:
    return os.path.splitext(os.path.basename(path_or_name.strip()))[0]


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.asin(min(1.0, math.sqrt(a)))


def load_union_coords(data_roots, prefixes) -> dict:
    """{GEOID: (lon, lat)}, unioned across every given quarter's
    coords.json (a tract's centroid doesn't change by quarter, so this
    just gives the most complete picture of the state's tract layout
    from whatever's already been downloaded)."""
    coords = {}
    for data_root, prefix in zip(data_roots, prefixes):
        coords_path = os.path.join(data_root, f"{prefix}_coords.json")
        with open(coords_path) as f:
            raw = json.load(f)
        for k, (lon, lat) in raw.items():
            coords[item_key(k)] = (float(lon), float(lat))
    return coords


def nearest_neighbor_distance_km(geoid_coords: dict, sample_size: int, seed: int) -> list:
    """1-NN distance (km) for a sample of tracts, against all others --
    just a reference stat for calibrating candidate block sizes."""
    geoids = list(geoid_coords)
    rng = random.Random(seed)
    sample = geoids if len(geoids) <= sample_size else rng.sample(geoids, sample_size)
    dists = []
    for g in sample:
        lon0, lat0 = geoid_coords[g]
        best = None
        for other in geoids:
            if other == g:
                continue
            lon1, lat1 = geoid_coords[other]
            d = haversine_km(lat0, lon0, lat1, lon1)
            if best is None or d < best:
                best = d
        dists.append(best)
    return dists


def neighbor_cross_split_fraction(geoid_coords: dict, buckets: dict, k: int,
                                   sample_size: int, seed: int) -> float:
    """For a sample of tracts, the average fraction of each one's K
    nearest OTHER tracts that landed in a different split bucket."""
    geoids = list(geoid_coords)
    rng = random.Random(seed)
    sample = geoids if len(geoids) <= sample_size else rng.sample(geoids, sample_size)

    fractions = []
    for g in sample:
        lon0, lat0 = geoid_coords[g]
        dists = []
        for other in geoids:
            if other == g:
                continue
            lon1, lat1 = geoid_coords[other]
            dists.append((haversine_km(lat0, lon0, lat1, lon1), other))
        dists.sort(key=lambda t: t[0])
        neighbors = [g2 for _, g2 in dists[:k]]
        cross = sum(1 for g2 in neighbors if buckets[g2] != buckets[g])
        fractions.append(cross / len(neighbors))
    return statistics.mean(fractions)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-roots", nargs="+", required=True)
    p.add_argument("--prefixes", nargs="+", required=True)
    p.add_argument("--block-degs", nargs="+", type=float,
                    default=[0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
                    help="Candidate spatial_block_deg values to sweep (degrees)")
    p.add_argument("--split", nargs=3, type=float, default=[0.8, 0.1, 0.1])
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--k-neighbors", type=int, default=5,
                    help="K nearest tracts to check per sampled tract for the leakage-exposure metric")
    p.add_argument("--sample-size", type=int, default=300,
                    help="Tracts sampled for the (O(sample x n)) neighbor-distance computations -- "
                         "keeps this tractable for larger states without needing the full O(n^2)")
    args = p.parse_args()

    if len(args.data_roots) != len(args.prefixes):
        raise SystemExit(f"--data-roots ({len(args.data_roots)}) and --prefixes "
                          f"({len(args.prefixes)}) must have the same length")

    coords = load_union_coords(args.data_roots, args.prefixes)
    n = len(coords)
    print(f"Loaded {n} tracts (union across {len(args.data_roots)} quarter(s))\n")

    nn_dists = nearest_neighbor_distance_km(coords, args.sample_size, args.seed)
    print(f"Nearest-tract distance (km), sampled {len(nn_dists)} tracts:")
    print(f"  min={min(nn_dists):.2f}  p25={statistics.quantiles(nn_dists, n=4)[0]:.2f}  "
          f"median={statistics.median(nn_dists):.2f}  p75={statistics.quantiles(nn_dists, n=4)[2]:.2f}  "
          f"max={max(nn_dists):.2f}")
    print("(a spatial_block_deg whose cell size, in km, is well below this median is basically "
          "unblocked -- see the per-candidate table below for cell sizes)\n")

    no_protection_baseline = sum(f * f for f in args.split)
    print(f"No-blocking baseline: two independent hash draws with split={args.split} agree by "
          f"pure chance {no_protection_baseline*100:.1f}% of the time, so ~"
          f"{(1-no_protection_baseline)*100:.1f}% cross-split-neighbor fraction is the 'no real "
          f"protection' starting point below.\n")

    print(f"{'block_deg':>10} {'cell~km':>9} {'n_blocks':>9} {'tracts/block (min/med/max)':>28} "
          f"{'train%':>7} {'val%':>7} {'test%':>7} {'cross-split nbrs':>18}")
    print("-" * 100)

    for block_deg in args.block_degs:
        block_of = {g: spatial_block_id(lon, lat, block_deg) for g, (lon, lat) in coords.items()}
        buckets = {g: assign_bucket(stable_unit_interval(block_of[g], args.seed), args.split)
                   for g in coords}

        block_counts = Counter(block_of.values())
        sizes = sorted(block_counts.values())
        n_blocks = len(block_counts)

        bucket_counts = Counter(buckets.values())
        pct = {b: 100 * bucket_counts.get(b, 0) / n for b in ("train", "val", "test")}

        cross_frac = neighbor_cross_split_fraction(
            coords, buckets, k=args.k_neighbors, sample_size=args.sample_size, seed=args.seed
        )

        # Rough km size of one block_deg x block_deg cell, using the mean
        # latitude across loaded tracts (longitude degrees shrink in km
        # away from the equator; latitude degrees don't).
        mean_lat = statistics.mean(lat for _, lat in coords.values())
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * math.cos(math.radians(mean_lat))
        cell_km = block_deg * (km_per_deg_lat + km_per_deg_lon) / 2

        print(f"{block_deg:>10.3f} {cell_km:>9.1f} {n_blocks:>9} "
              f"{sizes[0]:>8}/{statistics.median(sizes):>4.0f}/{sizes[-1]:>5}          "
              f"{pct['train']:>6.1f}% {pct['val']:>6.1f}% {pct['test']:>6.1f}% "
              f"{cross_frac*100:>16.1f}%")

    print("\nRead this as: cross-split-neighbor % should fall from the no-blocking baseline "
          "above, toward 0%, as block_deg increases. Pick the smallest block_deg where it's "
          "leveled off -- going bigger past that point mainly costs you split accuracy "
          "(train/val/test %% drifting from --split) and block-size evenness (max tracts/block "
          "growing), without meaningfully reducing leakage further.")


if __name__ == "__main__":
    main()
