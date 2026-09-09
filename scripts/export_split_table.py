"""
Write one row per tract -- GEOID, image path, split (train/val/test), lon,
lat -- to a CSV in the run's artifacts dir, so the train/val/test split
can be checked visually: turn it into a point GeoDataFrame directly from
lon/lat, or join it back onto the original tract shapefile by GEOID, and
color by split.

Reconstructs "train" the same way scripts/check_split_consistency.py
does, since sail doesn't persist a train_indices.txt anywhere: the full
item universe comes from <prefix>_ys.json ∩ <prefix>_coords.json (the
same files SimbaJSONDataset itself reads), and anything in that universe
not listed in val_indices.txt or test_indices.txt is train.

Usage:
    python export_split_table.py \
        --ckpt-dir /data/hbaier/new_data/tlag/az_imagery/q1_2016_s2_allbands/artifacts/az_q1_2016_allbands_v2 \
        --data-root /data/hbaier/new_data/tlag/az_imagery/q1_2016_s2_allbands/ \
        --prefix az_2016_q1_s2_allbands
    # writes <ckpt-dir>/split_table.csv by default; pass --out to override

Then, e.g.:
    import pandas as pd, geopandas as gpd
    df = pd.read_csv(".../split_table.csv")
    gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs="EPSG:4326")
    gdf.plot(column="split", categorical=True, legend=True)
"""

import argparse
import csv
import json
import os


def item_key(path_or_name: str) -> str:
    return os.path.splitext(os.path.basename(path_or_name.strip()))[0]


def load_indices_file(path: str) -> set:
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {item_key(line) for line in f if line.strip()}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", required=True, help="output_dir/experiment_name (holds val_indices.txt/test_indices.txt)")
    p.add_argument("--data-root", required=True, help="dataset.data_root (holds <prefix>_ys.json/_coords.json)")
    p.add_argument("--prefix", required=True, help="dataset.prefix")
    p.add_argument("--out", default=None, help="Output CSV path (default: <ckpt-dir>/split_table.csv)")
    args = p.parse_args()

    ys_path = os.path.join(args.data_root, f"{args.prefix}_ys.json")
    coords_path = os.path.join(args.data_root, f"{args.prefix}_coords.json")
    with open(ys_path) as f:
        ys = json.load(f)
    with open(coords_path) as f:
        coords = json.load(f)
    full_paths = sorted(set(ys) & set(coords))

    val_geoids = load_indices_file(os.path.join(args.ckpt_dir, "val_indices.txt"))
    test_geoids = load_indices_file(os.path.join(args.ckpt_dir, "test_indices.txt"))
    overlap = val_geoids & test_geoids
    if overlap:
        print(f"NOTE: {len(overlap)} GEOIDs are in both val_indices.txt and test_indices.txt "
              f"(this run used a 2-way split -- test is an alias for val, not a separate set).")

    out_path = args.out or os.path.join(args.ckpt_dir, "split_table.csv")
    n_by_split = {"train": 0, "val": 0, "test": 0}
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["GEOID", "image_path", "split", "lon", "lat"])
        for path in full_paths:
            geoid = item_key(path)
            if geoid in val_geoids:
                split = "val"
            elif geoid in test_geoids:
                split = "test"
            else:
                split = "train"
            lon, lat = coords[path]
            w.writerow([geoid, path, split, lon, lat])
            n_by_split[split] += 1

    print(f"Wrote {len(full_paths)} rows to {out_path}")
    print(f"  train={n_by_split['train']}  val={n_by_split['val']}  test={n_by_split['test']}")
    print("\nLoad with e.g.:")
    print("  import pandas as pd, geopandas as gpd")
    print(f"  df = pd.read_csv({out_path!r})")
    print("  gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat), crs='EPSG:4326')")
    print("  gdf.plot(column='split', categorical=True, legend=True)")


if __name__ == "__main__":
    main()
