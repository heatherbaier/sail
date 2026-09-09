"""
Verify that JSONGeoAdapter(split_strategy="stable") is actually giving the
same train/val/test bucket to the same tract across multiple quarters/years
-- i.e. sanity-check the guarantee data/splitting.py is supposed to provide,
against real training runs, not just the synthetic tests we ran earlier.

Why this needs to reconstruct "train" itself rather than just diffing
val_indices.txt/test_indices.txt: those two files alone have a blind spot.
A GEOID present in Q1's val_indices.txt but absent from Q2's val_indices.txt
AND Q2's test_indices.txt is consistent with two very different situations
that look identical from those two files alone -- (a) that GEOID's chip
just doesn't exist in Q2 (fine, expected), or (b) it DOES exist in Q2's
dataset but got assigned to Q2's train set (a real bug). You can't tell
those apart without knowing Q2's true train membership too.

sail doesn't currently persist a train_indices.txt anywhere, so this
reconstructs each quarter's true item universe itself, straight from the
same files SimbaJSONDataset reads (<prefix>_ys.json ∩ <prefix>_coords.json),
and derives train = full_universe - val - test. That gives a genuine 3-way
comparison with no blind spot, without needing any change to sail itself.

Usage:
    python check_split_consistency.py \
        --labels q1_2016 q2_2016 \
        --ckpt-dirs /data/hbaier/new_data/tlag/az_imagery/q1_2016_s2_allbands/artifacts/az_q1_2016_allbands_v1 \
                    /data/hbaier/new_data/tlag/az_imagery/q2_2016_s2_allbands/artifacts/az_q2_2016_allbands_v1 \
        --data-roots /data/hbaier/new_data/tlag/az_imagery/q1_2016_s2_allbands/ \
                     /data/hbaier/new_data/tlag/az_imagery/q2_2016_s2_allbands/ \
        --prefixes az_2016_q1_s2_allbands az_2016_q2_s2_allbands

(Works for any number of runs, not just two -- add more entries to each
--labels/--ckpt-dirs/--data-roots/--prefixes list as you train more
quarters, and every pair gets cross-checked.)
"""

import argparse
import json
import os


def item_key(path_or_name: str) -> str:
    """Same normalization sail's data/splitting.py uses: filename stem, not
    full path -- different quarters point data_root at different
    directories, so the full path differs even for the same tract."""
    return os.path.splitext(os.path.basename(path_or_name.strip()))[0]


def load_indices_file(path: str) -> set:
    if not os.path.exists(path):
        print(f"  WARNING: {path} not found -- treating as empty")
        return set()
    with open(path) as f:
        return {item_key(line) for line in f if line.strip()}


def load_full_universe(data_root: str, prefix: str) -> set:
    """Reconstructs exactly what SimbaJSONDataset.__init__ computes as
    self.items: the intersection of ys.json and coords.json keys."""
    ys_path = os.path.join(data_root, f"{prefix}_ys.json")
    coords_path = os.path.join(data_root, f"{prefix}_coords.json")
    with open(ys_path) as f:
        ys = json.load(f)
    with open(coords_path) as f:
        coords = json.load(f)
    keys = set(ys) & set(coords)
    return {item_key(k) for k in keys}


def bucket_of(geoid: str, run: dict) -> str:
    if geoid in run["train"]:
        return "train"
    if geoid in run["val"]:
        return "val"
    if geoid in run["test"]:
        return "test"
    return "??"  # shouldn't happen -- geoid came from run["full"] itself


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", nargs="+", required=True, help="Short name per run, e.g. q1_2016 q2_2016")
    p.add_argument("--ckpt-dirs", nargs="+", required=True, help="output_dir/experiment_name per run (holds val_indices.txt/test_indices.txt)")
    p.add_argument("--data-roots", nargs="+", required=True, help="dataset.data_root per run (holds <prefix>_ys.json/_coords.json)")
    p.add_argument("--prefixes", nargs="+", required=True, help="dataset.prefix per run")
    args = p.parse_args()

    n = len(args.labels)
    if not (len(args.ckpt_dirs) == len(args.data_roots) == len(args.prefixes) == n):
        raise SystemExit(
            f"--labels ({n}), --ckpt-dirs ({len(args.ckpt_dirs)}), "
            f"--data-roots ({len(args.data_roots)}), and --prefixes "
            f"({len(args.prefixes)}) must all have the same length -- one "
            f"entry per run, in the same order."
        )

    runs = {}
    for label, ckpt_dir, data_root, prefix in zip(args.labels, args.ckpt_dirs, args.data_roots, args.prefixes):
        val = load_indices_file(os.path.join(ckpt_dir, "val_indices.txt"))
        test = load_indices_file(os.path.join(ckpt_dir, "test_indices.txt"))
        full = load_full_universe(data_root, prefix)

        overlap = val & test
        if overlap:
            print(f"[{label}] WARNING: {len(overlap)} GEOIDs appear in BOTH "
                  f"val_indices.txt and test_indices.txt -- shouldn't happen "
                  f"with a genuine 3-way split; check this run's split config.")

        train = full - val - test
        runs[label] = {"full": full, "train": train, "val": val, "test": test}
        print(f"[{label}] full={len(full)}  train={len(train)}  val={len(val)}  test={len(test)}")

    print("\n" + "=" * 78)
    print("Pairwise cross-run consistency")
    print("(every GEOID present in BOTH runs' full item universe must land")
    print(" in the identical train/val/test bucket in both -- 100% or bust,")
    print(" not just 'high agreement': two INDEPENDENT random splits with a")
    print(" 0.8/0.1/0.1 ratio would still agree by pure chance on")
    print(" 0.8^2+0.1^2+0.1^2 = 66% of shared items, so a wrong seed/config")
    print(" looks like 'mostly working' rather than 'obviously broken' if")
    print(" you're only glancing at agreement rate instead of exact count.)")
    print("=" * 78)

    labels = args.labels
    any_mismatch = False
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            shared = runs[a]["full"] & runs[b]["full"]
            mismatches = [
                (geoid, bucket_of(geoid, runs[a]), bucket_of(geoid, runs[b]))
                for geoid in shared
            ]
            mismatches = [m for m in mismatches if m[1] != m[2]]

            n_shared = len(shared)
            n_mismatch = len(mismatches)
            pct = 100 * (1 - n_mismatch / n_shared) if n_shared else float("nan")
            status = "OK -- exact match" if n_mismatch == 0 else f"MISMATCH -- {n_mismatch}/{n_shared} disagree ({pct:.1f}% agreement)"
            print(f"\n{a} vs {b}: {n_shared} GEOIDs present in both -- {status}")
            if mismatches:
                any_mismatch = True
                print("  First 20 mismatches (GEOID: bucket_in_a  bucket_in_b):")
                for geoid, ba, bb in mismatches[:20]:
                    print(f"    {geoid}: {a}={ba}  {b}={bb}")

    print("\n" + "=" * 78)
    if any_mismatch:
        print("RESULT: inconsistencies found. Double-check every run listed above "
              "used the identical seed / split / split_strategy / spatial_block_deg "
              "in its train config -- a mismatch on any one of those silently "
              "produces an unrelated split for that run.")
        raise SystemExit(1)
    else:
        print("RESULT: fully consistent across all runs checked -- every GEOID "
              "present in more than one run landed in the identical bucket "
              "every time.")


if __name__ == "__main__":
    main()
