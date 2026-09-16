"""
Generate a ready-to-launch sail train config from --state/--year/--quarter/
--variable, instead of hand-editing a YAML file (and re-copying seed/split/
split_strategy/spatial_block_deg/band_mean/band_std by hand) for every run.

Per-state settings that must stay IDENTICAL across every quarter/year/
variable for that state (spatial_block_deg, band_mean, band_std, the
data_root/prefix naming convention) come from scripts/state_registry.yml
(kept next to this script, not under configs/ -- configs/ is gitignored
in this repo, since per-run generated/hand-written train configs are
local-only, but the registry is hand-maintained shared metadata that
should be version-controlled) -- fill that in once per state, using the real output of
scripts/find_spatial_block_deg.py and scripts/compute_shared_band_stats.py
(not guesses), and every config generated for that state reuses those
exact values automatically. That's what actually removes the manual-
copy-paste error risk -- the CLI args below only vary the things that are
SUPPOSED to vary per run (state, year, quarter, target variable).

experiment_name always encodes state+quarter+year+variable, so training
wealth_index_sat and wealth_index_housing_core for the same state/quarter
never collide and silently overwrite each other's checkpoints in the same
ckpt_dir. --version defaults to auto-detecting the next unused v<N> under
that state/quarter/variable's output_dir, so you don't have to track and
bump it by hand either.

Usage:
    python generate_train_config.py --state az --year 2016 --quarter 1 \
        --variable wealth_index_sat
    # writes configs/tlags/az/az_2016_q1_wealth_index_sat_train.yml,
    # auto-picking the next unused version (v1, v2, ...)

    python generate_train_config.py --state az --year 2016 --quarter 1 \
        --variable wealth_index_sat --launch
    # generates the config AND immediately runs `python launch.py --config ...`
"""

import argparse
import os
import subprocess
import sys

import yaml

TARGET_CHOICES = [
    "wealth_index",
    "wealth_index_sat",
    "wealth_index_housing_core",
    "wealth_index_transport",
    "wealth_index_financial",
    "wealth_index_utilities",
]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state_registry.yml")


def load_registry(path=REGISTRY_PATH):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_state_settings(state, registry, registry_path=REGISTRY_PATH):
    if state not in registry:
        raise SystemExit(f"--state {state!r} not in {registry_path} "
                          f"(known states: {sorted(registry)})")
    st = registry[state]
    required = ["data_root_template", "base_prefix_template", "spatial_block_deg",
                "band_mean", "band_std", "in_channels"]
    missing = [k for k in required if st.get(k) is None]
    if missing:
        raise SystemExit(
            f"--state {state!r} is missing required settings in {registry_path}: "
            f"{missing}. Run find_spatial_block_deg.py / compute_shared_band_stats.py "
            f"for {state} and fill these in before generating configs for it."
        )
    return st


def next_version(output_dir: str, exp_base: str) -> str:
    """Smallest v<N> not already present as a directory under output_dir."""
    n = 1
    while os.path.isdir(os.path.join(output_dir, f"{exp_base}_v{n}")):
        n += 1
    return f"v{n}"


def build_config(state, year, quarter, variable, epochs, lr, batch_size, version, registry):
    st = resolve_state_settings(state, registry)

    data_root = st["data_root_template"].format(state=state, year=year, quarter=quarter)
    base_prefix = st["base_prefix_template"].format(state=state, year=year, quarter=quarter)
    prefix = f"{base_prefix}_{variable}"

    data_root = data_root if data_root.endswith("/") else data_root + "/"
    output_dir = data_root + "artifacts/"

    exp_base = f"{state}_q{quarter}_{year}_{variable}"
    version = version or next_version(output_dir, exp_base)
    experiment_name = f"{exp_base}_{version}"

    cfg = {
        "task": "train",
        "experiment_name": experiment_name,
        "output_dir": output_dir,
        "dataset": {
            "type": "json",
            "data_root": data_root,
            "prefix": prefix,
            "batch_size": batch_size,
            "img_size": [256, 256],
            "num_workers": 0,
            "seed": 1337,
            "temporal": False,
            "write_files": True,
            "split": [0.8, 0.1, 0.1],
            "split_strategy": "stable",
            "spatial_block_deg": st["spatial_block_deg"],
            "band_mean": st["band_mean"],
            "band_std": st["band_std"],
        },
        "model": {
            "name": "swin",
            "params": {"in_channels": st["in_channels"]},
        },
        "trainer": {
            "epochs": epochs,
            "lr": lr,
            "device": "cuda",
            "eval_every": 1,
            "early_stop": None,
        },
    }
    return cfg, experiment_name


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--state", required=True)
    p.add_argument("--year", required=True, type=int)
    p.add_argument("--quarter", required=True, type=int, choices=[1, 2, 3, 4])
    p.add_argument("--variable", required=True, choices=TARGET_CHOICES)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.00001)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--version", default=None,
                    help="Default: auto-detect the next unused v<N> for this "
                         "state/quarter/year/variable combo")
    p.add_argument("--out", default=None, help="Default: configs/tlags/<state>/<state>_<year>_q<quarter>_<variable>_train.yml")
    p.add_argument("--launch", action="store_true",
                    help="Run `python launch.py --config <generated>` immediately after writing it")
    args = p.parse_args()

    registry = load_registry()
    cfg, experiment_name = build_config(
        args.state, args.year, args.quarter, args.variable,
        args.epochs, args.lr, args.batch_size, args.version, registry,
    )

    out_path = args.out or os.path.join(
        REPO_ROOT, "configs", "tlags", args.state,
        f"{args.state}_{args.year}_q{args.quarter}_{args.variable}_train.yml"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Wrote {out_path}")
    print(f"experiment_name: {experiment_name}")
    print(f"\n  python launch.py --config {out_path}\n")

    if args.launch:
        launch_py = os.path.join(REPO_ROOT, "launch.py")
        subprocess.run([sys.executable, launch_py, "--config", out_path], check=True)


if __name__ == "__main__":
    main()
