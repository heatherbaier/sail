"""
Generate a ready-to-launch sail `task: validate` config from --state/--year/
--quarter/--variable, matching an already-trained run produced by
generate_train_config.py -- instead of hand-copying a YAML file and making
sure its data_root/prefix/img_size/band_mean/band_std/model params and
experiment_name/output_dir line up exactly with that training run (get any
of those wrong and validate.py either loads the wrong checkpoint, the
wrong test_indices.txt, or silently re-derives band stats from a different
sample than training used).

Reuses generate_train_config.py's state_registry.yml lookup
(load_registry/resolve_state_settings) and data_root/prefix/experiment_name
construction directly (imported, not reimplemented) so a validate config
for a given state/year/quarter/variable is built from the exact same
per-state settings and naming convention as that run's train config --
they can't drift apart into pointing at different imagery or a different
band normalization.

Unlike generate_train_config.py's --version (which defaults to the next
UNUSED v<N>, since it's creating a new run), this defaults to the latest
EXISTING v<N> under that state/quarter/year/variable's output_dir -- the
run.py `task: validate` path needs an experiment_name whose
output_dir/experiment_name/ directory already has a checkpoint (and the
test_indices.txt that task: train wrote) in it, not an unused slot.

dataset.new is always written explicitly as False: sail.engine.run_validation
reads cfg["dataset"]["new"] with no default, and new: False is what makes
it evaluate on THIS run's own held-out test_indices.txt (new: True would
mean "validate against a completely different dataset", not what
--state/--year/--quarter/--variable here identify) -- pass --new to
override if you actually want that.

Also writes a matching SLURM job file, same as generate_train_config.py
(both run `python launch.py --config <path>` -- what actually happens is
controlled entirely by the config's task: field).

Usage:
    python generate_validate_config.py --state az --year 2016 --quarter 1 \
        --variable wealth_index_sat
    # writes configs/tlags/az/az_2016_q1_wealth_index_sat_validate.yml AND
    # .sh, pointed at the latest already-trained v<N> for that combo

    python generate_validate_config.py --state az --year 2016 --quarter 1 \
        --variable wealth_index_sat --version v1 --launch
    # validate a specific (not necessarily latest) trained version, and
    # immediately sbatch it
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_train_config import (  # noqa: E402
    REPO_ROOT, TARGET_CHOICES,
    load_registry, resolve_state_settings, write_job_file,
)


def latest_existing_version(output_dir: str, exp_base: str) -> str:
    """Largest v<N> already present as a directory under output_dir -- the
    opposite of generate_train_config.py's next_version(): validating
    needs the checkpoint (and test_indices.txt) a real `task: train` run
    already wrote there, not an unused slot."""
    existing = []
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            m = re.fullmatch(rf"{re.escape(exp_base)}_v(\d+)", name)
            if m:
                existing.append(int(m.group(1)))
    if not existing:
        raise SystemExit(
            f"No trained run found matching {exp_base}_v<N> under {output_dir} -- "
            f"run generate_train_config.py + task: train for this "
            f"state/year/quarter/variable first, or pass --version explicitly "
            f"if you know it exists under a different output_dir."
        )
    return f"v{max(existing)}"


def build_config(state, year, quarter, variable, version, new, device, registry):
    st = resolve_state_settings(state, registry)

    data_root = st["data_root_template"].format(state=state, year=year, quarter=quarter)
    base_prefix = st["base_prefix_template"].format(state=state, year=year, quarter=quarter)
    prefix = f"{base_prefix}_{variable}"

    data_root = data_root if data_root.endswith("/") else data_root + "/"
    output_dir = data_root + "artifacts/"

    exp_base = f"{state}_q{quarter}_{year}_{variable}"
    version = version or latest_existing_version(output_dir, exp_base)
    experiment_name = f"{exp_base}_{version}"

    cfg = {
        "task": "validate",
        "experiment_name": experiment_name,
        "output_dir": output_dir,
        "dataset": {
            "type": "json",
            "data_root": data_root,
            "prefix": prefix,
            "img_size": [256, 256],
            "num_workers": 0,
            "seed": 1337,
            "temporal": False,
            "new": new,
            "band_mean": st["band_mean"],
            "band_std": st["band_std"],
        },
        "model": {
            "name": "swin",
            "params": {"in_channels": st["in_channels"]},
        },
        "validator": {
            "device": device,
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
    p.add_argument("--version", default=None,
                    help="Which trained v<N> to validate. Default: the latest "
                         "one that already exists for this state/quarter/year/variable")
    p.add_argument("--new", action="store_true",
                    help="Validate against the FULL dataset at data_root/prefix instead "
                         "of this run's own held-out test_indices.txt (default: off -- "
                         "evaluate the model on its own test set)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None,
                    help="Default: configs/tlags/<state>/<state>_<year>_q<quarter>_<variable>_validate.yml")
    p.add_argument("--job-out", default=None, help="Default: same path as --out, with .sh instead of .yml")
    p.add_argument("--no-job-file", action="store_true", help="Skip writing the SLURM job file")
    p.add_argument("--nodes", default="1")
    p.add_argument("--gpus", default="1")
    p.add_argument("--cpus", default="8")
    p.add_argument("--walltime", default="72:00:00")
    p.add_argument("--partition", default="general")
    p.add_argument("--qos", default="grp_hbaier")
    p.add_argument("--conda-env", default="geomain")
    p.add_argument("--repo-dir", default="/home/hbaier/packages/sail/")
    p.add_argument("--launch", action="store_true",
                    help="Run `sbatch <job file>` immediately after writing it (requires the job file, i.e. not --no-job-file)")
    args = p.parse_args()

    registry = load_registry()
    cfg, experiment_name = build_config(
        args.state, args.year, args.quarter, args.variable,
        args.version, args.new, args.device, registry,
    )

    out_path = args.out or os.path.join(
        REPO_ROOT, "configs", "tlags", args.state,
        f"{args.state}_{args.year}_q{args.quarter}_{args.variable}_validate.yml"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"Wrote {out_path}")
    print(f"experiment_name: {experiment_name}")

    job_path = None
    if not args.no_job_file:
        job_path = args.job_out or os.path.splitext(out_path)[0] + ".sh"
        write_job_file(job_path, experiment_name, out_path, args.nodes, args.gpus,
                        args.cpus, args.walltime, args.partition, args.qos,
                        args.conda_env, args.repo_dir)
        print(f"Wrote {job_path}")
        print(f"\n  sbatch {job_path}\n")
    else:
        print(f"\n  python launch.py --config {out_path}\n")

    if args.launch:
        if job_path is None:
            raise SystemExit("--launch requires a job file -- drop --no-job-file")
        subprocess.run(["sbatch", job_path], check=True)


if __name__ == "__main__":
    main()
