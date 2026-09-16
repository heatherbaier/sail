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

Also writes a matching SLURM job file (.sh, next to the .yml) that runs
`python launch.py --config <that config>`. The SLURM directives (nodes,
GPU/CPU count, walltime, partition/QOS, module load, conda env, working
directory) are a fixed template taken from a real job file that's already
worked on ASU RC -- only the job name, config path, and output/error
filenames vary per run, since guessing at cluster-specific resource
settings is a good way to submit a job that fails outright or burns real
allocation for nothing. --conda-env/--repo-dir/--nodes/--gpus/--cpus/
--walltime/--partition/--qos override the template's defaults if your
setup ever changes.

Usage:
    python generate_train_config.py --state az --year 2016 --quarter 1 \
        --variable wealth_index_sat
    # writes configs/tlags/az/az_2016_q1_wealth_index_sat_train.yml AND
    # az_2016_q1_wealth_index_sat_train.sh, auto-picking the next unused
    # version (v1, v2, ...)

    python generate_train_config.py --state az --year 2016 --quarter 1 \
        --variable wealth_index_sat --launch
    # generates both files AND immediately runs `sbatch <job file>`
"""

import argparse
import os
import subprocess

import yaml

JOB_TEMPLATE = """#!/bin/bash
#SBATCH -N {nodes}            # number of nodes
#SBATCH -G {gpus}
#SBATCH -c {cpus}            # number of cores
#SBATCH -t {walltime}   # time in d-hh:mm:ss
#SBATCH -p {partition}      # partition
#SBATCH -q {qos}       # QOS
#SBATCH -J {experiment_name}
#SBATCH -o slurm.{experiment_name}.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e slurm.{experiment_name}.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user="%u@asu.edu"
#SBATCH --export=NONE   # Purge the job-submitting shell environment

#Load required software
module load mamba/latest

#Activate our environment
source activate {conda_env}

#Change to the directory of our script
cd {repo_dir}

#Run the software/python script
python launch.py --config {config_path}
"""

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


def write_job_file(job_path, experiment_name, config_path, nodes, gpus, cpus,
                    walltime, partition, qos, conda_env, repo_dir):
    content = JOB_TEMPLATE.format(
        nodes=nodes, gpus=gpus, cpus=cpus, walltime=walltime,
        partition=partition, qos=qos, experiment_name=experiment_name,
        conda_env=conda_env, repo_dir=repo_dir, config_path=config_path,
    )
    with open(job_path, "w") as f:
        f.write(content)


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
