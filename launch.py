#!/usr/bin/env python3
"""
Launch script for SAIL (Spatial AI Library)
Runs training / validation / explainability pipelines from YAML configs.
"""

import argparse
import os
import sys
import yaml

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from sail.engine import run as run_engine  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Run SAIL pipeline from config")
    parser.add_argument(
        "--config",
        "-c",
        required=True,
        help="Path to YAML configuration file (e.g., configs/phl_geoconv_regression.yaml)",
    )
    args = parser.parse_args()

    config_path = os.path.abspath(args.config)

    # Optional: copy config into output_dir for reproducibility
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg.get("output_dir", "artifacts/checkpoints/default_run")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_used.yaml"), "w") as f:
        yaml.safe_dump(cfg, f)

    print(f"🚀 Launching SAIL task: {cfg.get('task', 'train')}  (config: {config_path})")
    run_engine(config_path)
    print("✅ Run complete.")


if __name__ == "__main__":
    main()
