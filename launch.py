#!/usr/bin/env python3
"""
Launch script for SAIL (Spatial AI Library)
Runs training / validation / explainability pipelines from YAML configs.
"""

import argparse
import os
import sys

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

    # sail.engine.run() loads the config itself, dispatches on task:, and
    # (as of this change) also copies it to output_dir/experiment_name/
    # config_used.yaml once the run completes -- see engine.py's run() for
    # why that lives there now instead of here: it's the one function both
    # this script and the installed `simba` CLI command call, so putting it
    # here would only have covered this entry point.
    print(f"🚀 Launching SAIL (config: {config_path})")
    run_engine(config_path)
    print("✅ Run complete.")


if __name__ == "__main__":
    main()
