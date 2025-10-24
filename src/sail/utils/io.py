import os
import csv
import json
import yaml
import pandas as pd
from datetime import datetime


def ensure_dir(path: str):
    """Create a directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)
    return path


def timestamp():
    """Return a simple timestamp string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_metrics_csv(metrics: dict, out_path: str):
    """
    Save a dictionary of metrics to CSV.
    If the file exists, append as a new row.
    """
    ensure_dir(os.path.dirname(out_path))
    df = pd.DataFrame([metrics])
    if os.path.exists(out_path):
        old = pd.read_csv(out_path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out_path, index=False)
    print(f"💾 Saved metrics to {out_path}")


def save_json(data: dict, out_path: str):
    """Save a dictionary to JSON with indentation."""
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"💾 Saved JSON to {out_path}")


def save_yaml(data: dict, out_path: str):
    """Save a dictionary to YAML."""
    ensure_dir(os.path.dirname(out_path))
    with open(out_path, "w") as f:
        yaml.safe_dump(data, f)
    print(f"💾 Saved YAML to {out_path}")


def load_yaml(path: str) -> dict:
    """Load YAML config from a path."""
    with open(path, "r") as f:
        return yaml.safe_load(f)
