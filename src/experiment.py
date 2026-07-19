"""Utilities for reproducible, non-overwriting experiment records."""

from datetime import datetime, timezone
import csv
import hashlib
import json
import platform
from pathlib import Path

import torch


def make_run_id(config: dict) -> str:
    """Return a readable UTC timestamp plus a stable config fingerprint."""
    canonical = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{digest}"


def save_run_record(record: dict, output_dir: str = "outputs/runs") -> str:
    """Persist one run without overwriting earlier experiment results."""
    run_id = record.get("run_id") or make_run_id(record.get("config", {}))
    path = Path(output_dir) / f"{run_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = 2
    while path.exists():
        path = Path(output_dir) / f"{run_id}-{suffix}.json"
        suffix += 1
    path.write_text(json.dumps({**record, "run_id": run_id}, indent=2), encoding="utf-8")
    return str(path)


def environment_info() -> dict:
    """Capture the main runtime versions needed to reproduce a run."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }


def append_results_csv(run_id: str, config: dict, rows: list,
                       path: str = "outputs/results.csv") -> str:
    """Append tidy model results for spreadsheet-friendly analysis."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fields = ["run_id", "model", "accuracy", "macro_f1", "parameters",
              "subset_fraction", "epochs", "seed"]
    exists = target.exists()
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({
                "run_id": run_id, **row,
                "subset_fraction": config["subset_fraction"],
                "epochs": config["num_epochs"], "seed": config["seed"],
            })
    return str(target)
