"""Utilities for reproducible, non-overwriting experiment records."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


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
    path.write_text(json.dumps({**record, "run_id": run_id}, indent=2), encoding="utf-8")
    return str(path)
