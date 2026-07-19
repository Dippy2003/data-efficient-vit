"""Utilities for reproducible, non-overwriting experiment records."""

from datetime import datetime, timezone
import hashlib
import json


def make_run_id(config: dict) -> str:
    """Return a readable UTC timestamp plus a stable config fingerprint."""
    canonical = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{digest}"
