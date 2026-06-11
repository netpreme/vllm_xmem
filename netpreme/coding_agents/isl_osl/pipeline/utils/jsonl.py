"""Append-only per-problem JSONL writer (shared by the scraper + proxy).

Each problem gets its own folder ``<telemetry_dir>/<instance_id>/`` holding
``vllm_metrics.jsonl`` (scraper), ``vllm_traces.jsonl`` (proxy raw capture) and
``session_config.json`` (runner).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

_SAFE_FILE_RE = re.compile(r"[^A-Za-z0-9_\-.]")


def instance_dir(telemetry_dir: Path, instance_id: str) -> Path:
    """The per-problem folder ``<telemetry_dir>/<instance_id>/``, with the id
    made filesystem-safe (used as the folder name)."""
    return telemetry_dir / _SAFE_FILE_RE.sub("_", instance_id)[:200]


class JsonlWriter:
    """Append-only writer: one ``<telemetry_dir>/<instance_id>/<filename>`` per
    id. `filename` is e.g. ``vllm_metrics.jsonl`` (scraper) or ``vllm_traces.jsonl`` (proxy).
    """

    def __init__(self, telemetry_dir: Path, filename: str) -> None:
        self._dir = telemetry_dir
        self._filename = filename

    def write(self, instance_id: str, row: dict) -> None:
        if not instance_id:
            return
        problem_dir = instance_dir(self._dir, instance_id)
        problem_dir.mkdir(parents=True, exist_ok=True)
        with (problem_dir / self._filename).open("a") as fh:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
