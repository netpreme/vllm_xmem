"""Shared data-loading helpers for analysis/plot_*.py.

Every plot script reads `runs/<stamp>/per_problem/*.csv` (the turn-level
metrics) and optionally `problems.jsonl` (for per-problem fields like
difficulty). Centralizing those reads keeps each plot script focused on
rendering.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


# SWE-bench Verified difficulty buckets. The ">4 hours" bucket has only ~3
# problems in the 500-problem split, too few to be meaningful, so we fold
# it into "1+ hours" together with "1-4 hours".
VERIFIED_BUCKETS = ["<15 min fix", "15 min - 1 hour", "1+ hours"]
DIFFICULTY_REMAP = {">4 hours": "1+ hours", "1-4 hours": "1+ hours"}


def load_data(run_dir: Path) -> np.ndarray:
    """Load the canonical per-turn structured array from <run-dir>/data.npz.
    Build it first with `python build_data.py --run-dir <run-dir>`."""
    return np.load(run_dir / "data.npz")["turns"]


def num(row: dict, key: str) -> float:
    """Coerce a CSV cell to float; return NaN for missing/blank/non-numeric."""
    v = row.get(key)
    if v in (None, "", "None"):
        return math.nan
    try:
        return float(v)
    except ValueError:
        return math.nan


def load_per_problem_rows(run_dir: Path) -> dict[str, list[dict]]:
    """Read every <run_dir>/per_problem/*.csv into {instance_id: [rows]}."""
    out: dict[str, list[dict]] = {}
    for f in sorted((run_dir / "per_problem").glob("*.csv")):
        with f.open() as fh:
            out[f.stem] = list(csv.DictReader(fh))
    return out


def load_all_rows(run_dir: Path) -> list[dict]:
    """Flatten per-problem CSVs into one list of turn rows."""
    return [r for rows in load_per_problem_rows(run_dir).values() for r in rows]


def load_problem_field(run_dir: Path, field: str) -> dict[str, str]:
    """Pull `<field>` from problems.jsonl, keyed by instance_id.
    Lists are joined with commas; 'difficulty' is run through DIFFICULTY_REMAP."""
    out: dict[str, str] = {}
    for line in (run_dir / "problems.jsonl").open():
        r = json.loads(line)
        v = r.get(field)
        if isinstance(v, list):
            v = ",".join(map(str, v))
        if field == "difficulty":
            v = DIFFICULTY_REMAP.get(v, v)
        out[r["instance_id"]] = v
    return out
