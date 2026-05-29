"""SWE-bench dataset + run-directory helpers.

User-facing functions take primitive types only (str, int, Path). They
don't depend on argparse Namespaces or on each other.
"""

from __future__ import annotations

import json
import random as _random
from datetime import datetime
from pathlib import Path
from datasets import load_dataset

HERE = Path(__file__).resolve().parent.parent  # .../isl_osl


def setup_run_dir(stamp: str | None = None) -> Path:
    """Create runs/<stamp>/ (fresh timestamp if stamp is None). Idempotent
    on existing directories — safe to call for a resume."""
    if stamp is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = HERE / "runs" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def get_dataset(
    name: str, out_path: Path, num_problems: int = 500, random: int = 0, seed: int = 0
) -> list[dict]:
    """Return SWE-bench problems as a list of dicts.

    Downloads `name` from HuggingFace into `out_path` (one JSON object
    per line) the first time. Reuses the file on subsequent calls, which
    makes resuming an interrupted run a natural no-op."""
    if not out_path.exists():
        rows = list(load_dataset(name, split="test"))
        if random:
            _random.Random(seed).shuffle(rows)
            rows = rows[:random]
        else:
            rows = rows[:num_problems]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]


def pending_problems(dataset: list[dict], solved_path: Path) -> list[dict]:
    """Drop any problem whose instance_id is already in `solved_path`."""
    if not solved_path.exists():
        return list(dataset)
    solved = {x for x in solved_path.read_text().split() if x}
    return [p for p in dataset if p["instance_id"] not in solved]
