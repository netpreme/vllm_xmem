"""The analysis pass: build the dataset, render the figures.

``run(save_dir)`` builds ``<save_dir>/data.npz`` in-process (via metrics)
and renders every figure into ``<save_dir>/analysis/``. Importable by the
runner, or standalone:  ``python analysis/report.py <save_dir>``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

# So `from metrics import ...` resolves however we're imported/invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import build_records

HERE = Path(__file__).resolve().parent

# (plot script, output filename) — each is its own CLI script.
_PLOTS = [
    ("plot_dist_agg.py", "analysis_dist_agg.png"),
    ("plot_turns.py", "analysis_turns.png"),
    ("plot_cache_hit.py", "analysis_cache.png"),
    ("plot_latency_model.py", "analysis_latency_model.png"),
]


def run(save_dir: Path) -> None:
    """Build data.npz from the run's raw captures, then render all figures."""
    save_dir = Path(save_dir)
    out = save_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    n = len(list((save_dir / "telemetry").glob("*/meta.json")))
    suffix = f"{n} problems"

    # 1. Canonical per-turn dataset (in-process).
    turns = build_records(save_dir)
    np.savez_compressed(save_dir / "data.npz", turns=turns)
    print(f"[report] {len(turns):,} turns from {n} problems → {out}")

    # 2. Figures.
    for script, fname in _PLOTS:
        _plot(
            script=script,
            save_dir=save_dir,
            args=["--out", str(out / fname), "--title-suffix", suffix],
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("save_dir", type=Path)
    run(ap.parse_args().save_dir)
    return 0


def _plot(script: str, save_dir: Path, args: list[str]) -> None:
    print(f"  -> {script}")
    subprocess.run(
        [sys.executable, str(HERE / script), "--save-dir", str(save_dir), *args],
        check=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
