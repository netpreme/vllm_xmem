"""Aggregate OSL / ISL / ISL_uncached distributions across ALL problems.

Same histograms as plot_dist_grid.py but collapsed into a single 1×3 row
(no per-difficulty split).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dataset import load_data
from plot_dist_grid import (
    CLAUDE_BASELINE_ISL,
    METRICS,
    REGIONS_ISL_NEW,
    REGIONS_OSL,
    XLIM_ALL,
    annotate_regions,
    hist_panel,
)

METRIC_COLOR = {"osl": "#3b82f6", "isl": "#ec4899", "isl_new": "#22c55e"}
OVERLAYS = {
    "osl": (REGIONS_OSL, None),
    "isl": (None, (CLAUDE_BASELINE_ISL, "Claude Code baseline (~27k)")),
    "isl_new": (REGIONS_ISL_NEW, None),
}
# Aggregate panels are taller than grid panels; bump the ISL_uncached
# legend off the bottom so it doesn't sit on the lowest bar.
LEGEND_OVERRIDES = {
    "isl_new": {"loc": "lower right", "bbox_to_anchor": (1.0, 0.30)},
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title-suffix", default="")
    args = ap.parse_args()

    t = load_data(args.run_dir)
    real = t[t["osl"] > 0]
    vals = {m: real[m].astype(float) for m, *_ in METRICS}

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))
    fig.suptitle(f"Token distributions\n{args.title_suffix}", fontsize=13)

    panel_max = []
    for ci, (m, label, default_legend) in enumerate(METRICS):
        regions, baseline = OVERLAYS[m]
        panel_max.append(
            hist_panel(
                axes[ci],
                vals[m],
                label,
                "",
                METRIC_COLOR[m],
                regions=regions,
                baseline=baseline,
                legend_kwargs=LEGEND_OVERRIDES.get(m, default_legend),
            )
        )
        if ci == 0:
            axes[ci].set_ylabel("Count")

    for ci, (m, *_) in enumerate(METRICS):
        regions = OVERLAYS[m][0]
        ymax = panel_max[ci] * 1.05
        axes[ci].set_ylim(0, ymax)
        if regions is not None:
            annotate_regions(axes[ci], vals[m], ymax, regions=regions)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
