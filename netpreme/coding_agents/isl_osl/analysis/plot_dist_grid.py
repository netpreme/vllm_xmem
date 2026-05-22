"""Per-difficulty OSL / ISL / ISL_uncached histogram grid.

3 rows (OSL / ISL / ISL_uncached) × N columns (one per difficulty bucket).
Each cell is a log-scaled histogram with median + mean overlays. The OSL
row gets semantic-region bands (tool-call / plan / code-edit) and the
ISL_uncached row gets its own bands; the ISL row gets a vertical line at
the claude-code first-turn baseline.

This module also exports the shared rendering helpers used by
plot_dist_agg.py (one row, no bucketing).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from data import VERIFIED_BUCKETS, load_data

# ----- styling constants ----------------------------------------------------

# White stroke on region labels so they stay legible over histogram bars.
LABEL_STROKE = [pe.withStroke(linewidth=3, foreground="white")]

XLIM_ALL = (1, 300_000)
COL_COLOR = ["#3b82f6", "#ec4899", "#22c55e"]

# OSL bands derived from joining each OSL bin to its dominant tool name
# in the Qwen × Verified × 500 dataset:
#   <100   : Read/Bash/Grep/Glob = ~97% of turns      → exploration
#   100-1k : ExitPlanMode/TodoWrite/short Edit         → plan / small action
#   1k-5k+ : Edit + Write = ~90% of turns              → code edit
REGIONS_OSL = [
    (1,      100,     "#dbeafe", "tool calls"),
    (100,    1_000,   "#fef9c3", "plan /\nsmall actions"),
    (1_000,  300_000, "#fce7f3", "code edits"),
]

# ISL_new = input + cache_creation: the tokens NOT served from cache.
#   ≤200   : 62% of turns       → small tool result (Bash exit, Grep match)
#   200-2k : 24% of turns       → typical Read result
#   2k-20k : 10% of turns       → large Read / multi-file dump
#   20k+   :  3% of turns       → turn-1 baseline + auto-compaction
REGIONS_ISL_NEW = [
    (1,        200,     "#dbeafe", "small\ntool result"),
    (200,      2_000,   "#fef9c3", "file read"),
    (2_000,    20_000,  "#fce7f3", "large read"),
    (20_000,   300_000, "#fee2e2", "system prompt /\ncompaction"),
]

# Median first-substantive-turn ISL across 500 Verified problems under
# Qwen3-Coder-30B = 26,975 tokens (p10=26,750, p90=27,597). Breakdown
# (estimated at ~4.17 chars/token from the actual data):
#   ~6,400 tokens  (24%)  system prompt (claude behavioral rules)
#   ~19,500 tokens (71%)  28 tool schemas — Bash/Read/Edit/Write/Glob/Grep/Task/…
#   ~1,800 tokens  (5%)   first user message (task statement)
# The tool catalog dominates; this is what every new conversation pays before
# the first byte of useful work. Marked as a vertical reference on the ISL panel.
CLAUDE_BASELINE_ISL = 27_000

METRICS = [
    ("osl",     "OSL",          {"loc": "lower right"}),
    ("isl",     "ISL",          {"loc": "upper left"}),
    ("isl_new", "ISL uncached", {"loc": "lower right",
                                  "bbox_to_anchor": (1.0, 0.12)}),
]


# ----- shared rendering helpers --------------------------------------------

def hist_panel(ax, vals: np.ndarray, metric_label: str, bucket_label: str,
               color: str,
               regions: list | None = None,
               baseline: tuple[float, str] | None = None,
               legend_kwargs: dict | None = None) -> int:
    """Render one log-scaled histogram cell. Returns max bin count so the
    caller can uniformize y-axes across a row.

    regions  : optional [(lo, hi, fill, txt), ...] colored x-bands.
    baseline : optional (x, label) vertical reference line.
    """
    vals = vals[vals > 0]
    if regions:
        for lo, hi, fill, _ in regions:
            ax.axvspan(lo, hi, color=fill, alpha=0.45, zorder=0)
    if not len(vals):
        title = f"{metric_label} — {bucket_label}" if bucket_label else metric_label
        ax.set_title(f"{title}\n(no data)"); return 0

    bins = np.logspace(np.log10(XLIM_ALL[0]), np.log10(XLIM_ALL[1]), 36)
    counts, _, _ = ax.hist(vals, bins=bins, color=color, edgecolor="white",
                           linewidth=0.4, zorder=2)
    med, mean = float(np.median(vals)), float(np.mean(vals))
    ax.axvline(med,  color="black", ls="--", lw=1.4, label=f"med={med:.0f}",  zorder=3)
    ax.axvline(mean, color="black", ls=":",  lw=1.2, label=f"mean={mean:.0f}", zorder=3)
    if baseline is not None:
        x, lbl = baseline
        ax.axvline(x, color="#dc2626", ls="-", lw=1.6,
                   label=lbl, zorder=3, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlim(*XLIM_ALL)
    title = f"{metric_label} — {bucket_label}" if bucket_label else metric_label
    ax.set_title(f"{title}\nn={len(vals):,} turns", fontsize=10, fontweight="bold")
    ax.legend(**(legend_kwargs or {"loc": "upper right"}),
              fontsize=9, framealpha=0.95)
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    return int(counts.max()) if len(counts) else 0


def annotate_regions(ax, vals: np.ndarray, ymax: float, regions: list) -> None:
    """Bold region label + n=X (Y%) count inside each band, with a white
    text-stroke so bars underneath stay visible but text reads clearly."""
    for lo, hi, _fill, txt in regions:
        n = int(((vals >= lo) & (vals < hi)).sum())
        pct = 100 * n / len(vals) if len(vals) else 0
        cx = float(np.sqrt(lo * hi))
        ax.text(cx, ymax * 0.95, txt,
                ha="center", va="top",
                fontsize=8, color="#1f2937", fontweight="bold",
                linespacing=1.1, zorder=5, path_effects=LABEL_STROKE)
        ax.text(cx, ymax * 0.78, f"n={n}\n({pct:.0f}%)",
                ha="center", va="top",
                fontsize=7, color="#4b5563",
                zorder=5, path_effects=LABEL_STROKE)


# ----- main -----------------------------------------------------------------

OVERLAYS = {
    "osl":     (REGIONS_OSL,     None),
    "isl":     (None,            (CLAUDE_BASELINE_ISL, "Claude Code baseline (~27k)")),
    "isl_new": (REGIONS_ISL_NEW, None),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",      required=True, type=Path)
    ap.add_argument("--out",          required=True, type=Path)
    ap.add_argument("--title-suffix", default="")
    args = ap.parse_args()

    t = load_data(args.run_dir)
    real = t[t["category"] != "empty"]
    # Group values by (metric, difficulty bucket).
    vals: dict[tuple[str, str], np.ndarray] = {}
    for m, *_ in METRICS:
        for b in VERIFIED_BUCKETS:
            sel = real["difficulty"] == b
            vals[(m, b)] = real[sel][m].astype(float)

    n_rows, n_cols = len(METRICS), len(VERIFIED_BUCKETS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.2 * n_rows))
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(
        f"OSL / ISL / ISL_new distributions by difficulty (log-binned){suffix}",
        fontsize=14,
    )

    # First pass: histograms + per-row max count.
    row_max = [0] * n_rows
    for ri, (m, label, legend) in enumerate(METRICS):
        regions, baseline = OVERLAYS[m]
        for ci, b in enumerate(VERIFIED_BUCKETS):
            mx = hist_panel(axes[ri, ci], vals[(m, b)], label, b,
                            COL_COLOR[ci % len(COL_COLOR)],
                            regions=regions, baseline=baseline,
                            legend_kwargs=legend)
            row_max[ri] = max(row_max[ri], mx)
            if ci == 0:
                axes[ri, ci].set_ylabel(f"{label}\nCount")
            if ri == n_rows - 1:
                axes[ri, ci].set_xlabel("tokens (log scale)")

    # Second pass: uniformize per-row y-axis, annotate regions.
    for ri, (m, *_) in enumerate(METRICS):
        regions = OVERLAYS[m][0]
        ymax = row_max[ri] * 1.05
        for ci, b in enumerate(VERIFIED_BUCKETS):
            axes[ri, ci].set_ylim(0, ymax)
            if regions is not None:
                annotate_regions(axes[ri, ci], vals[(m, b)], ymax, regions)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
