#!/usr/bin/env python3
"""
3-row by 4-column histogram grid:
  Row 1: OSL distribution per difficulty bucket
  Row 2: ISL distribution per difficulty bucket
  Row 3: ISL_new (uncached) distribution per difficulty bucket

Each cell is a log-scaled histogram with median (dashed) and mean (dotted)
overlays. The OSL row also gets the semantic region bands (tool-call JSON /
text+tool / Edit-Write / large Write).

Usage:
  python3 plot_dist_grid.py \
      --run-dir runs/20260513_175826 \
      --out ~/analysis_dist_grid.png \
      --title-suffix "claude × Verified"

For Pro (no `difficulty` column):
  python3 plot_dist_grid.py --run-dir <pro_dir> --out ~/grid_pro.png \
      --bucket repo_language --buckets go python js ts
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

VERIFIED_BUCKETS = ["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours"]
COL_COLOR = ["#3b82f6", "#ec4899", "#22c55e", "#a855f7"]
# Semantic regions shared across all panels (OSL-style; same labels everywhere).
REGIONS = [
    (1,        100,    "#dbeafe", "tool call JSON"),
    (100,      1_000,  "#fef9c3", "text + tool"),
    (1_000,    5_000,  "#fce7f3", "Edit/Write"),
    (5_000,    50_000, "#dcfce7", "large Write"),
    (50_000,   300_000,"#e0e7ff", "huge context"),
]
# Uniform x-range across the entire figure so all panels are visually comparable.
XLIM_ALL = (1, 300_000)
METRICS = [
    ("osl",     "OSL"),
    ("isl",     "ISL"),
    ("isl_new", "ISL_new (uncached)"),
]


def num(r: dict, k: str) -> float:
    v = r.get(k)
    if v in (None, "", "None"):
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def load_bucket_map(run_dir: Path, field: str) -> dict[str, str]:
    out = {}
    for line in (run_dir / "problems.jsonl").open():
        r = json.loads(line)
        v = r.get(field)
        if isinstance(v, list):
            v = ",".join(map(str, v))
        out[r["instance_id"]] = v
    return out


def load_turns(run_dir: Path) -> dict[str, list[dict]]:
    pp = run_dir / "per_problem"
    out = {}
    for f in sorted(pp.glob("*.csv")):
        with f.open() as fh:
            out[f.stem] = list(csv.DictReader(fh))
    return out


def hist_panel(ax, vals: np.ndarray, label: str, bucket_label: str, color: str) -> int:
    """Render one histogram panel; returns the max bin count (for y-uniformizing)."""
    vals = vals[vals > 0]
    for lo, hi, fill, _txt in REGIONS:
        ax.axvspan(lo, hi, color=fill, alpha=0.45, zorder=0)
    if not len(vals):
        ax.set_title(f"{label} — {bucket_label}\n(no data)"); return 0
    bins = np.logspace(np.log10(XLIM_ALL[0]), np.log10(XLIM_ALL[1]), 36)
    counts, _, _ = ax.hist(vals, bins=bins, color=color, edgecolor="white",
                           linewidth=0.4, zorder=2)
    med, mean = float(np.median(vals)), float(np.mean(vals))
    ax.axvline(med, color="black", ls="--", lw=1.4, label=f"med={med:.0f}", zorder=3)
    ax.axvline(mean, color="black", ls=":", lw=1.2, label=f"mean={mean:.0f}", zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(*XLIM_ALL)
    ax.set_title(f"{label} — {bucket_label}\nn={len(vals)} turns",
                 fontsize=10, fontweight="bold")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.95)
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    return int(counts.max()) if len(counts) else 0


def annotate_regions(ax, vals: np.ndarray, ymax: float) -> None:
    """Region labels and percent-of-turns counts. Called after y-axis is fixed."""
    for lo, hi, _fill, txt in REGIONS:
        n = int(((vals >= lo) & (vals < hi)).sum())
        pct = 100 * n / len(vals) if len(vals) else 0
        cx = float(np.sqrt(lo * hi))
        ax.text(cx, ymax * 0.95, txt, ha="center", va="top", fontsize=6.5, color="#374151")
        ax.text(cx, ymax * 0.78, f"n={n}\n({pct:.0f}%)", ha="center", va="top",
                fontsize=6.5, color="#374151")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--bucket", default="difficulty")
    ap.add_argument("--buckets", nargs="+", default=None)
    ap.add_argument("--title-suffix", default="")
    ap.add_argument("--drop-empty", action="store_true", default=True)
    args = ap.parse_args()

    bucket_of = load_bucket_map(args.run_dir, args.bucket)
    turns_by_iid = load_turns(args.run_dir)
    if args.drop_empty:
        for iid, rows in list(turns_by_iid.items()):
            turns_by_iid[iid] = [r for r in rows if r.get("category") != "empty"]
    buckets = args.buckets or VERIFIED_BUCKETS

    # vals[(metric, bucket)] = list[float]
    vals: dict[tuple[str, str], list[float]] = defaultdict(list)
    for iid, rows in turns_by_iid.items():
        b = bucket_of.get(iid)
        if b not in buckets:
            continue
        for r in rows:
            for m, *_ in METRICS:
                v = num(r, m)
                if not np.isnan(v):
                    vals[(m, b)].append(v)

    rows, cols = len(METRICS), len(buckets)
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 4.2 * rows))
    if rows == 1:
        axes = axes.reshape(1, cols)
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(
        f"OSL / ISL / ISL_new distributions (log-binned) by {args.bucket}{suffix}",
        fontsize=14,
    )
    # First pass: render histograms and capture row-wise max count.
    row_max: list[int] = [0] * rows
    for ri, (m, label) in enumerate(METRICS):
        for ci, b in enumerate(buckets):
            color = COL_COLOR[ci % len(COL_COLOR)]
            mx = hist_panel(axes[ri, ci], np.array(vals[(m, b)]), label, b, color)
            row_max[ri] = max(row_max[ri], mx)
            if ci == 0:
                axes[ri, ci].set_ylabel(f"{label}\nCount")
            if ri == rows - 1:
                axes[ri, ci].set_xlabel("tokens (log scale)")
    # Second pass: enforce identical y-axis per row + add region annotations
    # using the now-finalized y range.
    for ri, (m, label) in enumerate(METRICS):
        ymax = row_max[ri] * 1.05
        for ci, b in enumerate(buckets):
            axes[ri, ci].set_ylim(0, ymax)
            annotate_regions(axes[ri, ci], np.array(vals[(m, b)]), ymax)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
