#!/usr/bin/env python3
"""
3-row by 4-column histogram grid:
  Row 1: OSL distribution per difficulty bucket
  Row 2: ISL distribution per difficulty bucket
  Row 3: ISL_new (uncached) distribution per difficulty bucket

Each cell is a log-scaled histogram with median (dashed) and mean (dotted)
overlays. Only the OSL row carries the semantic region bands (tool-call JSON
/ text+tool / Edit-Write / large Write / huge context); the ISL and ISL_new
rows are bare because those token ranges describe context size, not output
archetypes, and the OSL labels would mislabel them.

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

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

# White stroke around region label text so it stays legible when bars sit
# underneath it. Cleaner than a visible bbox.
LABEL_STROKE = [pe.withStroke(linewidth=3, foreground="white")]

VERIFIED_BUCKETS = ["<15 min fix", "15 min - 1 hour", "1+ hours"]
COL_COLOR = ["#3b82f6", "#ec4899", "#22c55e"]
# Collapse the sparse ">4 hours" bucket (≈3 problems) into "1+ hours" alongside
# "1-4 hours" so the hard bucket has enough samples to be meaningful.
DIFFICULTY_REMAP = {">4 hours": "1+ hours", "1-4 hours": "1+ hours"}
# Semantic regions for the OSL row, derived from the Qwen × Verified × 500
# data by joining each OSL bucket to its dominant tool name (see README).
# Boundaries chosen where the modal tool changes:
#   <100   : Read/Bash/Grep/Glob make up ~97% of turns      → exploration
#   100-1k : ExitPlanMode/TodoWrite/Read/short Edit         → plan / small action
#   1k-5k  : Edit + Write make up ~90% of turns             → code edit
# (5k+ band dropped: ~0 turns land there on either Qwen or Opus runs.)
REGIONS_OSL = [
    (1,      100,     "#dbeafe", "tool calls"),
    (100,    1_000,   "#fef9c3", "plan /\nsmall actions"),
    (1_000,  300_000, "#fce7f3", "code edits"),
]
# ISL_new = input + cache_creation = tokens NOT served from cache. On turn 1
# this is the full claude baseline; on later turns it's whatever new content
# the latest tool_result added. Categories derived from the Qwen × Verified
# × 500 distribution (counts in plot_dist_grid.py header).
REGIONS_ISL_NEW = [
    (1,        200,     "#dbeafe", "small\ntool result"),
    (200,      2_000,   "#fef9c3", "file read"),
    (2_000,    20_000,  "#fce7f3", "large read"),
    (20_000,   300_000, "#fee2e2", "system prompt /\ncompaction"),
]
# Backward-compat alias used by the OSL annotation code.
REGIONS = REGIONS_OSL

# Median first-substantive-turn ISL across 500 SWE-bench Verified problems
# under Qwen3-Coder-30B = 26,975 tokens (p10=26,750, p90=27,597). That is
# the claude code baseline: system prompt + 18 tool schemas + CLAUDE.md +
# task statement. Drawn as a vertical reference line on the ISL panel.
CLAUDE_BASELINE_ISL = 27_000
# Uniform x-range across the entire figure so all panels are visually comparable.
XLIM_ALL = (1, 300_000)
METRICS = [
    ("osl",     "OSL",          {"loc": "lower right"}),
    ("isl",     "ISL",          {"loc": "upper left"}),
    # Lifted ~12% above the axis bottom so it sits just above the lowest bar.
    ("isl_new", "ISL uncached", {"loc": "lower right",
                                  "bbox_to_anchor": (1.0, 0.12)}),
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
        if field == "difficulty":
            v = DIFFICULTY_REMAP.get(v, v)
        out[r["instance_id"]] = v
    return out


def load_turns(run_dir: Path) -> dict[str, list[dict]]:
    pp = run_dir / "per_problem"
    out = {}
    for f in sorted(pp.glob("*.csv")):
        with f.open() as fh:
            out[f.stem] = list(csv.DictReader(fh))
    return out


def hist_panel(ax, vals: np.ndarray, label: str, bucket_label: str, color: str,
               regions: list | None = None,
               baseline: tuple[float, str] | None = None,
               legend_kwargs: dict | None = None) -> int:
    """Render one histogram panel; returns the max bin count (for y-uniformizing).

    regions  : optional list of (lo, hi, fill, txt) — drawn as colored bands.
    baseline : optional (x, label) — drawn as a vertical reference line.
    """
    vals = vals[vals > 0]
    if regions:
        for lo, hi, fill, _txt in regions:
            ax.axvspan(lo, hi, color=fill, alpha=0.45, zorder=0)
    if not len(vals):
        ax.set_title(f"{label} — {bucket_label}\n(no data)"); return 0
    bins = np.logspace(np.log10(XLIM_ALL[0]), np.log10(XLIM_ALL[1]), 36)
    counts, _, _ = ax.hist(vals, bins=bins, color=color, edgecolor="white",
                           linewidth=0.4, zorder=2)
    med, mean = float(np.median(vals)), float(np.mean(vals))
    ax.axvline(med, color="black", ls="--", lw=1.4, label=f"med={med:.0f}", zorder=3)
    ax.axvline(mean, color="black", ls=":", lw=1.2, label=f"mean={mean:.0f}", zorder=3)
    if baseline is not None:
        x_base, base_lbl = baseline
        ax.axvline(x_base, color="#dc2626", ls="-", lw=1.6,
                   label=base_lbl, zorder=3, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlim(*XLIM_ALL)
    ax.set_title(f"{label} — {bucket_label}\nn={len(vals)} turns",
                 fontsize=10, fontweight="bold")
    ax.legend(**(legend_kwargs or {"loc": "upper right"}),
              fontsize=9, framealpha=0.95)
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    return int(counts.max()) if len(counts) else 0


def annotate_regions(ax, vals: np.ndarray, ymax: float,
                     regions: list = REGIONS) -> None:
    """Region label + count inside the panel as bold plain text with a thin
    white stroke so bars underneath stay visible but the text reads clearly."""
    for lo, hi, _fill, txt in regions:
        n = int(((vals >= lo) & (vals < hi)).sum())
        pct = 100 * n / len(vals) if len(vals) else 0
        cx = float(np.sqrt(lo * hi))
        ax.text(cx, ymax * 0.95, txt,
                ha="center", va="top",
                fontsize=8, color="#1f2937", fontweight="bold",
                linespacing=1.1, zorder=5,
                path_effects=LABEL_STROKE)
        ax.text(cx, ymax * 0.78, f"n={n}\n({pct:.0f}%)",
                ha="center", va="top",
                fontsize=7, color="#4b5563",
                zorder=5,
                path_effects=LABEL_STROKE)


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
    # Per-metric semantic overlays: OSL gets archetype bands, ISL_new gets
    # input-content-type bands, ISL gets a vertical baseline at the claude
    # code first-turn ISL.
    overlay_for = {
        "osl":     (REGIONS_OSL,     None),
        "isl":     (None,            (CLAUDE_BASELINE_ISL, "Claude Code baseline (~27k)")),
        "isl_new": (REGIONS_ISL_NEW, None),
    }
    row_max: list[int] = [0] * rows
    for ri, (m, label, legend_kwargs) in enumerate(METRICS):
        regions, baseline = overlay_for.get(m, (None, None))
        for ci, b in enumerate(buckets):
            color = COL_COLOR[ci % len(COL_COLOR)]
            mx = hist_panel(axes[ri, ci], np.array(vals[(m, b)]), label, b, color,
                            regions=regions, baseline=baseline,
                            legend_kwargs=legend_kwargs)
            row_max[ri] = max(row_max[ri], mx)
            if ci == 0:
                axes[ri, ci].set_ylabel(f"{label}\nCount")
            if ri == rows - 1:
                axes[ri, ci].set_xlabel("tokens (log scale)")
    # Second pass: enforce identical y-axis per row + add region annotations
    # using the now-finalized y range.
    for ri, (m, label, _kwargs) in enumerate(METRICS):
        regions, _ = overlay_for.get(m, (None, None))
        ymax = row_max[ri] * 1.05
        for ci, b in enumerate(buckets):
            axes[ri, ci].set_ylim(0, ymax)
            if regions is not None:
                annotate_regions(axes[ri, ci], np.array(vals[(m, b)]),
                                 ymax, regions=regions)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
