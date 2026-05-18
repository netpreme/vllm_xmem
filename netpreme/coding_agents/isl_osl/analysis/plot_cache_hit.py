#!/usr/bin/env python3
"""
vLLM prefix cache hit-rate analysis, two-panel view:

Left  : cache_hit_rate per turn index within a problem, stratified by
        difficulty. Solid line = mean; shaded band = min-max range across
        problems at each turn index.
Right : per-difficulty distribution of cache_hit_rate as individual turns,
        with mean and median annotated.

Filters:
  - category != "empty"
  - turn 1 (cold cache) excluded
  - compaction events excluded: cache_hit_rate < 0.5 AND isl_new > 50_000
    (these are the ~143k full-recompute events from context auto-compaction)

Usage:
  python3 plot_cache_hit.py \
      --run-dir runs/20260513_175826 \
      --out analysis/analysis_cache.png \
      --title-suffix "claude × Verified"
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

VERIFIED_BUCKETS = ["<15 min fix", "15 min - 1 hour", "1+ hours"]
BUCKET_COLOR = {
    "<15 min fix":     "#3b82f6",
    "15 min - 1 hour": "#ec4899",
    "1+ hours":        "#22c55e",
}
# Collapse the sparse ">4 hours" bucket (≈3 problems) into "1+ hours" alongside
# "1-4 hours" so the hard bucket has enough samples to be meaningful.
DIFFICULTY_REMAP = {">4 hours": "1+ hours", "1-4 hours": "1+ hours"}


def num(r: dict, k: str) -> float:
    v = r.get(k)
    if v in (None, "", "None"):
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def load_difficulty(run_dir: Path) -> dict[str, str]:
    out = {}
    for line in (run_dir / "problems.jsonl").open():
        r = json.loads(line)
        d = r.get("difficulty")
        out[r["instance_id"]] = DIFFICULTY_REMAP.get(d, d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title-suffix", default="")
    ap.add_argument("--min-samples", type=int, default=2,
                    help="Truncate turn-index axis where fewer problems remain.")
    ap.add_argument("--max-turns", type=int, default=100,
                    help="Hard cap on the left panel's x-axis (turn index).")
    args = ap.parse_args()

    difficulty = load_difficulty(args.run_dir)

    # Walk per-problem CSVs in order; track turn index (1-based) per problem
    # for kept rows. Apply cold-start and compaction filters.
    by_bucket_turn: dict[tuple[str, int], list[float]] = defaultdict(list)
    by_bucket_all: dict[str, list[float]] = defaultdict(list)
    for f in sorted((args.run_dir / "per_problem").glob("*.csv")):
        iid = f.stem
        diff = difficulty.get(iid)
        if diff not in VERIFIED_BUCKETS:
            continue
        with f.open() as fh:
            ti = 0
            for r in csv.DictReader(fh):
                if r.get("category") == "empty":
                    continue
                isl = num(r, "isl")
                isl_new = num(r, "isl_new")
                hit = num(r, "cache_hit_rate")
                if not all(np.isfinite(x) for x in (isl, isl_new, hit)):
                    continue
                if isl <= 0:
                    continue
                ti += 1
                if ti == 1:
                    continue                                  # cold-start
                if hit < 0.5 and isl_new > 50_000:
                    continue                                  # compaction event
                by_bucket_turn[(diff, ti)].append(hit * 100)
                by_bucket_all[diff].append(hit * 100)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(f"vLLM prefix cache hit rate{suffix}", fontsize=12)

    # --- Left: per-turn cache hit — individual dots + mean line per difficulty.
    # Plotted hardest-first so the legend lists hardest at top.
    ax = axes[0]
    for bucket in reversed(VERIFIED_BUCKETS):
        if bucket not in by_bucket_all:
            continue
        color = BUCKET_COLOR[bucket]
        # Individual turn dots (one per problem at each turn index).
        all_x, all_y = [], []
        xs, means = [], []
        max_ti = max((ti for (b, ti) in by_bucket_turn if b == bucket), default=0)
        max_ti = min(max_ti, args.max_turns)
        for ti in range(2, max_ti + 1):
            vals = by_bucket_turn.get((bucket, ti), [])
            if not vals:
                continue
            all_x.extend([ti] * len(vals))
            all_y.extend(vals)
            if len(vals) >= args.min_samples:
                xs.append(ti); means.append(float(np.mean(vals)))
        if all_x:
            ax.scatter(all_x, all_y, s=4, alpha=0.18, color=color,
                       edgecolors="none")
        if xs:
            ax.plot(xs, means, color=color, lw=1.8, label=f"{bucket} (mean)")
    ax.set_xlabel("Turn")
    ax.set_ylabel("Cache hit %")
    ax.set_title("Cache hit % per turn\n"
                 "(turn 1 excluded · compaction turns excluded)",
                 fontsize=11, fontweight="bold")
    ax.set_xlim(2, args.max_turns)
    ax.set_ylim(60, 101)
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

    # --- Right: per-difficulty distribution with mean / median annotations.
    ax = axes[1]
    rng = np.random.default_rng(0)
    xticks, xticklabels = [], []
    for i, bucket in enumerate(VERIFIED_BUCKETS):
        vals = by_bucket_all.get(bucket, [])
        if not vals:
            continue
        arr = np.array(vals)
        # Horizontal jitter for visibility.
        jitter = rng.uniform(-0.18, 0.18, size=len(arr))
        ax.scatter(np.full_like(arr, i) + jitter, arr,
                   s=6, alpha=0.25, color=BUCKET_COLOR[bucket],
                   edgecolors="none",
                   label="individual turns" if i == 0 else None)
        mean_v = float(arr.mean())
        med_v = float(np.median(arr))
        ax.hlines(mean_v, i - 0.3, i + 0.3, color="black", lw=1.6,
                  label="mean" if i == 0 else None)
        ax.hlines(med_v, i - 0.3, i + 0.3, color="black", lw=1.6,
                  linestyles="--", label="median" if i == 0 else None)
        ax.text(i + 0.32, mean_v, f"{mean_v:.1f}%", va="center",
                fontsize=7, color="#374151")
        ax.text(i + 0.32, med_v, f"{med_v:.1f}%", va="center",
                fontsize=7, color="#374151")
        ax.text(i, 60.5, f"n={len(arr):,}", ha="center", va="bottom",
                fontsize=8, color="#6b7280")
        xticks.append(i); xticklabels.append(bucket)
    ax.set_xticks(xticks); ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Cache hit %")
    ax.set_title("Cache hit % — all turns\n(compaction turns excluded)",
                 fontsize=11, fontweight="bold")
    ax.set_ylim(60, 101)
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
