#!/usr/bin/env python3
"""
Combined 2x3 figure:
  Row 1 (histograms)        — overall distribution of OSL, ISL, ISL_new
  Row 2 (per-turn bands)    — distribution as a function of turn number for the
                              same three metrics, drawn as p10/p50/p90 bands
                              with a mean overlay.

Input layout: per-problem CSVs at <run_dir>/per_problem/<iid>.csv.

Usage:
  python3 plot_combined_dist.py \
      --run-dir runs/20260513_175826 \
      --out ~/analysis_combined.png \
      --title-suffix "claude × Verified"
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRICS = [
    ("osl",     "OSL",                "#22c55e", (10,    50_000)),
    ("isl",     "ISL",                "#3b82f6", (100,   300_000)),
    ("isl_new", "ISL_new (uncached)", "#a855f7", (1,     200_000)),
]


def num(r: dict, k: str) -> float:
    v = r.get(k)
    if v in (None, "", "None"):
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def load_turns(run_dir: Path) -> dict[str, list[dict]]:
    pp = run_dir / "per_problem"
    if not pp.exists():
        raise SystemExit(f"missing {pp}")
    out = {}
    for f in sorted(pp.glob("*.csv")):
        with f.open() as fh:
            rows = list(csv.DictReader(fh))
        rows.sort(key=lambda r: float(r.get("ts") or 0))
        out[f.stem] = rows
    return out


def hist_panel(ax, vals: np.ndarray, label: str, color: str, xlim: tuple[float, float]) -> None:
    vals = vals[vals > 0]
    if not len(vals):
        ax.set_title(f"{label}\n(no data)"); return
    bins = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 36)
    ax.hist(vals, bins=bins, color=color, edgecolor="white", linewidth=0.4)
    med, mean = float(np.median(vals)), float(np.mean(vals))
    ax.axvline(med, color="black", ls="--", lw=1.4, label=f"med={med:.0f}")
    ax.axvline(mean, color="black", ls=":", lw=1.2, label=f"mean={mean:.0f}")
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.set_xlabel(f"{label} tokens (log scale)")
    ax.set_ylabel("Count")
    ax.set_title(f"{label} distribution\nn={len(vals)} turns", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.grid(True, axis="y", ls="--", alpha=0.3)


def turn_band_panel(ax, trajectories: dict[str, list[dict]], metric: str, label: str, color: str) -> None:
    items = [ts for ts in trajectories.values() if len(ts) >= 1]
    if not items:
        ax.set_title(f"{label} per turn\n(no data)"); return
    maxlen = max(len(ts) for ts in items)
    # at each turn k, collect values across problems
    p10, p50, p90, mean = [], [], [], []
    ns = []
    for k in range(maxlen):
        vals = [num(ts[k], metric) for ts in items if k < len(ts)]
        vals = [v for v in vals if not np.isnan(v) and v > 0]
        if not vals:
            p10.append(np.nan); p50.append(np.nan); p90.append(np.nan)
            mean.append(np.nan); ns.append(0); continue
        p10.append(float(np.percentile(vals, 10)))
        p50.append(float(np.percentile(vals, 50)))
        p90.append(float(np.percentile(vals, 90)))
        mean.append(float(np.mean(vals)))
        ns.append(len(vals))
    x = np.arange(maxlen)
    p10a, p50a, p90a, meana = map(np.array, (p10, p50, p90, mean))
    valid = ~np.isnan(p50a)
    # Only plot turns where we still have at least 5 problems contributing data
    enough = np.array(ns) >= 5
    use = valid & enough
    ax.fill_between(x[use], p10a[use], p90a[use], color=color, alpha=0.22, label="p10–p90")
    ax.plot(x[use], p50a[use], color=color, lw=1.8, label="p50")
    ax.plot(x[use], meana[use], color="black", lw=1.5, ls="--", label="mean")
    ax.set_xlabel("Turn number")
    ax.set_ylabel(f"{label} (tokens)")
    ax.set_title(f"{label} per turn", fontsize=11, fontweight="bold")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)
    ax.grid(True, ls="--", alpha=0.3)
    # Cap the x-axis where we still have enough samples so the right edge isn't
    # dominated by 1-2 outlier problems with very long turn counts.
    last = int(np.argmax(np.cumsum(use[::-1]) > 0))  # last True index from end
    if use.any():
        ax.set_xlim(0, x[use].max() + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title-suffix", default="")
    ap.add_argument("--drop-empty", action="store_true", default=True,
                    help="skip rows with category=='empty' (title-gen intercepts)")
    args = ap.parse_args()

    turns_by_iid = load_turns(args.run_dir)
    if args.drop_empty:
        for iid, rows in list(turns_by_iid.items()):
            turns_by_iid[iid] = [r for r in rows if r.get("category") != "empty"]

    # Flatten all turns for the histograms.
    all_vals: dict[str, list[float]] = defaultdict(list)
    for rows in turns_by_iid.values():
        for r in rows:
            for m, *_ in METRICS:
                v = num(r, m)
                if not np.isnan(v):
                    all_vals[m].append(v)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(
        f"OSL / ISL / ISL_new distributions (top) and per-turn percentile bands (bottom){suffix}",
        fontsize=13,
    )
    for col, (m, label, color, xlim) in enumerate(METRICS):
        hist_panel(axes[0, col], np.array(all_vals[m]), label, color, xlim)
        turn_band_panel(axes[1, col], turns_by_iid, m, label, color)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
