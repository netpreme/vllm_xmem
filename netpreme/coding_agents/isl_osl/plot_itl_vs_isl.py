#!/usr/bin/env python3
"""
Per-turn scatter of decode latency (ITL) against context size (ISL).

Two panels side by side:
  Left:  Y = itl_ms      X = isl              ("starting KV size")
  Right: Y = itl_ms      X = isl + osl/2      ("average KV size during decode")

Each point = one turn. Color = response category. Marker size scales with osl.
A linear fit (on the kept points) is overlaid.

Filters applied:
  - category != "empty"   (title-generation intercepts)
  - itl_ms not null       (streaming turns only)
  - osl >= 5              (mean-of-few-tokens is too noisy)

Usage:
  python3 plot_itl_vs_isl.py \
      --run-dir runs/20260513_175826 \
      --out ~/analysis_itl_vs_isl.png \
      --title-suffix "claude × Verified"
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CATEGORY_COLORS = {
    "text_only": "#3b82f6",
    "tool_only": "#22c55e",
    "mixed":     "#ec4899",
}

# Shared with plot_dist_grid.py — same x-axis bands so the two figures read
# against the same token-range archetypes.
REGIONS = [
    (1,        100,    "#dbeafe", "tool call JSON"),
    (100,      1_000,  "#fef9c3", "text + tool"),
    (1_000,    5_000,  "#fce7f3", "Edit/Write"),
    (5_000,    50_000, "#dcfce7", "large Write"),
    (50_000,   300_000,"#e0e7ff", "huge context"),
]


def num(r: dict, k: str) -> float:
    v = r.get(k)
    if v in (None, "", "None"):
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def load_turns(run_dir: Path) -> list[dict]:
    out = []
    for f in sorted((run_dir / "per_problem").glob("*.csv")):
        with f.open() as fh:
            out.extend(csv.DictReader(fh))
    return out


def scatter_panel(ax, xs, ys, cats, osls, title, xlabel, color_map) -> None:
    # Marker size by osl (log scale).
    s = np.clip(8 + np.log1p(osls) * 6, 8, 80)
    for cat in ["tool_only", "mixed", "text_only"]:
        mask = cats == cat
        if not mask.any():
            continue
        ax.scatter(xs[mask], ys[mask], s=s[mask], alpha=0.32,
                   color=color_map[cat], edgecolors="white", linewidths=0.2,
                   label=f"{cat} (n={int(mask.sum())})")
    # Linear fit in log-x space.
    keep = (xs > 0) & np.isfinite(ys)
    if keep.sum() >= 10:
        lx = np.log10(xs[keep])
        # Standard least-squares line: y = m * log10(x) + b.
        m, b = np.polyfit(lx, ys[keep], 1)
        xx = np.logspace(np.log10(max(xs[keep].min(), 1)),
                         np.log10(xs[keep].max()), 200)
        ax.plot(xx, m * np.log10(xx) + b, color="black", lw=1.6,
                label=f"fit: itl ≈ {m:.2f}·log10(x) + {b:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("itl_ms  (per-token decode latency)")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title-suffix", default="")
    ap.add_argument("--min-osl", type=int, default=5)
    args = ap.parse_args()

    rows = load_turns(args.run_dir)

    isls, osls, itls, cats = [], [], [], []
    for r in rows:
        if r.get("category") == "empty":
            continue
        itl = num(r, "itl_ms")
        osl = num(r, "osl")
        isl = num(r, "isl")
        if not (np.isfinite(itl) and np.isfinite(osl) and np.isfinite(isl)):
            continue
        if osl < args.min_osl or isl <= 0 or itl <= 0:
            continue
        isls.append(isl); osls.append(osl); itls.append(itl)
        cats.append(r.get("category", "?"))
    isls = np.array(isls); osls = np.array(osls); itls = np.array(itls)
    cats = np.array(cats)
    print(f"plotting {len(isls)} turns (after filtering)")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(f"Per-turn ITL vs ISL (decode latency vs KV-cache size){suffix}",
                 fontsize=13)

    scatter_panel(
        axes[0], isls, itls, cats, osls,
        title="ITL vs ISL (starting KV size)",
        xlabel="isl (tokens, log scale)",
        color_map=CATEGORY_COLORS,
    )
    scatter_panel(
        axes[1], isls + osls / 2, itls, cats, osls,
        title="ITL vs ISL + OSL/2 (avg KV size during decode)",
        xlabel="isl + osl/2 (tokens, log scale)",
        color_map=CATEGORY_COLORS,
    )

    # Annotation about marker size.
    axes[0].annotate("marker size ∝ log(osl)",
                     xy=(0.98, 0.02), xycoords="axes fraction",
                     ha="right", va="bottom", fontsize=8, color="#6b7280")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
