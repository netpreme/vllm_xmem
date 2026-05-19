#!/usr/bin/env python3
"""
Aggregate ISL / ISL_new / OSL distributions across ALL problems (no per-
difficulty split). Same histograms as plot_dist_grid.py but collapsed into
a single 1×3 row.

Usage:
  python3 plot_dist_agg.py \
      --run-dir runs/20260513_175826 \
      --out analysis/analysis_dist_agg.png \
      --title-suffix "claude × Verified × 500"
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Reuse the same region definitions + label-stroke effect as the per-
# difficulty grid so the two figures stay consistent.
from plot_dist_grid import (
    REGIONS_OSL, REGIONS_ISL_NEW, CLAUDE_BASELINE_ISL,
    XLIM_ALL, METRICS, num, load_turns, LABEL_STROKE,
)


def annotate_regions(ax, vals: np.ndarray, ymax: float, regions) -> None:
    for lo, hi, _fill, txt in regions:
        n = int(((vals >= lo) & (vals < hi)).sum())
        pct = 100 * n / len(vals) if len(vals) else 0
        cx = float(np.sqrt(lo * hi))
        ax.text(cx, ymax * 0.95, txt,
                ha="center", va="top",
                fontsize=10, color="#1f2937", fontweight="bold",
                linespacing=1.1, zorder=5,
                path_effects=LABEL_STROKE)
        ax.text(cx, ymax * 0.78, f"n={n}\n({pct:.0f}%)",
                ha="center", va="top",
                fontsize=9, color="#4b5563",
                zorder=5,
                path_effects=LABEL_STROKE)


def hist_panel(ax, vals: np.ndarray, label: str, color: str,
               regions, baseline, legend_kwargs: dict) -> int:
    vals = vals[vals > 0]
    if regions:
        for lo, hi, fill, _txt in regions:
            ax.axvspan(lo, hi, color=fill, alpha=0.45, zorder=0)
    if not len(vals):
        ax.set_title(f"{label}\n(no data)"); return 0
    bins = np.logspace(np.log10(XLIM_ALL[0]), np.log10(XLIM_ALL[1]), 36)
    counts, _, _ = ax.hist(vals, bins=bins, color=color, edgecolor="white",
                           linewidth=0.4, zorder=2)
    med, mean = float(np.median(vals)), float(np.mean(vals))
    ax.axvline(med, color="black", ls="--", lw=1.4,
               label=f"med={med:.0f}", zorder=3)
    ax.axvline(mean, color="black", ls=":", lw=1.2,
               label=f"mean={mean:.0f}", zorder=3)
    if baseline is not None:
        x_base, base_lbl = baseline
        ax.axvline(x_base, color="#dc2626", ls="-", lw=1.6,
                   label=base_lbl, zorder=3, alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlim(*XLIM_ALL)
    ax.set_title(f"{label} — n={len(vals):,} turns",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("tokens (log scale)")
    ax.legend(**legend_kwargs, fontsize=10, framealpha=0.95)
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    return int(counts.max()) if len(counts) else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title-suffix", default="")
    args = ap.parse_args()

    turns_by_iid = load_turns(args.run_dir)
    all_rows = []
    for rows in turns_by_iid.values():
        all_rows.extend(r for r in rows if r.get("category") != "empty")

    vals_per_metric = {}
    for m, *_ in METRICS:
        vals_per_metric[m] = np.array(
            [num(r, m) for r in all_rows if not np.isnan(num(r, m))]
        )

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(
        f"OSL / ISL / ISL_new distributions — aggregate across all problems{suffix}",
        fontsize=13,
    )

    metric_color = {"osl": "#3b82f6", "isl": "#ec4899", "isl_new": "#22c55e"}
    overlay_for = {
        "osl":     (REGIONS_OSL,     None),
        "isl":     (None,            (CLAUDE_BASELINE_ISL, "Claude Code baseline (~27k)")),
        "isl_new": (REGIONS_ISL_NEW, None),
    }
    panel_max = []
    # In the aggregate figure the panels are taller than in the per-difficulty
    # grid, so the ISL-uncached legend's bbox offset from the grid (~12%) ends
    # up sitting too close to the bottom. Bump it to ~30% so it floats well
    # clear of the lowest bar.
    legend_overrides = {
        "isl_new": {"loc": "lower right", "bbox_to_anchor": (1.0, 0.30)},
    }
    for ci, (m, label, legend_kwargs) in enumerate(METRICS):
        regions, baseline = overlay_for.get(m, (None, None))
        mx = hist_panel(axes[ci], vals_per_metric[m], label, metric_color[m],
                        regions=regions, baseline=baseline,
                        legend_kwargs=legend_overrides.get(m, legend_kwargs))
        panel_max.append(mx)
        if ci == 0:
            axes[ci].set_ylabel("Count")

    for ci, (m, *_rest) in enumerate(METRICS):
        regions, _ = overlay_for.get(m, (None, None))
        ymax = panel_max[ci] * 1.05
        axes[ci].set_ylim(0, ymax)
        if regions is not None:
            annotate_regions(axes[ci], vals_per_metric[m], ymax, regions=regions)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")

    # Sidecar .npz with the binned histogram data + raw per-turn values, so
    # the same distribution can be re-loaded and re-binned without the PNG.
    # Load with:
    #   d = np.load("analysis_dist_agg.npz")
    #   d["bin_edges"], d["osl_counts"], d["osl_values"], ...
    bin_edges = np.logspace(np.log10(XLIM_ALL[0]), np.log10(XLIM_ALL[1]), 36)
    save_dict: dict[str, np.ndarray] = {"bin_edges": bin_edges}
    for m, _label, *_ in METRICS:
        v = vals_per_metric[m]
        v_pos = v[v > 0]
        counts, _ = np.histogram(v_pos, bins=bin_edges) if len(v_pos) else (
            np.zeros(len(bin_edges) - 1, dtype=int), bin_edges
        )
        save_dict[f"{m}_values"] = v_pos
        save_dict[f"{m}_counts"] = counts
    npz_path = args.out.with_suffix(".npz")
    np.savez_compressed(npz_path, **save_dict)
    print(f"wrote {npz_path}")


if __name__ == "__main__":
    main()
