"""Distribution of turns-per-problem, aggregate + by difficulty.

Reads data.npz, counts substantive turns per problem (drops 'empty' rows),
and renders a 1×4 row: [aggregate, <15 min fix, 15 min - 1 hour, 1+ hours].
Each panel is a histogram with median + mean overlays.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import VERIFIED_BUCKETS, load_data

BUCKET_COLOR = {
    "<15 min fix":     "#3b82f6",
    "15 min - 1 hour": "#ec4899",
    "1+ hours":        "#22c55e",
    "all":             "#6b7280",
}


def turns_per_problem(t: np.ndarray) -> dict[str, np.ndarray]:
    """Return {bucket: array of turn counts per problem}, plus an 'all' key."""
    substantive = t[t["category"] != "empty"]
    counts_per_iid: dict[str, int] = {}
    diff_per_iid: dict[str, str] = {}
    for iid in np.unique(substantive["instance_id"]):
        rows = substantive[substantive["instance_id"] == iid]
        counts_per_iid[iid] = len(rows)
        diff_per_iid[iid] = rows["difficulty"][0]

    by_bucket: dict[str, list[int]] = {b: [] for b in VERIFIED_BUCKETS}
    for iid, n in counts_per_iid.items():
        b = diff_per_iid[iid]
        if b in by_bucket:
            by_bucket[b].append(n)
    out = {b: np.array(v) for b, v in by_bucket.items()}
    out["all"] = np.array(list(counts_per_iid.values()))
    return out


def hist_panel(ax, vals: np.ndarray, title: str, color: str, x_max: int) -> int:
    """Render one histogram panel; returns the tallest bar so the caller
    can uniformize y across selected panels."""
    if not len(vals):
        ax.set_title(f"{title}\n(no data)"); return 0
    bins = np.arange(0, x_max + 5, 5)
    counts, _, _ = ax.hist(vals, bins=bins, color=color,
                           edgecolor="white", linewidth=0.4)
    med, mean = float(np.median(vals)), float(np.mean(vals))
    ax.axvline(med,  color="black", ls="--", lw=1.4, label=f"med={med:.0f}")
    ax.axvline(mean, color="black", ls=":",  lw=1.2, label=f"mean={mean:.0f}")
    ax.set_xlim(0, x_max)
    ax.set_title(f"{title}\nn={len(vals)} problems", fontsize=10, fontweight="bold")
    ax.set_xlabel("turns per problem")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    return int(counts.max()) if len(counts) else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",      required=True, type=Path)
    ap.add_argument("--out",          required=True, type=Path)
    ap.add_argument("--title-suffix", default="")
    args = ap.parse_args()

    counts = turns_per_problem(load_data(args.run_dir))
    # Share x-axis range across all panels so they're visually comparable.
    # Clip the very long tail (some hard problems have >500 turns) at p99
    # so the bulk of the distribution is readable.
    x_max = int(np.percentile(counts["all"], 99) * 1.1) if len(counts["all"]) else 50
    x_max = max(x_max, 50)

    panels = [("all", "all problems"), *((b, b) for b in VERIFIED_BUCKETS)]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.5),
                             constrained_layout=True)
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(f"Turns per problem{suffix}", fontsize=13)

    bucket_maxes = []
    for ax, (key, title) in zip(axes, panels):
        bar_max = hist_panel(ax, counts[key], title, BUCKET_COLOR[key], x_max)
        if key != "all":
            bucket_maxes.append(bar_max)
    # Aggregate panel keeps its own y-axis (it has many more samples).
    # Per-difficulty panels share the same y so bars are visually comparable.
    if bucket_maxes:
        ymax = max(bucket_maxes) * 1.05
        for ax, (key, _) in zip(axes, panels):
            if key != "all":
                ax.set_ylim(0, ymax)
    axes[0].set_ylabel("problems")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
