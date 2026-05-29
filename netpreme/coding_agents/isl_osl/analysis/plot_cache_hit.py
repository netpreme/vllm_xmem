"""Prefix-cache hit-rate analysis, two-panel view.

Left  : cache_hit_rate per turn index within a problem, by difficulty.
Right : per-difficulty distribution of cache_hit_rate over all turns.

Filters: drop empty turns, drop turn-1 cold-start, drop auto-compaction
turns (cache_hit_rate < 50% AND isl_new > 50k — the ~143k full-recompute
events that would otherwise dominate the aggregate).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from dataset import VERIFIED_BUCKETS, cache_hit_rate, load_data

BUCKET_COLOR = {
    "<15 min fix": "#3b82f6",
    "15 min - 1 hour": "#ec4899",
    "1+ hours": "#22c55e",
}


def _build_buckets(
    t: np.ndarray,
) -> tuple[dict[tuple[str, int], list[float]], dict[str, list[float]]]:
    """Group cache-hit % by (bucket, turn-after-filter) and by bucket alone.

    `turn` in data.npz is the raw 1-based turn including empty rows. Here
    we re-rank within each problem after dropping empty turns, so "turn 1"
    in the figure is the first SUBSTANTIVE turn (which is always cache-cold)
    and gets excluded.
    """
    by_turn: dict[tuple[str, int], list[float]] = defaultdict(list)
    by_bucket: dict[str, list[float]] = defaultdict(list)
    hit = cache_hit_rate(t)
    for iid in np.unique(t["instance_id"]):
        mask = t["instance_id"] == iid
        problem = t[mask]
        problem_hit = hit[mask]
        bucket = problem["difficulty"][0]
        if bucket not in VERIFIED_BUCKETS:
            continue
        ti = 0
        for r, h in zip(problem, problem_hit):
            if r["osl"] <= 0 or r["isl"] <= 0:
                continue
            ti += 1
            if ti == 1:  # cold-start
                continue
            if h < 0.5 and r["isl_new"] > 50_000:  # compaction
                continue
            hit_pct = float(h) * 100
            by_turn[(bucket, ti)].append(hit_pct)
            by_bucket[bucket].append(hit_pct)
    return by_turn, by_bucket


def _plot_per_turn(ax, by_turn, max_turns: int, min_samples: int) -> None:
    """Left panel: cache-hit dots per turn + mean line per difficulty."""
    for bucket in reversed(VERIFIED_BUCKETS):  # hardest first
        color = BUCKET_COLOR[bucket]
        xs_all, ys_all, xs_mean, ys_mean = [], [], [], []
        max_ti = min(
            max((ti for (b, ti) in by_turn if b == bucket), default=0),
            max_turns,
        )
        for ti in range(2, max_ti + 1):
            vals = by_turn.get((bucket, ti), [])
            if not vals:
                continue
            xs_all += [ti] * len(vals)
            ys_all += vals
            if len(vals) >= min_samples:
                xs_mean.append(ti)
                ys_mean.append(float(np.mean(vals)))
        if xs_all:
            ax.scatter(xs_all, ys_all, s=4, alpha=0.18, color=color, edgecolors="none")
        if xs_mean:
            ax.plot(xs_mean, ys_mean, color=color, lw=1.8, label=f"{bucket} (mean)")
    ax.set(xlabel="Turn", ylabel="Cache hit %", xlim=(2, max_turns), ylim=(60, 101))
    ax.set_title("Cache hit % per turn", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)


def _plot_per_bucket(ax, by_bucket) -> None:
    """Right panel: per-difficulty distribution + mean/median annotations."""
    rng = np.random.default_rng(0)
    xticks, xticklabels = [], []
    for i, bucket in enumerate(VERIFIED_BUCKETS):
        vals = by_bucket.get(bucket, [])
        if not vals:
            continue
        arr = np.array(vals)
        ax.scatter(
            np.full_like(arr, i) + rng.uniform(-0.18, 0.18, len(arr)),
            arr,
            s=6,
            alpha=0.25,
            color=BUCKET_COLOR[bucket],
            edgecolors="none",
            label="individual turns" if i == 0 else None,
        )
        mean_v, med_v = float(arr.mean()), float(np.median(arr))
        ax.hlines(
            mean_v,
            i - 0.3,
            i + 0.3,
            color="black",
            lw=1.6,
            label="mean" if i == 0 else None,
        )
        ax.hlines(
            med_v,
            i - 0.3,
            i + 0.3,
            color="black",
            lw=1.6,
            linestyles="--",
            label="median" if i == 0 else None,
        )
        ax.text(
            i + 0.32, mean_v, f"{mean_v:.1f}%", va="center", fontsize=7, color="#374151"
        )
        ax.text(
            i + 0.32, med_v, f"{med_v:.1f}%", va="center", fontsize=7, color="#374151"
        )
        ax.text(
            i,
            60.5,
            f"n={len(arr):,}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#6b7280",
        )
        xticks.append(i)
        xticklabels.append(bucket)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Cache hit %")
    ax.set_ylim(60, 101)
    ax.set_title("Cache hit % per level", fontsize=11, fontweight="bold")
    ax.grid(True, axis="y", ls="--", alpha=0.3)
    # Lift the legend off the bottom so it doesn't overlap the n=… labels.
    ax.legend(
        loc="lower right", bbox_to_anchor=(1.0, 0.10), fontsize=8, framealpha=0.95
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title-suffix", default="")
    ap.add_argument(
        "--min-samples",
        type=int,
        default=2,
        help="omit per-turn means with fewer samples than this",
    )
    ap.add_argument(
        "--max-turns",
        type=int,
        default=100,
        help="hard x-axis cap on the per-turn panel",
    )
    args = ap.parse_args()

    t = load_data(args.run_dir)
    by_turn, by_bucket = _build_buckets(t)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    fig.suptitle(f"Prefix cache hit rate\n{args.title_suffix}", fontsize=12)
    _plot_per_turn(axes[0], by_turn, args.max_turns, args.min_samples)
    _plot_per_bucket(axes[1], by_bucket)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
