"""KV-cache tier breakdown: where each input token came from.

Every input token is served from exactly one of three disjoint sources,
so the three shares sum to 100%:

    HBM hit       prefix_cache_hits / isl          (local GPU prefix cache)
    offload hit   external_prefix_cache_hits / isl (CPU/offload KV connector)
    recompute     residual                         (prefix-cache miss)

Left  : stacked composition per turn index (mean over problems), with the
        HBM block-pool occupancy (kv_cache_usage_pct) overlaid as a line.
Right : token-weighted overall composition across all turns (one stacked bar).

Filters: drop empty turns (isl <= 0).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from metrics import (
    hbm_hit_rate,
    load_data,
    offload_hit_rate,
    recompute_rate,
    recompute_tokens,
)

# green = fast/local, amber = offload, red = costly recompute.
TIER_COLOR = {"HBM hit": "#22c55e", "offload hit": "#f59e0b", "recompute": "#ef4444"}
KV_COLOR = "#1e3a8a"


def _per_turn(t: np.ndarray, max_turns: int):
    """Mean tier shares (%) and mean kv-cache usage (%) per turn index."""
    isl = t["isl"].astype(np.float64)
    t = t[isl > 0]
    hbm, cpu, rec = (
        hbm_hit_rate(t) * 100,
        offload_hit_rate(t) * 100,
        recompute_rate(t) * 100,
    )
    kv = t["kv_cache_usage_pct"].astype(np.float64)
    turns = t["turn"].astype(int)
    xs, hbm_m, cpu_m, rec_m, kv_m = [], [], [], [], []
    hi = min(int(turns.max()), max_turns) if len(turns) else 0
    for ti in range(1, hi + 1):
        sel = turns == ti
        if not sel.any():
            continue
        xs.append(ti)
        hbm_m.append(float(hbm[sel].mean()))
        cpu_m.append(float(cpu[sel].mean()))
        rec_m.append(float(rec[sel].mean()))
        kv_m.append(float(kv[sel].mean()))
    return tuple(np.array(a) for a in (xs, hbm_m, cpu_m, rec_m, kv_m))


def _plot_per_turn(ax, t: np.ndarray, max_turns: int) -> None:
    xs, hbm, cpu, rec, kv = _per_turn(t, max_turns)
    if not len(xs):
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    ax.stackplot(
        xs,
        hbm,
        cpu,
        rec,
        labels=["HBM hit", "offload hit", "recompute"],
        colors=[
            TIER_COLOR["HBM hit"],
            TIER_COLOR["offload hit"],
            TIER_COLOR["recompute"],
        ],
        alpha=0.85,
    )
    ax.set(
        xlabel="Turn",
        ylabel="Share of input tokens (%)",
        xlim=(1, xs.max()),
        ylim=(0, 100),
    )
    ax.set_title("Per-turn token source composition", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.3)

    kv_ax = ax.twinx()
    kv_ax.plot(xs, kv, color=KV_COLOR, lw=1.8, ls="--", label="kv_cache_usage_pct")
    kv_ax.set_ylabel("HBM block-pool usage (%)", color=KV_COLOR)
    kv_ax.tick_params(axis="y", labelcolor=KV_COLOR)
    kv_ax.set_ylim(0, max(1.0, float(kv.max()) * 1.2))

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = kv_ax.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, framealpha=0.95)


def _plot_overall(ax, t: np.ndarray) -> None:
    """Token-weighted overall composition: one stacked bar."""
    isl = t["isl"].astype(np.int64)
    t = t[isl > 0]
    total = int(t["isl"].sum())
    if total == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return
    shares = {
        "HBM hit": int(t["prefix_cache_hits"].sum()) / total * 100,
        "offload hit": int(t["external_prefix_cache_hits"].sum()) / total * 100,
        "recompute": int(recompute_tokens(t).sum()) / total * 100,
    }
    bottom = 0.0
    for name, pct in shares.items():
        ax.bar(0, pct, bottom=bottom, width=0.6, color=TIER_COLOR[name], label=name)
        if pct >= 2:
            ax.text(
                0,
                bottom + pct / 2,
                f"{name}\n{pct:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
        bottom += pct
    ax.set(xlim=(-0.6, 0.6), ylim=(0, 100), ylabel="Share of input tokens (%)")
    ax.set_xticks([])
    ax.set_title(
        f"Overall (token-weighted, {total:,} tokens)", fontsize=11, fontweight="bold"
    )
    ax.grid(True, axis="y", ls="--", alpha=0.3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title-suffix", default="")
    ap.add_argument(
        "--max-turns", type=int, default=100, help="x-axis cap on the per-turn panel"
    )
    args = ap.parse_args()

    t = load_data(args.save_dir)

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(16, 6),
        gridspec_kw={"width_ratios": [3, 1]},
        constrained_layout=True,
    )
    fig.suptitle(
        f"KV-cache tier breakdown (HBM / offload / recompute)\n{args.title_suffix}",
        fontsize=12,
    )
    _plot_per_turn(axes[0], t, args.max_turns)
    _plot_overall(axes[1], t)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
