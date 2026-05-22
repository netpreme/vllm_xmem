"""Per-turn KV-cache + time decomposition for one representative problem.

Two stacked-bar subplots sharing the x-axis (turn index within the problem):

  TOP — KV cache (GB)
    prefill (cached) = isl_cached × per_token_kv_bytes  ← inherited from prev turns
    recompute        = isl_new    × per_token_kv_bytes  ← fresh prefill this turn
    decode           = osl        × per_token_kv_bytes  ← generated this turn

  BOTTOM — time (ms)
    cached lookup    = γ × isl_cached     (γ ≈ 5.97 ms / 1k cached tokens, from TTFT fit)
    recompute        = ttft_ms − cached   (what's left of TTFT)
    decode           = decode_ms          (observed)

A ribbon below identifies each turn as main agent vs Task-tool sub-agent.

If --instance-id isn't supplied we pick a representative problem in the
medium-difficulty bucket: turn-count near the median, no compaction events.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from data import load_data

# Same constant as build_data.py.
KV_BYTES_PER_TOKEN = 48 * 4 * 128 * 2 * 2

# From the tier-3 TTFT fit in plot_ttft_prefill.py:
#   TTFT ≈ 56.9 + 5.97·(isl_cached/1k) + 79.93·(isl_new/1k) + 2.12·(isl_new·isl/1M)
# γ is the cache-lookup coefficient: ~5.97 ms per 1k cached tokens.
GAMMA_CACHE_MS_PER_TOK = 5.97 / 1000

# (label, blue=cached, red=recompute, green=decode) — kept in sync between the
# KV and time stacks so the colors mean the same thing in both panels.
COLOR_CACHED    = "#3b82f6"
COLOR_RECOMPUTE = "#ef4444"
COLOR_DECODE    = "#22c55e"

KV_COMPONENTS = [
    ("prefill (cached)", "isl_cached", COLOR_CACHED),
    ("recompute",        "isl_new",    COLOR_RECOMPUTE),
    ("decode",           "osl",        COLOR_DECODE),
]


def pick_representative(t: np.ndarray) -> str:
    """Medium difficulty, num_turns near 32, no compaction, median total isl_new."""
    candidates: list[tuple[str, int, int]] = []
    for iid in np.unique(t["instance_id"]):
        rows = t[t["instance_id"] == iid]
        if rows["difficulty"][0] != "15 min - 1 hour":
            continue
        real = rows[rows["category"] != "empty"]
        n = len(real)
        if n < 25 or n > 40:
            continue
        if (real["cache_hit_rate"] < 0.5).any() and (real["isl_new"] > 50_000).any():
            continue
        candidates.append((iid, n, int(real["isl_new"].sum())))
    candidates.sort(key=lambda x: x[2])
    return candidates[len(candidates) // 2][0]


def pick_samples(t: np.ndarray, n: int = 10) -> list[str]:
    """Pick `n` diverse problems for sample figures by stratifying on
    turn count. Filters out compaction problems for cleaner figures."""
    stats: list[tuple[str, int]] = []
    for iid in np.unique(t["instance_id"]):
        rows = t[(t["instance_id"] == iid) & (t["category"] != "empty")]
        if len(rows) < 15:
            continue
        if ((rows["cache_hit_rate"] < 0.5) & (rows["isl_new"] > 50_000)).any():
            continue
        stats.append((iid, len(rows)))
    stats.sort(key=lambda s: s[1])           # sort by turn count
    if len(stats) <= n:
        return [s[0] for s in stats]
    idx = np.linspace(0, len(stats) - 1, n).astype(int)
    return [stats[i][0] for i in idx]


def render(t: np.ndarray, iid: str, out: Path, title_suffix: str) -> None:
    """Render one two-panel figure for one problem and save to `out`."""
    problem = t[(t["instance_id"] == iid) & (t["category"] != "empty")]
    difficulty = problem["difficulty"][0] if len(problem) else ""

    turns = np.arange(1, len(problem) + 1)
    is_sub = problem["agent"] == "sub"
    n_sub  = int(is_sub.sum())

    # ---- KV (GB) components ------------------------------------------------
    kv_series = {name: problem[col].astype(float)
                       * KV_BYTES_PER_TOKEN / 1024**3
                 for name, col, _ in KV_COMPONENTS}

    # ---- Time (ms) components ---------------------------------------------
    ttft  = problem["ttft_ms"].astype(float)
    dec   = problem["decode_ms"].astype(float)
    cached_ms = problem["isl_cached"].astype(float) * GAMMA_CACHE_MS_PER_TOK
    cached_ms = np.minimum(cached_ms, ttft)               # never exceed TTFT
    recompute_ms = np.maximum(ttft - cached_ms, 0.0)
    time_stack = [
        ("cached lookup", cached_ms,    COLOR_CACHED),
        ("recompute",     recompute_ms, COLOR_RECOMPUTE),
        ("decode",        dec,          COLOR_DECODE),
    ]

    fig, (ax_kv, ax_t) = plt.subplots(2, 1, figsize=(13, 9.5),
                                      sharex=True, constrained_layout=True)
    suffix = f" — {title_suffix}" if title_suffix else ""
    fig.suptitle(
        f"{iid}  ({difficulty}, {len(turns)} turns: "
        f"{len(turns)-n_sub} main / {n_sub} sub){suffix}",
        fontsize=11,
    )

    # ---- top panel: KV cache (GB). Sub-agent bars are hatched (///). ------
    bottom = np.zeros(len(turns))
    for name, col, color in KV_COMPONENTS:
        vals = kv_series[name]
        # Main and sub-agent bars are drawn together so the legend has one
        # entry per component; then we walk the rectangles and add a hatch
        # pattern on the sub-agent ones.
        bars = ax_kv.bar(turns, vals, width=1.0, bottom=bottom,
                         color=color, edgecolor="white", linewidth=0.3,
                         label=name)
        for rect, sub in zip(bars, is_sub):
            if sub:
                rect.set_hatch("///")
                rect.set_edgecolor("white")
        bottom += vals
    for x, total in zip(turns, bottom):
        if x % 5 == 0:
            ax_kv.text(x, total * 1.015, f"{total:.1f}", ha="center",
                       va="bottom", fontsize=7, color="#374151")
    ax_kv.set_ylabel("KV cache (GB)")
    ax_kv.set_ylim(0, bottom.max() * 1.10 if len(bottom) else 1)
    ax_kv.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax_kv.grid(True, axis="y", ls="--", alpha=0.3)
    from matplotlib.patches import Patch
    kv_legend_handles = [
        *[Patch(facecolor=c, label=n) for n, _, c in KV_COMPONENTS],
        Patch(facecolor="white", edgecolor="black", hatch="///",
              label="sub-agent (hatched)"),
    ]
    ax_kv.legend(handles=kv_legend_handles, loc="upper left",
                 fontsize=10, framealpha=0.95, title="KV component / agent")

    # ---- bottom panel: per-turn wall-clock time (ms) ----------------------
    # Sub-agent bars are hatched, same convention as the KV panel.
    bottom_t = np.zeros(len(turns))
    for name, vals, color in time_stack:
        bars = ax_t.bar(turns, vals, width=1.0, bottom=bottom_t,
                        color=color, edgecolor="white", linewidth=0.3,
                        label=name)
        for rect, sub in zip(bars, is_sub):
            if sub:
                rect.set_hatch("///")
                rect.set_edgecolor("white")
        bottom_t += vals
    for x, total in zip(turns, bottom_t):
        if x % 5 == 0:
            ax_t.text(x, total * 1.015, f"{int(total)}", ha="center",
                      va="bottom", fontsize=7, color="#374151")
    ax_t.set_ylabel("per-turn wall time (ms)")
    ax_t.set_ylim(0, bottom_t.max() * 1.10 if len(bottom_t) else 1)
    ax_t.grid(True, axis="y", ls="--", alpha=0.3)
    time_legend_handles = [
        *[Patch(facecolor=c, label=n) for n, _, c in time_stack],
        Patch(facecolor="white", edgecolor="black", hatch="///",
              label="sub-agent (hatched)"),
    ]
    ax_t.legend(handles=time_legend_handles, loc="upper left",
                fontsize=10, framealpha=0.95, title="time component / agent")

    ax_t.set_xlabel("turn index within problem (substantive turns)")
    ax_t.set_xlim(0.5, len(turns) + 0.5)
    ax_t.xaxis.set_major_locator(plt.MultipleLocator(max(1, len(turns) // 20)))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (problem={iid}, turns={len(turns)})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",      required=True, type=Path)
    ap.add_argument("--out",          required=True, type=Path,
                    help="output PNG for the representative problem")
    ap.add_argument("--title-suffix", default="")
    ap.add_argument("--instance-id",  default=None,
                    help="problem id for the --out figure; "
                         "auto-picked from medium-difficulty if unset")
    ap.add_argument("--samples-dir",  type=Path, default=None,
                    help="if set, also write 10 diverse-problem samples "
                         "as kv_<iid>.png to this directory")
    args = ap.parse_args()

    t = load_data(args.run_dir)
    iid = args.instance_id or pick_representative(t)
    render(t, iid, args.out, args.title_suffix)

    if args.samples_dir is not None:
        for sample_iid in pick_samples(t, n=10):
            render(t, sample_iid,
                   args.samples_dir / f"kv_{sample_iid}.png",
                   args.title_suffix)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
