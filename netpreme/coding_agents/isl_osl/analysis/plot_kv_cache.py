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

Task-tool sub-agent turns (agent == "sub") are drawn with a `///` hatch
on top of the same color stack so they stand out without changing the
component color encoding.

If --instance-id isn't supplied we pick a representative problem in the
medium-difficulty bucket: turn-count near the median, no compaction events.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter

from metrics import agent, cache_hit_rate, isl_cached, load_data, ttft_ms

# Hatch pattern used to mark Task-tool sub-agent turns on every stacked bar.
SUB_AGENT_HATCH = "///"

# Same constant as build_dataset.py.
KV_BYTES_PER_TOKEN = 48 * 4 * 128 * 2 * 2

# From the tier-3 TTFT fit in plot_ttft_prefill.py:
#   TTFT ≈ 56.9 + 5.97·(isl_cached/1k) + 79.93·(isl_new/1k) + 2.12·(isl_new·isl/1M)
# γ is the cache-lookup coefficient: ~5.97 ms per 1k cached tokens.
GAMMA_CACHE_MS_PER_TOK = 5.97 / 1000

# (label, blue=cached, red=recompute, green=decode) — kept in sync between the
# KV and time stacks so the colors mean the same thing in both panels.
COLOR_CACHED = "#3b82f6"
COLOR_RECOMPUTE = "#ef4444"
COLOR_DECODE = "#22c55e"

KV_COMPONENTS = [
    ("prefill (cached)", "isl_cached", COLOR_CACHED),  # derived, see render()
    ("recompute", "isl_new", COLOR_RECOMPUTE),
    ("decode", "osl", COLOR_DECODE),
]


def _apply_sub_agent_hatch(bars, is_sub: np.ndarray) -> None:
    """Walk a `BarContainer` and stamp the `///` hatch on bars whose
    corresponding turn is a Task-tool sub-agent."""
    for rect, sub in zip(bars, is_sub):
        if sub:
            rect.set_hatch(SUB_AGENT_HATCH)
            rect.set_edgecolor("white")


def _legend_handles(components, n_sub: int) -> list[Patch]:
    """Build a combined legend: one patch per stack component plus, if any
    sub-agent turns exist in this problem, a single hatched patch
    explaining the encoding."""
    handles = [Patch(facecolor=c, label=n) for n, _, c in components]
    if n_sub > 0:
        handles.append(
            Patch(
                facecolor="white",
                edgecolor="black",
                hatch=SUB_AGENT_HATCH,
                label="sub-agent (hatched)",
            )
        )
    return handles


def pick_representative(t: np.ndarray) -> str:
    """Medium difficulty, num_turns near 32, no compaction, median total isl_new."""
    candidates: list[tuple[str, int, int]] = []
    for iid in np.unique(t["instance_id"]):
        rows = t[t["instance_id"] == iid]
        if rows["difficulty"][0] != "15 min - 1 hour":
            continue
        real = rows[rows["osl"] > 0]
        n = len(real)
        if n < 25 or n > 40:
            continue
        hit = cache_hit_rate(real)
        if (hit < 0.5).any() and (real["isl_new"] > 50_000).any():
            continue
        candidates.append((iid, n, int(real["isl_new"].sum())))
    candidates.sort(key=lambda x: x[2])
    return candidates[len(candidates) // 2][0]


def pick_samples(t: np.ndarray, n: int = 10) -> list[str]:
    """Pick `n` diverse problems for sample figures by stratifying on
    turn count. Filters out compaction problems for cleaner figures."""
    stats: list[tuple[str, int]] = []
    for iid in np.unique(t["instance_id"]):
        rows = t[(t["instance_id"] == iid) & (t["osl"] > 0)]
        if len(rows) < 15:
            continue
        hit = cache_hit_rate(rows)
        if ((hit < 0.5) & (rows["isl_new"] > 50_000)).any():
            continue
        stats.append((iid, len(rows)))
    stats.sort(key=lambda s: s[1])  # sort by turn count
    if len(stats) <= n:
        return [s[0] for s in stats]
    idx = np.linspace(0, len(stats) - 1, n).astype(int)
    return [stats[i][0] for i in idx]


def render(t: np.ndarray, iid: str, out: Path, title_suffix: str) -> None:
    """Render one two-panel figure for one problem and save to `out`."""
    problem = t[(t["instance_id"] == iid) & (t["osl"] > 0)]
    difficulty = problem["difficulty"][0] if len(problem) else ""

    turns = np.arange(1, len(problem) + 1)
    is_sub = agent(problem) == "sub"
    n_sub = int(is_sub.sum())

    # ---- KV (GB) components ------------------------------------------------
    # `isl_cached` is derived; everything else is a raw column.
    cached_tokens = isl_cached(problem)
    token_series = {
        "isl_cached": cached_tokens,
        "isl_new": problem["isl_new"],
        "osl": problem["osl"],
    }
    kv_series = {
        name: token_series[col].astype(float) * KV_BYTES_PER_TOKEN / 1024**3
        for name, col, _ in KV_COMPONENTS
    }

    # ---- Time (ms) components ---------------------------------------------
    ttft = ttft_ms(problem)
    dec = problem["decode_ms"].astype(float)
    cached_ms = cached_tokens.astype(float) * GAMMA_CACHE_MS_PER_TOK
    cached_ms = np.minimum(cached_ms, ttft)  # never exceed TTFT
    recompute_ms = np.maximum(ttft - cached_ms, 0.0)
    time_stack = [
        ("cached lookup", cached_ms, COLOR_CACHED),
        ("recompute", recompute_ms, COLOR_RECOMPUTE),
        ("decode", dec, COLOR_DECODE),
    ]

    fig, (ax_kv, ax_t) = plt.subplots(
        2, 1, figsize=(13, 9.5), sharex=True, constrained_layout=True
    )
    suffix = f" — {title_suffix}" if title_suffix else ""
    subtitle = (
        f"{iid}  ({difficulty}, {len(turns)} turns: "
        f"{len(turns) - n_sub} main / {n_sub} sub){suffix}"
    )
    fig.suptitle(subtitle, fontsize=11)

    # ---- top panel: KV cache (GB) -----------------------------------------
    # Stacked bars first, then a single pass that hatches the sub-agent
    # bars in-place. Keeping the hatching as a post-processing step keeps
    # the main bar-drawing loop readable.
    bottom = np.zeros(len(turns))
    for name, col, color in KV_COMPONENTS:
        vals = kv_series[name]
        bars = ax_kv.bar(
            turns,
            vals,
            width=1.0,
            bottom=bottom,
            color=color,
            edgecolor="white",
            linewidth=0.3,
            label=name,
        )
        _apply_sub_agent_hatch(bars, is_sub)
        bottom += vals
    for x, total in zip(turns, bottom):
        if x % 5 == 0:
            ax_kv.text(
                x,
                total * 1.015,
                f"{total:.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#374151",
            )
    ax_kv.set_ylabel("KV cache (GB)")
    ax_kv.set_ylim(0, bottom.max() * 1.10 if len(bottom) else 1)
    ax_kv.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax_kv.grid(True, axis="y", ls="--", alpha=0.3)
    ax_kv.legend(
        handles=_legend_handles(KV_COMPONENTS, n_sub),
        loc="upper left",
        fontsize=10,
        framealpha=0.95,
        title="KV component / agent",
    )

    # ---- bottom panel: per-turn wall-clock time (ms) ----------------------
    bottom_t = np.zeros(len(turns))
    for name, vals, color in time_stack:
        bars = ax_t.bar(
            turns,
            vals,
            width=1.0,
            bottom=bottom_t,
            color=color,
            edgecolor="white",
            linewidth=0.3,
            label=name,
        )
        _apply_sub_agent_hatch(bars, is_sub)
        bottom_t += vals
    for x, total in zip(turns, bottom_t):
        if x % 5 == 0:
            ax_t.text(
                x,
                total * 1.015,
                f"{int(total)}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#374151",
            )
    ax_t.set_ylabel("per-turn wall time (ms)")
    ax_t.set_ylim(0, bottom_t.max() * 1.10 if len(bottom_t) else 1)
    ax_t.grid(True, axis="y", ls="--", alpha=0.3)
    ax_t.legend(
        handles=_legend_handles(time_stack, n_sub),
        loc="upper left",
        fontsize=10,
        framealpha=0.95,
        title="time component / agent",
    )

    ax_t.set_xlabel("turn index within problem (substantive turns)")
    ax_t.set_xlim(0.5, len(turns) + 0.5)
    ax_t.xaxis.set_major_locator(plt.MultipleLocator(max(1, len(turns) // 20)))

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  (problem={iid}, turns={len(turns)})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", required=True, type=Path)
    ap.add_argument(
        "--out",
        required=True,
        type=Path,
        help="output PNG for the representative problem",
    )
    ap.add_argument("--title-suffix", default="")
    ap.add_argument(
        "--instance-id",
        default=None,
        help="problem id for the --out figure; "
        "auto-picked from medium-difficulty if unset",
    )
    ap.add_argument(
        "--samples-dir",
        type=Path,
        default=None,
        help="if set, also write 10 diverse-problem samples "
        "as kv_<iid>.png to this directory",
    )
    args = ap.parse_args()

    t = load_data(args.save_dir)
    iid = args.instance_id or pick_representative(t)
    render(t, iid, args.out, args.title_suffix)

    if args.samples_dir is not None:
        for sample_iid in pick_samples(t, n=10):
            render(
                t,
                sample_iid,
                args.samples_dir / f"kv_{sample_iid}.png",
                args.title_suffix,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
