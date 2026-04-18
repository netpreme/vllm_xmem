#!/usr/bin/env python3
"""
Analyze ISL/OSL token distributions and cache hit rates from SWE-bench task results.

Reads per-task metrics.json files produced by benchmark/tasks/run_swebench.py
and generates figures broken down by difficulty level (<15 min, 15 min–1 hour,
1–4 hours, >4 hours).

What you need to run first:
    python3 benchmark/tasks/run_swebench.py --all
    # Results land in benchmark/tasks/results/<instance_id>_<timestamp>/metrics.json

Figures produced (saved alongside the results directory):
  Per difficulty level:
    analysis_cache_<level>.png   per-turn cache hit % evolution + overall distribution
    analysis_dist_<level>.png    OSL histogram + ISL per-turn scatter

  Combined across all levels:
    analysis_cache.png
    analysis_dist.png

Cache hit % is estimated per turn as:
    hit% = (prev_isl + prev_osl) / curr_isl  (capped at 100%)
Turn 0 is always a cold start (0%). Turns where ISL drops (context compaction) are excluded.

Usage:
    python3 analysis/task_analysis.py                          # reads benchmark/tasks/results/
    python3 analysis/task_analysis.py path/to/results/
    python3 analysis/task_analysis.py dir1/ dir2/              # overlay two runs for comparison
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
SCRIPT_DIR  = Path(__file__).resolve().parent
DEFAULT_DIR = SCRIPT_DIR.parent / "benchmark" / "tasks" / "results"

SELECTED_IDS: set[str] | None = None  # None = use all available results

# difficulty levels in logical order
LEVEL_ORDER = ["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours", "unknown"]

COLORS  = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]
MARKERS = ["o", "s", "^", "D", "v"]

# Stable color keyed by difficulty level — same level → same color in every figure
LEVEL_COLOR  = {lvl: COLORS[i % len(COLORS)]  for i, lvl in enumerate(LEVEL_ORDER)}
LEVEL_MARKER = {lvl: MARKERS[i % len(MARKERS)] for i, lvl in enumerate(LEVEL_ORDER)}


# ── data loading ──────────────────────────────────────────────────────────────

def load_dir(results_dir: Path) -> list[dict]:
    records = []
    for mf in sorted(results_dir.glob("*/metrics.json")):
        try:
            data = json.loads(mf.read_text())
            if SELECTED_IDS is None or data.get("instance_id") in SELECTED_IDS:
                records.append(data)
        except Exception as e:
            print(f"  Warning: {mf}: {e}", file=sys.stderr)
    return records


def extract(record: dict) -> dict | None:
    turns = record.get("turns", [])
    if not turns:
        return None

    isl = [t["isl"] for t in turns if t.get("isl") is not None]
    osl = [t["osl"] for t in turns if t.get("osl") is not None]
    if not isl or not osl:
        return None

    summary   = record.get("summary", {})
    final_isl = record.get("final_usage", {}).get("input_tokens")

    # ── per-turn cache hit estimate ───────────────────────────────────────────
    # hit% = (prev_isl + prev_osl) / curr_isl * 100  (capped at 100)
    # Turn 0: cold start → 0 %.  Compaction turn (ISL drops): None (excluded).
    cache_hits: list[float | None] = [0.0]
    total_cached = 0.0
    for i in range(1, len(turns)):
        p_isl = turns[i - 1].get("isl") or 0
        p_osl = turns[i - 1].get("osl") or 0
        c_isl = turns[i].get("isl") or 0
        if c_isl < p_isl:
            cache_hits.append(None)
        else:
            hit = min((p_isl + p_osl) / c_isl * 100, 100.0) if c_isl else 0.0
            cache_hits.append(hit)
            total_cached += min(p_isl + p_osl, c_isl)

    total_input       = sum(t.get("isl") or 0 for t in turns)
    overall_cache_hit = total_cached / total_input * 100 if total_input else 0.0
    valid_hits        = [h for h in cache_hits[1:] if h is not None]
    avg_cache_hit     = float(np.mean(valid_hits)) if valid_hits else 0.0

    return {
        "instance_id":        record["instance_id"],
        "difficulty":         str(record.get("difficulty") or "unknown"),
        "avg_isl_per_turn":   float(np.mean(isl)),
        "avg_osl_per_turn":   float(np.mean(osl)),
        "total_isl":          float(final_isl if final_isl else sum(isl)),
        "total_osl":          float(summary.get("total_osl") or sum(osl)),
        "avg_tools_per_turn": float(summary.get("avg_tools_per_turn") or 0),
        "total_turns":        len(turns),
        "isl_per_turn":       isl,
        "osl_per_turn":       osl,
        "cache_hits_per_turn": cache_hits,
        "avg_cache_hit_pct":  avg_cache_hit,
        "overall_cache_hit_pct": overall_cache_hit,
    }


def group_by_level(records: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r["difficulty"]].append(r)
    return dict(groups)


def sorted_levels(groups: dict) -> list[str]:
    present = set(groups.keys())
    ordered = [l for l in LEVEL_ORDER if l in present]
    ordered += sorted(present - set(LEVEL_ORDER))   # any extra levels at end
    return ordered


# ── cache figure ──────────────────────────────────────────────────────────────

def plot_cache_figure(datasets: list[tuple[str, dict]], rng: np.random.Generator,
                      out_suffix: str = "") -> None:
    """Cache hit % evolution (line) + overall distribution (scatter)."""
    fig, (ax_line, ax_dist) = plt.subplots(1, 2, figsize=(14, 5))

    # ── left: per-turn mean with min/max band, starting from turn 1 ──────────
    for j, (name, grouped) in enumerate(datasets):
        ds_label = name if len(datasets) > 1 else None
        for li, lvl in enumerate(sorted_levels(grouped)):
            color  = LEVEL_COLOR.get(lvl, COLORS[li % len(COLORS)])
            ls     = ["-", "--", ":", "-."][j % 4]
            series = [r["cache_hits_per_turn"] for r in grouped[lvl]]
            if not series:
                continue
            max_t  = max(len(s) for s in series)

            xs, means, mins, maxs = [], [], [], []
            for t in range(1, max_t):   # start from 1, skip turn 0 (always 0)
                # skip None (compaction turns)
                vals = [s[t] for s in series if t < len(s) and s[t] is not None]
                if len(vals) >= 2:
                    xs.append(t)
                    means.append(float(np.mean(vals)))
                    mins.append(float(np.min(vals)))
                    maxs.append(float(np.max(vals)))

            lbl = f"{lvl}" + (f" [{ds_label}]" if ds_label else "")
            ax_line.plot(xs, means, color=color, linestyle=ls, linewidth=2, label=lbl)
            ax_line.fill_between(xs, mins, maxs, color=color, alpha=0.15)

    ax_line.set_title("Cache hit % per turn\n(turn 0 excluded · compaction turns excluded)",
                      fontsize=11, fontweight="bold")
    ax_line.set_xlabel("Turn", fontsize=9)
    ax_line.set_ylabel("Cache hit %", fontsize=9)
    ax_line.set_ylim(45, 105)
    ax_line.grid(alpha=0.2, linestyle="--")
    ax_line.spines[["top", "right"]].set_visible(False)

    # legend: level colors + explanation of line/band
    existing_handles, existing_labels = ax_line.get_legend_handles_labels()
    proxy_band  = matplotlib.patches.Patch(facecolor="gray", alpha=0.3,
                                           label="░ min – max range")
    ax_line.legend(handles=existing_handles + [proxy_band],
                   fontsize=8, loc="lower right")

    # ── right: overall cache hit % per instance ───────────────────────────────
    all_levels: list[str] = []
    for _, grouped in datasets:
        for lvl in sorted_levels(grouped):
            if lvl not in all_levels:
                all_levels.append(lvl)

    n_ds  = len(datasets)
    width = 0.6 / max(n_ds, 1)

    for j, (name, grouped) in enumerate(datasets):
        offset = (j - (n_ds - 1) / 2) * width

        for xi, lvl in enumerate(all_levels):
            color = LEVEL_COLOR.get(lvl, COLORS[xi % len(COLORS)])
            vals  = []
            for r in grouped.get(lvl, []):
                vals.extend(v for v in r["cache_hits_per_turn"][1:] if v is not None)
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            jx  = xi + offset + rng.uniform(-width * 0.4, width * 0.4, len(arr))
            ax_dist.scatter(jx, arr, color=color, marker="o", alpha=0.25, s=12, zorder=3)
            m   = float(arr.mean())
            med = float(np.median(arr))
            x0, x1 = xi + offset - width * 0.45, xi + offset + width * 0.45
            ax_dist.hlines(m,   x0, x1, colors=color,   linewidths=2.5, zorder=4)
            ax_dist.hlines(med, x0, x1, colors="black", linewidths=1.5,
                           linestyles="--", zorder=4)
            ax_dist.text(x1 + 0.04, m,   f"{m:.1f}%",   color=color,   fontsize=7,
                         va="center", ha="left")
            ax_dist.text(x1 + 0.04, med, f"{med:.1f}%", color="black", fontsize=7,
                         va="center", ha="left")
            ax_dist.text(xi + offset, 46.5, f"n={len(arr)}",
                         ha="center", va="bottom", fontsize=7, color="gray")

    # legend: one mean handle per level (colored) + one median handle (black dashed)
    legend_handles = []
    for lvl in all_levels:
        color = LEVEL_COLOR.get(lvl, COLORS[0])
        legend_handles.append(
            matplotlib.lines.Line2D([0], [0], color=color, linewidth=2.5, label=f"mean — {lvl}")
        )
    legend_handles.append(
        matplotlib.lines.Line2D([0], [0], color="black", linewidth=1.5,
                                linestyle="--", label="median")
    )
    ax_dist.legend(handles=legend_handles, fontsize=8, loc="upper left",
                   bbox_to_anchor=(0.0, 0.26),
                   bbox_transform=ax_dist.transAxes)

    ax_dist.set_title("Cache hit % — all turns\n(compaction turns excluded)",
                      fontsize=10, fontweight="bold")
    ax_dist.set_ylabel("Cache hit %", fontsize=9)
    ax_dist.set_ylim(45, 105)
    ax_dist.set_xticks(range(len(all_levels)))
    ax_dist.set_xticklabels(all_levels, fontsize=8, rotation=15, ha="right")
    ax_dist.grid(axis="y", alpha=0.25, linestyle="--")
    ax_dist.spines[["top", "right"]].set_visible(False)

    title = f"vLLM prefix cache hit rate" + (f" — {out_suffix}" if out_suffix else " by difficulty level")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fname = f"analysis_cache_{out_suffix}.png" if out_suffix else "analysis_cache.png"
    out = SCRIPT_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── distribution figure ───────────────────────────────────────────────────────

def plot_dist_figure(datasets: list[tuple[str, dict]], rng: np.random.Generator,
                     out_suffix: str = "") -> None:
    """
    Row 0: OSL log-binned frequency histogram per level, with region annotations.
    Row 1: ISL scatter plot (x = turn index, y = ISL value) per level.
    """
    all_levels: list[str] = []
    for _, grouped in datasets:
        for lvl in sorted_levels(grouped):
            if lvl not in all_levels:
                all_levels.append(lvl)

    n_levels = len(all_levels)
    # extra top margin so OSL region labels don't overlap titles
    fig, axes = plt.subplots(2, n_levels, figsize=(5 * n_levels, 10), squeeze=False)
    fig.subplots_adjust(top=0.88, hspace=0.45)

    # ── OSL regions annotation config ────────────────────────────────────────
    # (x_start, x_end, label)
    OSL_REGIONS = [
        (10,    100,   "tool call JSON\n(Glob/Read/Grep)"),
        (100,   1_000, "text + tool\ncombined"),
        (1_000, 5_000, "Edit/Write\nmoderate"),
        (5_000, 30_000,"large Write\n(full file)"),
    ]
    REGION_COLORS = ["#e3f2fd", "#fff9c4", "#fce4ec", "#e8f5e9"]
    bins_osl = np.logspace(1, 5.3, 45)

    tok_fmt = matplotlib.ticker.FuncFormatter(
        lambda v, _: f"{int(v):,}" if v < 1000 else f"{v/1000:.0f}k"
    )

    # ── row 0: OSL histograms ─────────────────────────────────────────────────
    for col, lvl in enumerate(all_levels):
        ax    = axes[0][col]
        color = LEVEL_COLOR.get(lvl, COLORS[col % len(COLORS)])

        vals = []
        for _, grouped in datasets:
            for r in grouped.get(lvl, []):
                vals.extend(max(v, 1) for v in r.get("osl_per_turn", []))

        arr_for_count = np.array(vals, dtype=float) if vals else np.array([])
        x_max = float(arr_for_count.max()) * 1.2 if len(arr_for_count) else 5e4
        x_max = max(x_max, 5e4)   # always show at least up to 50k

        # extend last region to actual data max
        regions_dynamic = list(OSL_REGIONS[:-1]) + [
            (OSL_REGIONS[-1][0], x_max, OSL_REGIONS[-1][2])
        ]

        if vals:
            arr = arr_for_count
            ax.hist(arr, bins=bins_osl, color=color, alpha=0.75,
                    edgecolor="white", linewidth=0.3, zorder=3)
            med  = float(np.median(arr))
            mean = float(np.mean(arr))
            ax.axvline(med,  color="black",  linewidth=1.5, linestyle="--",
                       label=f"med={med:,.0f}", zorder=4)
            ax.axvline(mean, color="dimgray", linewidth=1.2, linestyle=":",
                       label=f"mean={mean:,.0f}", zorder=4)
            ax.legend(fontsize=7, loc="lower right")

        # region shading + boundary lines + labels + counts
        from matplotlib.transforms import blended_transform_factory
        trans = blended_transform_factory(ax.transData, ax.transAxes)
        for (x0, x1, label), rc in zip(regions_dynamic, REGION_COLORS):
            ax.axvspan(x0, x1, color=rc, alpha=0.5, zorder=1)
            ax.axvline(x1, color="gray", linewidth=0.8, linestyle="--",
                       alpha=0.6, zorder=2)
            mid   = 10 ** ((np.log10(x0) + np.log10(x1)) / 2)
            count = int(((arr_for_count >= x0) & (arr_for_count < x1)).sum())
            pct   = 100 * count / len(arr_for_count) if len(arr_for_count) else 0
            ax.text(mid, 0.97, label, transform=trans,
                    ha="center", va="top", fontsize=6.5, color="#333",
                    multialignment="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, lw=0))
            ax.text(mid, 0.72, f"n={count}\n({pct:.0f}%)", transform=trans,
                    ha="center", va="top", fontsize=6.5, color="#333",
                    multialignment="center")

        ax.set_xscale("log")
        ax.set_xlim(10, x_max)
        ax.set_xlabel("OSL tokens (log scale)", fontsize=8)
        ax.set_ylabel("Count" if col == 0 else "", fontsize=8)
        n_total = len(arr_for_count)
        ax.set_title(f"OSL — {lvl}\nn={n_total} turns", fontsize=10, fontweight="bold")
        ax.xaxis.set_major_formatter(tok_fmt)
        ax.grid(axis="y", alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)

    # ── row 1: ISL per-instance lines + scatter, gradient-colored ────────────
    # Each level uses its own colormap (Blues / Reds / Greens),
    # light → dark across instances so individual runs are distinguishable.
    for col, lvl in enumerate(all_levels):
        ax    = axes[1][col]
        color = LEVEL_COLOR.get(lvl, COLORS[col % len(COLORS)])

        # collect all instances across datasets for this level
        instances = []
        for _, grouped in datasets:
            instances.extend(grouped.get(lvl, []))

        n_inst = len(instances)
        # alpha range: 0.25 (light, many instances) → 0.75 (dark, few instances)
        alphas = np.linspace(0.25, 0.75, max(n_inst, 1))

        all_turns_flat: list[int] = []
        for ii, (r, alpha) in enumerate(zip(instances, alphas)):
            isl_seq = r.get("isl_per_turn", [])
            if not isl_seq:
                continue
            xs = list(range(len(isl_seq)))
            ys = [float(v) for v in isl_seq]
            ax.plot(xs, ys, color=color, linewidth=1.0, alpha=alpha, zorder=2)
            ax.scatter(xs, ys, color=color, s=2, alpha=alpha, zorder=3)
            all_turns_flat.extend(xs)

        # bold mean line in the darkest shade
        if instances:
            max_t = max(len(r.get("isl_per_turn", [])) for r in instances)
            xs_m, means = [], []
            for t in range(max_t):
                vals = [float(r["isl_per_turn"][t])
                        for r in instances
                        if t < len(r.get("isl_per_turn", []))]
                if vals:
                    xs_m.append(t)
                    means.append(float(np.mean(vals)))
            ax.plot(xs_m, means, color="black", linewidth=2.5,
                    zorder=4, label="mean", linestyle="-")
            ax.legend(fontsize=7)

        ax.set_xlabel("Turn number", fontsize=8)
        ax.set_ylabel("ISL (tokens)" if col == 0 else "", fontsize=8)
        ax.set_title(f"ISL per turn — {lvl}", fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(tok_fmt)
        ax.grid(alpha=0.2, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.text(0.03, 0.97, f"n={n_inst} runs", transform=ax.transAxes,
                fontsize=7, va="top", color="gray")

    title = ("OSL distribution (log-binned) and ISL per turn"
             + (f" — {out_suffix}" if out_suffix else " by difficulty level"))
    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fname = f"analysis_dist_{out_suffix}.png" if out_suffix else "analysis_dist.png"
    out = SCRIPT_DIR / fname
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def _level_slug(lvl: str) -> str:
    return (lvl.replace(">", "gt").replace("<", "lt")
               .replace(" ", "_").replace("-", "_").replace("/", "_"))


def main() -> None:
    dirs = [Path(a) for a in sys.argv[1:]] if len(sys.argv) > 1 else [DEFAULT_DIR]

    datasets: list[tuple[str, dict]] = []
    for d in dirs:
        if not d.exists():
            print(f"ERROR: {d} does not exist", file=sys.stderr)
            sys.exit(1)
        raw       = load_dir(d)
        extracted = [e for r in raw if (e := extract(r)) is not None]
        grouped   = group_by_level(extracted)
        datasets.append((d.name, grouped))

        levels_summary = ", ".join(
            f"{lvl}={len(grouped[lvl])}" for lvl in sorted_levels(grouped)
        )
        print(f"  {d.name}: {len(extracted)} instances  [{levels_summary}]")

    if not datasets:
        print("No data found.")
        sys.exit(1)

    rng = np.random.default_rng(42)

    plot_cache_figure(datasets, rng)
    plot_dist_figure(datasets, rng)


if __name__ == "__main__":
    main()
