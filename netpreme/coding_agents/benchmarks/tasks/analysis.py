#!/usr/bin/env python3
"""
Analyze SWE-bench run results by difficulty level.

Produces a single figure with 5 panels (one per metric):
  - Avg ISL / turn     (mean input sequence length per vLLM call)
  - Avg OSL / turn     (mean output sequence length per vLLM call)
  - Total ISL          (sum of input tokens across all turns)
  - Total OSL          (sum of output tokens across all turns)
  - Avg tools / turn

Each panel: x = difficulty level, y = metric value.
Points = individual instances (jittered), bar = mean, shading = ±1 std.

Usage:
    python3 analysis.py                        # tasks/results/
    python3 analysis.py path/to/results/
    python3 analysis.py dir1/ dir2/            # overlay dirs for validation
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SCRIPT_DIR  = Path(__file__).resolve().parent
DEFAULT_DIR = SCRIPT_DIR / "results"
OUT_FILE    = SCRIPT_DIR / "analysis.png"

# Fixed instance sets (10 / 10 / 6) used for consistent comparisons.
# To disable filtering and use all available results, set to None.
SELECTED_IDS: set[str] | None = {
    # <15 min fix  (10)
    "astropy__astropy-14309",
    "astropy__astropy-14995",
    "astropy__astropy-7166",
    "astropy__astropy-7336",
    "django__django-10097",
    "django__django-10880",
    "django__django-10914",
    "django__django-10999",
    "django__django-11066",
    "django__django-11099",
    # 15 min - 1 hour  (10)
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13453",
    "astropy__astropy-13977",
    "astropy__astropy-14096",
    "astropy__astropy-14182",
    "astropy__astropy-14365",
    "astropy__astropy-14508",
    "astropy__astropy-14539",
    # 1-4 hours  (6 — all available)
    "astropy__astropy-13398",
    "astropy__astropy-13579",
    "astropy__astropy-14369",
    "django__django-10554",
    "django__django-11138",
    "django__django-11400",
}

# difficulty levels in logical order
LEVEL_ORDER = ["<15 min fix", "15 min - 1 hour", "1-4 hours", ">4 hours", "unknown"]

METRICS = [
    ("isl_per_turn",       "ISL / turn",        "tokens"),   # all per-turn values
    ("osl_per_turn",       "OSL / turn",        "tokens"),   # all per-turn values
    ("total_isl",          "Total ISL",         "tokens"),
    ("total_osl",          "Total OSL",         "tokens"),
    ("avg_tools_per_turn", "Avg tools / turn",  "count"),
]

COLORS  = ["#2196F3", "#E91E63", "#4CAF50", "#FF9800", "#9C27B0"]
MARKERS = ["o", "s", "^", "D", "v"]


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
    # Turn 0: cold start → 0 %
    # Turn N: cached prefix = prev_isl + prev_osl; new = tool result tokens
    # hit % = (prev_isl + prev_osl) / curr_isl * 100   [capped at 100]
    cache_hits: list[float] = [0.0]   # turn 0 always cold
    total_cached = 0.0
    for i in range(1, len(turns)):
        p_isl = turns[i - 1].get("isl") or 0
        p_osl = turns[i - 1].get("osl") or 0
        c_isl = turns[i].get("isl") or 0
        hit   = min((p_isl + p_osl) / c_isl * 100, 100.0) if c_isl else 0.0
        cache_hits.append(hit)
        total_cached += min(p_isl + p_osl, c_isl)

    total_input         = sum(t.get("isl") or 0 for t in turns)
    overall_cache_hit   = total_cached / total_input * 100 if total_input else 0.0
    avg_cache_hit       = float(np.mean(cache_hits[1:])) if len(cache_hits) > 1 else 0.0

    return {
        "instance_id":        record["instance_id"],
        "difficulty":         str(record.get("difficulty") or "unknown"),
        "avg_isl_per_turn":   float(np.mean(isl)),
        "avg_osl_per_turn":   float(np.mean(osl)),
        "total_isl":          float(final_isl if final_isl else sum(isl)),
        "total_osl":          float(summary.get("total_osl") or sum(osl)),
        "avg_tools_per_turn": float(summary.get("avg_tools_per_turn") or 0),
        "total_turns":        len(turns),
        "isl_per_turn":       isl,                  # raw ISL value at each turn
        "osl_per_turn":       osl,                  # raw OSL value at each turn
        "cache_hits_per_turn": cache_hits,          # estimated hit % at each turn
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


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_panel(ax, datasets: list[tuple[str, dict]], metric: str, title: str, ylabel: str,
               rng: np.random.Generator) -> None:
    """Plot one metric panel with all datasets overlaid."""
    # collect all levels across datasets to fix x positions
    all_levels: list[str] = []
    for _, grouped in datasets:
        for lvl in sorted_levels(grouped):
            if lvl not in all_levels:
                all_levels.append(lvl)

    n_ds     = len(datasets)
    width    = 0.6 / max(n_ds, 1)   # per-dataset strip width within each level slot

    for j, (name, grouped) in enumerate(datasets):
        color  = COLORS[j % len(COLORS)]
        marker = MARKERS[j % len(MARKERS)]
        offset = (j - (n_ds - 1) / 2) * width

        for xi, lvl in enumerate(all_levels):
            raw = [r.get(metric) for r in grouped.get(lvl, [])]
            # flatten: if values are lists (per-turn), expand them
            vals = []
            for v in raw:
                if isinstance(v, list):
                    vals.extend(v)
                elif v is not None:
                    vals.append(v)
            if not vals:
                continue
            arr = np.array(vals, dtype=float)
            jx  = xi + offset + rng.uniform(-width * 0.4, width * 0.4, len(arr))
            ax.scatter(jx, arr, color=color, marker=marker, alpha=0.65, s=45, zorder=3)

            m, s   = arr.mean(), arr.std()
            x0, x1 = xi + offset - width * 0.45, xi + offset + width * 0.45
            ax.hlines(m, x0, x1, colors=color, linewidths=2.5, zorder=4)
            ax.fill_between([x0, x1], [m - s, m - s], [m + s, m + s],
                            color=color, alpha=0.18, zorder=2)

    # auto y-limit: clip at IQR fence (Q3 + 1.5×IQR) so outliers don't squash the view
    all_vals: list[float] = []
    for _, grouped in datasets:
        for lvl in all_levels:
            for r in grouped.get(lvl, []):
                v = r.get(metric)
                if isinstance(v, list):
                    all_vals.extend(float(x) for x in v if x is not None)
                elif v is not None:
                    all_vals.append(float(v))
    if all_vals:
        arr_all = np.array(all_vals)
        q1, q3  = np.percentile(arr_all, 25), np.percentile(arr_all, 75)
        fence   = q3 + 1.5 * (q3 - q1)
        y_max   = fence * 1.05
        ax.set_ylim(bottom=max(0, arr_all.min() * 0.95), top=y_max)

    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_xticks(range(len(all_levels)))
    ax.set_xticklabels(all_levels, fontsize=8, rotation=15, ha="right")
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(
            lambda v, _: f"{v/1000:.0f}k" if v >= 1000 else f"{v:.1f}"
        )
    )


def summary_panel(ax, datasets: list[tuple[str, dict]]) -> None:
    ax.axis("off")
    lines = ["Instances loaded\n"]
    for name, grouped in datasets:
        total = sum(len(v) for v in grouped.values())
        lines.append(f"{name}  ({total} runs)")
        for lvl in sorted_levels(grouped):
            recs = grouped[lvl]
            turns = [r["total_turns"] for r in recs]
            lines.append(
                f"  {lvl:<20} n={len(recs):<3}  "
                f"turns {min(turns)}-{max(turns)} (avg {np.mean(turns):.0f})"
            )
        lines.append("")
    ax.text(0.04, 0.97, "\n".join(lines), transform=ax.transAxes,
            fontsize=8, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#f5f5f5", alpha=0.8))


# ── cache figure ──────────────────────────────────────────────────────────────

def plot_cache_figure(datasets: list[tuple[str, dict]], rng: np.random.Generator) -> None:
    """Second figure: cache hit % evolution (line) + overall distribution (scatter)."""
    fig, (ax_line, ax_dist) = plt.subplots(1, 2, figsize=(14, 5))

    # ── left: per-turn mean ± std evolution ──────────────────────────────────
    for j, (name, grouped) in enumerate(datasets):
        ds_label = name if len(datasets) > 1 else None
        for li, lvl in enumerate(sorted_levels(grouped)):
            color  = COLORS[li % len(COLORS)]
            ls     = ["-", "--", ":", "-."][j % 4]
            series = [r["cache_hits_per_turn"] for r in grouped[lvl]]
            max_t  = max(len(s) for s in series)

            xs, means, stds = [], [], []
            for t in range(max_t):
                vals = [s[t] for s in series if t < len(s)]
                if len(vals) >= 2:
                    xs.append(t)
                    means.append(float(np.mean(vals)))
                    stds.append(float(np.std(vals)))

            lbl = f"{lvl}" + (f" [{ds_label}]" if ds_label else "")
            ax_line.plot(xs, means, color=color, linestyle=ls, linewidth=2, label=lbl)
            ax_line.fill_between(xs,
                                 [m - s for m, s in zip(means, stds)],
                                 [m + s for m, s in zip(means, stds)],
                                 color=color, alpha=0.12)

    ax_line.set_title("Cache hit % per turn\n(mean ± 1 std across instances)",
                      fontsize=11, fontweight="bold")
    ax_line.set_xlabel("Turn number", fontsize=9)
    ax_line.set_ylabel("Cache hit %", fontsize=9)
    ax_line.set_ylim(-5, 105)
    ax_line.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax_line.legend(fontsize=8, loc="lower right")
    ax_line.grid(alpha=0.2, linestyle="--")
    ax_line.spines[["top", "right"]].set_visible(False)

    # ── right: overall cache hit % per instance ───────────────────────────────
    all_levels: list[str] = []
    for _, grouped in datasets:
        for lvl in sorted_levels(grouped):
            if lvl not in all_levels:
                all_levels.append(lvl)

    n_ds  = len(datasets)
    width = 0.6 / max(n_ds, 1)

    for j, (name, grouped) in enumerate(datasets):
        color  = COLORS[j % len(COLORS)]
        marker = MARKERS[j % len(MARKERS)]
        offset = (j - (n_ds - 1) / 2) * width

        for xi, lvl in enumerate(all_levels):
            vals = [r["overall_cache_hit_pct"] for r in grouped.get(lvl, [])]
            if not vals:
                continue
            arr    = np.array(vals, dtype=float)
            jx     = xi + offset + rng.uniform(-width * 0.4, width * 0.4, len(arr))
            ax_dist.scatter(jx, arr, color=color, marker=marker, alpha=0.65, s=50, zorder=3)
            m, s   = arr.mean(), arr.std()
            x0, x1 = xi + offset - width * 0.45, xi + offset + width * 0.45
            ax_dist.hlines(m, x0, x1, colors=color, linewidths=2.5, zorder=4)
            ax_dist.fill_between([x0, x1], [m - s, m - s], [m + s, m + s],
                                 color=color, alpha=0.18, zorder=2)
            ax_dist.annotate(f"{m:.1f}%", xy=(xi + offset, m + s + 1),
                             ha="center", fontsize=7, color=color)

    ax_dist.set_title("Overall cache hit % per instance\n(mean ± 1 std)",
                      fontsize=11, fontweight="bold")
    ax_dist.set_ylabel("Cache hit %", fontsize=9)
    ax_dist.set_ylim(-5, 105)
    ax_dist.set_xticks(range(len(all_levels)))
    ax_dist.set_xticklabels(all_levels, fontsize=8, rotation=15, ha="right")
    ax_dist.grid(axis="y", alpha=0.25, linestyle="--")
    ax_dist.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "vLLM prefix cache hit rate by difficulty level\n"
        "(estimated: hit = (prev_isl + prev_osl) / curr_isl)",
        fontsize=12,
    )
    fig.tight_layout()
    out = SCRIPT_DIR / "analysis_cache.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


# ── distribution figure ───────────────────────────────────────────────────────

def plot_dist_figure(datasets: list[tuple[str, dict]], rng: np.random.Generator) -> None:
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
        color = COLORS[col % len(COLORS)]

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
    LEVEL_CMAPS = [plt.cm.Blues, plt.cm.Reds, plt.cm.Greens,
                   plt.cm.Purples, plt.cm.Oranges]

    for col, lvl in enumerate(all_levels):
        ax   = axes[1][col]
        cmap = LEVEL_CMAPS[col % len(LEVEL_CMAPS)]

        # collect all instances across datasets for this level
        instances = []
        for _, grouped in datasets:
            instances.extend(grouped.get(lvl, []))

        n_inst  = len(instances)
        # shade range: 0.30 (light) → 0.85 (dark)
        shades  = np.linspace(0.30, 0.85, max(n_inst, 1))

        all_turns_flat: list[int] = []
        for ii, (r, shade) in enumerate(zip(instances, shades)):
            isl_seq = r.get("isl_per_turn", [])
            if not isl_seq:
                continue
            color  = cmap(shade)
            xs     = list(range(len(isl_seq)))
            ys     = [float(v) for v in isl_seq]
            # connecting line
            ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.6, zorder=2)
            # dots on top
            ax.scatter(xs, ys, color=color, s=12, alpha=0.8, zorder=3)
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

    fig.suptitle(
        "OSL distribution (log-binned) and ISL per turn by difficulty level",
        fontsize=12,
    )
    fig.tight_layout()
    out = SCRIPT_DIR / "analysis_dist.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

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
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    axes = axes.flatten()

    for i, (metric, title, ylabel) in enumerate(METRICS):
        plot_panel(axes[i], datasets, metric, title, ylabel, rng)

    summary_panel(axes[5], datasets)

    # legend for multiple datasets
    if len(datasets) > 1:
        handles = [
            mpatches.Patch(color=COLORS[j % len(COLORS)], label=name)
            for j, (name, _) in enumerate(datasets)
        ]
        fig.legend(handles=handles, loc="lower right",
                   fontsize=9, framealpha=0.8, title="Dataset")

    fig.suptitle(
        "SWE-bench run metrics by difficulty level\n"
        "(dots = per-instance · bar = mean · shading = ±1 std)",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(OUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT_FILE}")

    plot_cache_figure(datasets, rng)
    plot_dist_figure(datasets, rng)


if __name__ == "__main__":
    main()
