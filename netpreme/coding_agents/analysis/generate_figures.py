#!/usr/bin/env python3
"""
Generate comparison figures for hybrid-cpu vs hybrid-mtier benchmark results.

Produces 5 figures saved to benchmark/results_benchmarks/<run>/plots/:
    fig2  — TTFT speedup (CPU÷MTier) grouped bar, all percentiles (p50/p90/p95/p99)
    fig3  — Queue wait speedup grouped bar, all percentiles
    fig4  — TTFT p50 + p99 head-to-head line chart with speedup annotations
    fig5  — Throughput speedup: output tokens/s and tasks/hr (MTier÷CPU)
    fig1  — Hero 6-panel summary: cache hit rates, TTFT, throughput, E2E breakdown

What you need to run first:
    1. Run the benchmark for both setups across concurrency levels:
           python3 benchmark/run_concurrent.py --setup hybrid-cpu  --concurrency 4 8 12 16 20
           python3 benchmark/run_concurrent.py --setup hybrid-mtier --concurrency 4 8 12 16 20

    2. Combine the per-run CSVs into a single summary (or point RESULTS_DIR at an
       existing bench_combined_* directory that already has summary.csv + summary.json).

    3. Update RESULTS_DIR below to point at the combined results directory.

Usage:
    python3 analysis/generate_figures.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).parent.parent / "benchmark" / "results_benchmarks" / "bench_combined_20260418_040848"
OUT_DIR = RESULTS_DIR / "plots"
OUT_DIR.mkdir(exist_ok=True)

CPU_COLOR   = "#2196F3"   # blue
MTIER_COLOR = "#FF5722"   # deep orange

import json as _json

df    = pd.read_csv(RESULTS_DIR / "summary.csv")
cpu   = df[df["config"] == "hybrid-cpu"].sort_values("concurrency").reset_index(drop=True)
mtier = df[df["config"] == "hybrid-mtier"].sort_values("concurrency").reset_index(drop=True)
conc  = cpu["concurrency"].tolist()
x     = np.arange(len(conc))

# Load queue_ms and prefill_ms from the per-run directory that matches each
# CSV row's e2e_p50, since summary.json in the combined dir has stale entries
# for some concurrency levels that were re-run separately.
_BENCH_BASE = RESULTS_DIR.parent

def _load_matched_prom() -> dict:
    """Return {config: {field: {pct: [values ordered by conc]}}}."""
    # Cache all per-run json entries keyed by (config, concurrency)
    run_index: dict = {}  # (cfg, conc) -> list of json entries
    for d in sorted(_BENCH_BASE.iterdir()):
        if not d.is_dir() or d.name.startswith("bench_combined"): continue
        sj = d / "summary.json"
        if not sj.exists(): continue
        for entry in _json.load(open(sj)):
            key = (entry["config"], entry["concurrency"])
            run_index.setdefault(key, []).append(entry)

    result: dict = {}
    for _, row in df.iterrows():
        cfg = row["config"]; c = int(row["concurrency"]); target = row["e2e_p50"]
        best = min(run_index.get((cfg, c), []),
                   key=lambda e: abs((e.get("e2e_ms", {}).get("p50") or 1e9) - target),
                   default=None)
        result.setdefault(cfg, {}).setdefault("queue_ms",   {})[c] = best.get("queue_ms",   {}) if best else {}
        result.setdefault(cfg, {}).setdefault("prefill_ms", {})[c] = best.get("prefill_ms", {}) if best else {}
    return result

_prom = _load_matched_prom()

def _prom_series(cfg: str, field: str, pct: str) -> list:
    return [_prom[cfg][field].get(c, {}).get(pct) or 0.0 for c in conc]

QUEUE = {
    cfg: {p: _prom_series(cfg, "queue_ms", p) for p in ("p50", "p90", "p95", "p99")}
    for cfg in ("hybrid-cpu", "hybrid-mtier")
}
PREFILL = {
    cfg: {p: _prom_series(cfg, "prefill_ms", p) for p in ("p50", "p90", "p95", "p99")}
    for cfg in ("hybrid-cpu", "hybrid-mtier")
}

# p95 = solid (prominent), p50 = dashed (secondary), p90 = dash-dot, p99 = dotted
PCT_STYLES  = [("--", "o", 0.75),   # p50 — dashed
               ("-.", "s", 0.60),   # p90 — dash-dot
               ("-",  "^", 1.00),   # p95 — solid
               (":",  "D", 0.45)]   # p99 — dotted
PCTS        = ["p50", "p90", "p95", "p99"]
PCT_COLS    = {"ttft": ["ttft_p50","ttft_p90","ttft_p95","ttft_p99"],
               "e2e":  ["e2e_p50","e2e_p90","e2e_p95","e2e_p99"]}


def savefig(fig, name):
    p = OUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {p}")


def pct_legend(ax):
    """Legend: color = CPU/MTier, line style = p95 solid / p50 dashed."""
    handles = [
        mlines.Line2D([], [], color=CPU_COLOR,   linewidth=2, label="CPU"),
        mlines.Line2D([], [], color=MTIER_COLOR, linewidth=2, label="MTier"),
        mlines.Line2D([], [], color="gray", linestyle="-",  linewidth=2, label="p95 (solid)"),
        mlines.Line2D([], [], color="gray", linestyle="--", linewidth=2, label="p50 (dashed)"),
    ]
    ax.legend(handles=handles, fontsize=9, ncol=2)


def pair_legend(ax):
    """Legend for p95-solid / p50-dashed pair plots: color = setup, style = percentile."""
    handles = [
        mlines.Line2D([], [], color=CPU_COLOR,   linewidth=2, label="CPU"),
        mlines.Line2D([], [], color=MTIER_COLOR, linewidth=2, label="MTier"),
        mlines.Line2D([], [], color="gray", linestyle="-",  linewidth=2, label="p95 (solid)"),
        mlines.Line2D([], [], color="gray", linestyle="--", linewidth=2, label="p50 (dashed)"),
    ]
    ax.legend(handles=handles, fontsize=9, ncol=2)


def annotate_saturation(ax, data_space=True):
    """Draw vertical regime boundaries: HBM-not-saturated | HBM-full | Queue-saturated."""
    x1 = 10   if data_space else 1.5   # between c=8 and c=12
    x2 = 14   if data_space else 2.5   # between c=12 and c=16
    dx = 0.4  if data_space else 0.12  # text offset in data units
    t  = ax.get_xaxis_transform()      # x: data coords, y: axes fraction

    ax.axvline(x1, color="#1565C0", linestyle=":", linewidth=1.2, alpha=0.5, zorder=2)
    ax.text(x1 - dx, 0.97, "← HBM not\n   saturated yet",
            transform=t, ha="right", va="top",
            fontsize=7, color="#1565C0", style="italic")

    ax.axvline(x2, color="#555555", linestyle=":", linewidth=1.2, alpha=0.5, zorder=2)
    ax.text(x2 + dx, 0.97, "Queue\nsaturated →",
            transform=t, ha="left", va="top",
            fontsize=7, color="#555555", style="italic")


# ── Fig 2: TTFT speedup all percentiles grouped bar ──────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("TTFT Speedup: CPU ÷ MTier — All Percentiles  (>1 = MTier faster)", fontsize=13)

w = 0.18
offsets = [-1.5*w, -0.5*w, 0.5*w, 1.5*w]
shade = ["#1E88E5","#1565C0","#0D47A1","#082533"]

for col, pct, offset, c_ in zip(PCT_COLS["ttft"], PCTS, offsets, shade):
    speedups = (cpu[col].values / mtier[col].values).tolist()
    bars = ax.bar(x + offset, speedups, w, label=pct, color=c_, alpha=0.85)
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                f"{s:.1f}×", ha="center", fontsize=7, rotation=90, va="bottom")

ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="No difference")
ax.set_xticks(x); ax.set_xticklabels([f"c={c}" for c in conc], fontsize=10)
ax.set_ylabel("Speedup (CPU TTFT ÷ MTier TTFT)", fontsize=11)
ax.legend(fontsize=9, title="Percentile"); ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, ax.get_ylim()[1]*1.15)
annotate_saturation(ax, data_space=False)
fig.tight_layout()
savefig(fig, "fig2_ttft_speedup.png")


# ── Fig 3: Queue speedup all percentiles grouped bar ─────────────────────────
fig, ax = plt.subplots(figsize=(12, 6))
fig.suptitle("Queue Speedup: CPU ÷ MTier — All Percentiles  (>1 = MTier shorter queue)", fontsize=13)

for pct, offset, c_ in zip(PCTS, offsets, shade):
    speedups = [c/m for c, m in zip(QUEUE["hybrid-cpu"][pct], QUEUE["hybrid-mtier"][pct])]
    bars = ax.bar(x + offset, speedups, w, label=pct, color=c_, alpha=0.85)
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.03,
                f"{s:.1f}×", ha="center", fontsize=7, rotation=90, va="bottom")

ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="No difference")
ax.set_xticks(x); ax.set_xticklabels([f"c={c}" for c in conc], fontsize=10)
ax.set_ylabel("Speedup (CPU queue ÷ MTier queue)", fontsize=11)
ax.legend(fontsize=9, title="Percentile"); ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, ax.get_ylim()[1]*1.15)
annotate_saturation(ax, data_space=False)
fig.tight_layout()
savefig(fig, "fig3_queue_speedup.png")


# ── Fig 4: TTFT p50 + p99 head-to-head on one plot ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("CPU vs MTier — Median (p50) and Tail (p99) TTFT\nSpeedup annotations show CPU÷MTier",
             fontsize=13)

for ax, col, title in [
    (axes[0], "ttft_p50", "TTFT p50 (Median)"),
    (axes[1], "ttft_p99", "TTFT p99 (Tail — worst 1%)"),
]:
    ax.plot(conc, cpu[col],   color=CPU_COLOR,   marker="o", linewidth=2.5,
            markersize=9, label="hybrid-cpu")
    ax.plot(conc, mtier[col], color=MTIER_COLOR, marker="s", linewidth=2.5,
            markersize=9, label="hybrid-mtier")
    for i, c_ in enumerate(conc):
        speedup = cpu[col].iloc[i] / mtier[col].iloc[i]
        mid_y   = (cpu[col].iloc[i] + mtier[col].iloc[i]) / 2
        if speedup > 1.15:
            ax.annotate(f"{speedup:.1f}×", (c_, mid_y), ha="center",
                        fontsize=9, fontweight="bold", color=MTIER_COLOR,
                        bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax.set_yscale("log")
    ax.set_xlabel("Concurrency", fontsize=11); ax.set_ylabel("TTFT (ms)", fontsize=11)
    ax.set_title(title, fontsize=11); ax.set_xticks(conc); ax.set_xticklabels([str(c) for c in conc])
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3, which="both")
    annotate_saturation(ax)

fig.tight_layout()
savefig(fig, "fig4_ttft_head_to_head.png")


# ── Fig 5: Throughput speedup ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Throughput Speedup: MTier ÷ CPU  (>1 = MTier faster)", fontsize=13)

for ax, cpu_v, mtier_v, ylabel, title in [
    (axes[0],
     cpu["output_tok_per_sec"].tolist(), mtier["output_tok_per_sec"].tolist(),
     "MTier tok/s ÷ CPU tok/s", "Output Tokens/s Speedup"),
    (axes[1],
     [r*3600 for r in cpu["tasks_per_sec"]], [r*3600 for r in mtier["tasks_per_sec"]],
     "MTier tasks/hr ÷ CPU tasks/hr", "Tasks/hr Speedup"),
]:
    speedups = [m/c for m, c in zip(mtier_v, cpu_v)]
    bars = ax.bar(x, speedups, 0.5,
                  color=[MTIER_COLOR if s >= 1 else CPU_COLOR for s in speedups], alpha=0.85)
    ax.axhline(1.0, color="black", linewidth=1.5, linestyle="--", label="No difference")
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{s:.2f}×", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels([f"c={c}" for c in conc], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10); ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(speedups)*1.25)
    annotate_saturation(ax, data_space=False)

fig.tight_layout()
savefig(fig, "fig5_throughput_speedup.png")


# ── Fig 1: Hero 6-panel summary ─────────────────────────────────────────────
# Layout: [Cache hit | TTFT | Throughput]
#         [E2E breakdown | E2E latency | Queue wait]
fig, axes = plt.subplots(2, 3, figsize=(20, 12))
fig.suptitle(
    "MTier KV Offload vs CPU DRAM — Full Benchmark Summary\n"
    "Qwen3-Coder-30B-A3B-FP8  ·  SWE-bench hard+vhard  ·  TP=1  ·  1× H100  "
    "│  blue = hybrid-cpu  │  orange = hybrid-mtier",
    fontsize=13, y=0.99
)

# [0,0] Cache hit rates
ax = axes[0, 0]
ax.plot(conc, cpu["gpu_hit_pct"],   color=CPU_COLOR,   marker="o", linestyle="-",  linewidth=2, markersize=7, label="CPU GPU hit%")
ax.plot(conc, cpu["cpu_hit_pct"],   color=CPU_COLOR,   marker="s", linestyle="--", linewidth=2, markersize=7, label="CPU ext hit%")
ax.plot(conc, mtier["gpu_hit_pct"], color=MTIER_COLOR, marker="o", linestyle="-",  linewidth=2, markersize=7, label="MTier GPU hit%")
ax.plot(conc, mtier["cpu_hit_pct"], color=MTIER_COLOR, marker="s", linestyle="--", linewidth=2, markersize=7, label="MTier ext hit%")
ax.set_xticks(conc); ax.set_xticklabels([str(c) for c in conc]); ax.set_ylim(0, 110)
ax.set_xlabel("Concurrency"); ax.set_ylabel("Hit Rate (%)")
ax.set_title("Cache Hit Rates", fontsize=11)
ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)
annotate_saturation(ax)

# [0,1] TTFT p50 + p95
ax = axes[0, 1]
ax.plot(conc, cpu["ttft_p95"],   color=CPU_COLOR,   marker="^", linewidth=2,   markersize=7, label="CPU p95")
ax.plot(conc, mtier["ttft_p95"], color=MTIER_COLOR, marker="D", linewidth=2,   markersize=7, label="MTier p95")
ax.plot(conc, cpu["ttft_p50"],   color=CPU_COLOR,   marker="o", linewidth=1.5, markersize=6, linestyle="--", alpha=0.75, label="CPU p50")
ax.plot(conc, mtier["ttft_p50"], color=MTIER_COLOR, marker="s", linewidth=1.5, markersize=6, linestyle="--", alpha=0.75, label="MTier p50")
ax.set_yscale("log"); ax.set_xticks(conc); ax.set_xticklabels([str(c) for c in conc])
ax.set_xlabel("Concurrency"); ax.set_ylabel("TTFT (ms)")
ax.set_title("TTFT p95 + p50", fontsize=11)
pair_legend(ax); ax.grid(True, alpha=0.3, which="both")
annotate_saturation(ax)

# [0,2] Output tok/s p95 + p50
ax = axes[0, 2]
ax.plot(conc, cpu["turn_tps_p95"],   color=CPU_COLOR,   marker="^", linewidth=2,   markersize=7, label="CPU p95")
ax.plot(conc, mtier["turn_tps_p95"], color=MTIER_COLOR, marker="D", linewidth=2,   markersize=7, label="MTier p95")
ax.plot(conc, cpu["turn_tps_p50"],   color=CPU_COLOR,   marker="o", linewidth=1.5, markersize=6, linestyle="--", alpha=0.75, label="CPU p50")
ax.plot(conc, mtier["turn_tps_p50"], color=MTIER_COLOR, marker="s", linewidth=1.5, markersize=6, linestyle="--", alpha=0.75, label="MTier p50")
ax.set_xticks(conc); ax.set_xticklabels([str(c) for c in conc])
ax.set_xlabel("Concurrency"); ax.set_ylabel("Output Tokens / s")
ax.set_title("System Throughput (tok/s)", fontsize=11)
pair_legend(ax); ax.grid(True, alpha=0.3)
ax.set_ylim(0, ax.get_ylim()[1]*1.15)
annotate_saturation(ax)

# [1,1] E2E breakdown: queue + prefill + decode, p50 (solid) and p95 (hatched)
ax_e2e_bar = axes[1, 1]

Q_COLOR  = "#455A64"   # dark slate  — queue
PF_COLOR = "#FFA726"   # amber        — prefill
DC_COLOR = CPU_COLOR   # blue         — decode

bar_w = 0.14
p50_offs = np.array([-1.7, -0.7]) * bar_w   # CPU_p50, MTier_p50
p95_offs = np.array([ 0.7,  1.7]) * bar_w   # CPU_p95, MTier_p95

series_cfg = [
    ("hybrid-cpu",   "p50", cpu,   p50_offs[0]),
    ("hybrid-mtier", "p50", mtier, p50_offs[1]),
    ("hybrid-cpu",   "p95", cpu,   p95_offs[0]),
    ("hybrid-mtier", "p95", mtier, p95_offs[1]),
]

for i, xi in enumerate(x):
    for cfg, pct, dframe, off in series_cfg:
        t_val  = float(dframe[f"ttft_{pct}"].iloc[i])
        e_val  = float(dframe[f"e2e_{pct}"].iloc[i])
        dc_val = max(e_val - t_val, 0)
        # Normalize queue+prefill proportionally to fill exactly ttft.
        # Raw percentiles from separate histograms don't sum to ttft_pct,
        # so bars would be short and comparisons misleading.
        q_raw  = max(QUEUE[cfg][pct][i],   0)
        pf_raw = max(PREFILL[cfg][pct][i], 0)
        sub    = q_raw + pf_raw
        if sub > 0:
            q_val  = t_val * q_raw  / sub
            pf_val = t_val * pf_raw / sub
        else:
            q_val, pf_val = t_val, 0.0
        xpos     = xi + off
        ec       = MTIER_COLOR if cfg == "hybrid-mtier" else "black"
        kw       = dict(width=bar_w, edgecolor=ec, linewidth=0.7)
        ax_e2e_bar.bar(xpos, q_val,  bottom=0,           color=Q_COLOR,  **kw)
        ax_e2e_bar.bar(xpos, pf_val, bottom=q_val,        color=PF_COLOR, **kw)
        ax_e2e_bar.bar(xpos, dc_val, bottom=q_val+pf_val, color=DC_COLOR, **kw)

ax_e2e_bar.set_yscale("log")
ax_e2e_bar.set_xticks(x); ax_e2e_bar.set_xticklabels([str(c) for c in conc])
ax_e2e_bar.set_xlabel(""); ax_e2e_bar.set_ylabel("Latency (ms, log)")
ax_e2e_bar.set_title("E2E Time Breakdown per Request", fontsize=11)
ax_e2e_bar.grid(axis="y", alpha=0.3, which="both")

# p50 / p95 group labels
t_ax = ax_e2e_bar.get_xaxis_transform()
for xi in x:
    ax_e2e_bar.text(xi + np.mean(p50_offs), -0.07, "p50", transform=t_ax,
                    ha="center", va="top", fontsize=7, color="#444")
    ax_e2e_bar.text(xi + np.mean(p95_offs), -0.07, "p95", transform=t_ax,
                    ha="center", va="top", fontsize=7, color="#444")

leg = [
    mpatches.Patch(facecolor=Q_COLOR,  edgecolor="none", label="Queue"),
    mpatches.Patch(facecolor=PF_COLOR, edgecolor="none", label="Prefill"),
    mpatches.Patch(facecolor=DC_COLOR, edgecolor="none", label="Decode"),
    mpatches.Patch(facecolor="white", edgecolor=MTIER_COLOR, linewidth=1.5, label="MTier"),
    mpatches.Patch(facecolor="white", edgecolor="black",     linewidth=1.5, label="CPU"),
]
ax_e2e_bar.legend(handles=leg, fontsize=8, ncol=1, loc="upper left",
                  framealpha=0.9, edgecolor="#ccc")
annotate_saturation(ax_e2e_bar, data_space=False)

# [1,0] E2E latency line chart — shared y-axis with bar panel
ax_e2e_line = axes[1, 0]
ax_e2e_line.plot(conc, cpu["e2e_p95"],   color=CPU_COLOR,   marker="^", linewidth=2,   markersize=7, label="CPU p95")
ax_e2e_line.plot(conc, mtier["e2e_p95"], color=MTIER_COLOR, marker="D", linewidth=2,   markersize=7, label="MTier p95")
ax_e2e_line.plot(conc, cpu["e2e_p50"],   color=CPU_COLOR,   marker="o", linewidth=1.5, markersize=6, linestyle="--", alpha=0.75, label="CPU p50")
ax_e2e_line.plot(conc, mtier["e2e_p50"], color=MTIER_COLOR, marker="s", linewidth=1.5, markersize=6, linestyle="--", alpha=0.75, label="MTier p50")
ax_e2e_line.set_yscale("log")
ax_e2e_line.set_xticks(conc); ax_e2e_line.set_xticklabels([str(c) for c in conc])
ax_e2e_line.set_xlabel("Concurrency"); ax_e2e_line.set_ylabel("E2E Latency (ms, log)")
ax_e2e_line.set_title("E2E Latency p95 + p50", fontsize=11)
pair_legend(ax_e2e_line); ax_e2e_line.grid(True, alpha=0.3, which="both")
annotate_saturation(ax_e2e_line)

# Shared y-axis across E2E line, E2E bar, and Queue wait panels
all_e2e = np.concatenate([
    cpu["e2e_p50"].values, cpu["e2e_p95"].values,
    mtier["e2e_p50"].values, mtier["e2e_p95"].values,
])
e2e_ymin = max(all_e2e.min() * 0.7, 100)
e2e_ymax = all_e2e.max() * 1.5

# [1,2] unused
axes[1, 2].set_visible(False)

# Apply shared y-axis to both bottom E2E panels
for _ax in [ax_e2e_line, ax_e2e_bar]:
    _ax.set_ylim(e2e_ymin, e2e_ymax)

fig.tight_layout()
savefig(fig, "fig1_hero_summary.png")

print("\nAll figures saved to:", OUT_DIR)
