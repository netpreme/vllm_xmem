#!/usr/bin/env python3
"""
2×3 subplot summary across concurrency levels for mtier vs cpu.

Each subplot: x = concurrency (categorical), y = one metric, two lines:
mtier (purple) and cpu (orange). Error bars = ±1 stddev across iterations.

Loads tier12_summary_per_run.csv (one row per [run, setup, tier="full"])
written by compute_tier12_summary.py.
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DEFAULT = Path(__file__).resolve().parents[1] / "benchmarks" / "results_benchmarks"

_COLORS = {"hybrid-mtier": "#5e3c99", "hybrid-cpu": "#e66101"}
_LABELS = {"hybrid-mtier": "mtier",   "hybrid-cpu": "CPU offload"}

# (csv_key, label, scale, ylabel)
# Qwen3-Coder 30B FP8 KV size per token (bytes). 42 GB HBM ÷ 776k token budget.
_KV_BYTES_PER_TOKEN = 54_000

METRICS = [
    ("ttft_ms",        "TTFT",                                1.0,    "TTFT (ms)"),
    ("e2e_s",          "E2E latency",                         1.0,    "E2E latency (s)"),
    ("prefill_s",      "Prefill phase",                       1000.0, "Prefill (ms)"),
    ("decode_s",       "Decode phase",                        1.0,    "Decode (s)"),
    ("itl_ms",         "ITL (per output token)",              1.0,    "ITL (ms)"),
    ("__rps__",        "Throughput",                          1.0,    "Requests / s"),
    ("__off_gbps__",   "Offload bandwidth sustained",         1.0,    "Offload GB / s"),
    ("__offdomfrac__", "Time in offload-dominant regime",     100.0,  "% of wall time (>50%)"),
]


def load_rows(csv_path: Path, tier: str) -> dict:
    """rows[(setup, concurrency)] = list of dict rows for the requested tier.

    Each row is augmented with derived fields:
        __rps__        = ttft_count / duration (req/sec, for this tier's window)
        __off_gbps__   = offload_tokens * 54KB / duration / 1e9 (offload GB/s)
        __offdomfrac__ = duration(tier="offload>50%") / duration(tier="full")
                         — computed by pairing rows on (level_dir)
    """
    # Index every row in the CSV
    all_rows: list[dict] = list(csv.DictReader(open(csv_path)))
    # Pair tiers by level_dir (one mtier full + one offload>50%, one cpu full + one offload>50%)
    paired: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for r in all_rows:
        key = (r.get("level_dir", ""), r.get("setup", ""))
        paired[key][r["tier"]] = r

    grouped = defaultdict(list)
    for r in all_rows:
        if r["tier"] != tier: continue
        try:
            r["concurrency"] = int(float(r["concurrency"]))
        except Exception:
            continue
        # Per-tier req/s + offload GB/s
        try:
            n_req = float(r.get("ttft_ms_n_req") or 0)
            dur = float(r.get("duration_s") or 0)
            off_tok = float(r.get("offload_tokens") or 0)
            r["__rps__"]      = (n_req / dur) if dur > 0 else None
            r["__off_gbps__"] = (off_tok * _KV_BYTES_PER_TOKEN / 1e9 / dur) if dur > 0 else None
        except Exception:
            r["__rps__"] = None; r["__off_gbps__"] = None
        # Offload-dominant fraction — only meaningful on tier="full" rows
        try:
            pair = paired.get((r.get("level_dir", ""), r.get("setup", "")), {})
            full_dur = float(pair.get("full", {}).get("duration_s") or 0)
            off_dur  = float(pair.get("offload>50%", {}).get("duration_s") or 0)
            r["__offdomfrac__"] = (off_dur / full_dur) if full_dur > 0 else 0.0
        except Exception:
            r["__offdomfrac__"] = 0.0
        grouped[(r["setup"], r["concurrency"])].append(r)
    return grouped


def mean_std(rows: list, key: str, scale: float):
    vals = []
    for r in rows:
        v = r.get(key)
        if v in (None, "", "None"): continue
        try:
            vals.append(float(v) * scale)
        except Exception:
            continue
    if not vals:
        return (None, None, 0)
    return (mean(vals), pstdev(vals) if len(vals) > 1 else 0.0, len(vals))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    ap.add_argument("--tier", default="full", choices=("full", "offload>50%"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    csv_path = args.root / "tier12_summary_per_run.csv"
    if not csv_path.exists():
        raise SystemExit(f"missing: {csv_path}")
    grouped = load_rows(csv_path, args.tier)
    if not grouped:
        raise SystemExit(f"no rows with tier={args.tier!r} in {csv_path}")

    concs = sorted({c for (_, c) in grouped.keys()})
    setups = ["hybrid-cpu", "hybrid-mtier"]
    x = np.arange(len(concs))

    fig, axes = plt.subplots(2, 4, figsize=(20, 9), squeeze=False)
    for idx, (key, title, scale, ylabel) in enumerate(METRICS):
        ax = axes[idx // 4][idx % 4]
        for setup in setups:
            ys, errs = [], []
            for c in concs:
                rs = grouped.get((setup, c), [])
                m, s, _ = mean_std(rs, key, scale)
                ys.append(m); errs.append(s if s is not None else 0)
            ys_arr   = np.array([np.nan if v is None else v for v in ys], dtype=float)
            errs_arr = np.array([0 if v is None else v for v in errs], dtype=float)
            ax.errorbar(x, ys_arr, yerr=errs_arr,
                        color=_COLORS[setup], marker="o", markersize=7,
                        linewidth=2, capsize=4, capthick=1.2,
                        label=_LABELS[setup])
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in concs])
        ax.set_xlabel("Concurrency (C)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.set_ylim(bottom=0)
        if idx == 0:
            ax.legend(loc="upper left", fontsize=10)

    suffix = "full run" if args.tier == "full" else "offload-dominant windows (>50%)"
    fig.suptitle(f"mtier vs CPU offload across concurrency — {suffix}\n"
                 f"(error bars = ±1σ across iterations)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    out = args.out or args.root / (
        "concurrency_sweep_full.png" if args.tier == "full"
        else "concurrency_sweep_offload.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
