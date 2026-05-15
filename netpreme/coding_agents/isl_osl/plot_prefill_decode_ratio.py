#!/usr/bin/env python3
"""
Stacked average latency by turn index:
  x = turn index within problem
  y = average ttft_ms (prefill) stacked with average decode_ms (decode)
  sum = average e2e latency at that turn index

Filters:
  - category != "empty"
  - ttft_ms, decode_ms finite and >= 0; at least one > 0
  - isl, osl finite and > 0

Usage:
  python3 plot_prefill_decode_ratio.py \
      --run-dir runs/20260513_175826 \
      --out analysis/analysis_prefill_decode_ratio.png \
      --title-suffix "claude × Verified"
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def num(r: dict, k: str) -> float:
    v = r.get(k)
    if v in (None, "", "None"):
        return float("nan")
    try:
        return float(v)
    except ValueError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title-suffix", default="")
    ap.add_argument("--min-samples", type=int, default=10,
                    help="Truncate turn-index axis where fewer problems remain.")
    args = ap.parse_args()

    ttft, decode, isl, osl, turn_idx = [], [], [], [], []
    for f in sorted((args.run_dir / "per_problem").glob("*.csv")):
        with f.open() as fh:
            ti = 0
            for r in csv.DictReader(fh):
                if r.get("category") == "empty":
                    continue
                t = num(r, "ttft_ms"); d = num(r, "decode_ms")
                i = num(r, "isl");      o = num(r, "osl")
                if not all(np.isfinite(x) for x in (t, d, i, o)):
                    continue
                if t < 0 or d < 0 or i <= 0 or o <= 0 or t + d <= 0:
                    continue
                ti += 1
                ttft.append(t); decode.append(d)
                isl.append(i); osl.append(o); turn_idx.append(ti)
    ttft = np.array(ttft); decode = np.array(decode); turn_idx = np.array(turn_idx)
    print(f"plotting {len(ttft)} turns (after filtering)")

    max_ti = int(turn_idx.max())
    xs, counts, avg_ttft, avg_decode = [], [], [], []
    for ti in range(1, max_ti + 1):
        m = turn_idx == ti
        n = int(m.sum())
        if n < args.min_samples:
            continue
        xs.append(ti); counts.append(n)
        avg_ttft.append(float(ttft[m].mean()))
        avg_decode.append(float(decode[m].mean()))
    xs = np.array(xs); counts = np.array(counts)
    avg_ttft = np.array(avg_ttft); avg_decode = np.array(avg_decode)
    avg_e2e = avg_ttft + avg_decode
    print(f"per-turn axis: turns 1..{xs.max() if len(xs) else 0} "
          f"with ≥{args.min_samples} samples")

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(
        f"Average e2e latency = prefill + decode, by turn index{suffix}",
        fontsize=12,
    )

    ax.stackplot(
        xs,
        [avg_ttft, avg_decode],
        labels=["prefill (ttft)", "decode"],
        colors=["#3b82f6", "#22c55e"],
        alpha=0.85,
    )
    ax.plot(xs, avg_e2e, color="black", lw=1.2, label="e2e (sum)")
    ax.set_xlabel("turn index within problem")
    ax.set_ylabel("average latency (ms)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(True, ls="--", alpha=0.3)
    ax.set_xlim(xs.min() if len(xs) else 0, xs.max() if len(xs) else 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
