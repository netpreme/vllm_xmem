#!/usr/bin/env python3
"""
Per-request E2E time decomposition + bytes transferred, mtier vs cpu.

Two panels (one figure):
  Left  — absolute clock time stacked (queue / prefill / decode), in seconds.
  Right — bytes transferred from the offload tier per request, in MB.
          Two bars per concurrency (cpu, mtier), with total GB moved over
          the full run annotated above each bar.

X = concurrency. Data: tier12_summary_per_run.csv (tier="full"), pooled
across iters. KV size assumed 54 KB/token (Qwen3-Coder 30B FP8 KV).
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT_DEFAULT = Path(__file__).resolve().parents[1] / "benchmarks" / "results_benchmarks"

_KV_BYTES_PER_TOKEN = 54_000

_PHASES = [
    ("queue",   "Queue wait",     "#9ecae1"),  # light blue
    ("prefill", "Prefill phase",  "#76B900"),  # NVIDIA green — the bandwidth-relevant phase
    ("decode",  "Decode phase",   "#e66101"),  # orange
]
_SETUP_COLOR = {"hybrid-cpu": "#e66101", "hybrid-mtier": "#5e3c99"}


def load_means(csv_path: Path, tier: str = "full"):
    """Return means[(setup, concurrency)] = phase averages + bytes-transferred stats."""
    grouped = defaultdict(lambda: {"queue": [], "prefill": [], "decode": [], "e2e": [],
                                   "off_tok_per_req": [], "off_tok_total": []})
    for r in csv.DictReader(open(csv_path)):
        if r.get("tier") != tier:
            continue
        try:
            c = int(float(r["concurrency"]))
            setup = r["setup"]
            queue_s   = (float(r["queue_ms"])   / 1000.0) if r["queue_ms"]   else None
            prefill_s = float(r["prefill_s"])              if r["prefill_s"] else None
            decode_s  = float(r["decode_s"])               if r["decode_s"]  else None
            e2e_s     = float(r["e2e_s"])                  if r["e2e_s"]     else None
            n_req     = float(r["ttft_ms_n_req"])          if r["ttft_ms_n_req"] else 0
            off_tok   = float(r["offload_tokens"])         if r["offload_tokens"] else 0
        except Exception:
            continue
        if None in (queue_s, prefill_s, decode_s):
            continue
        grouped[(setup, c)]["queue"].append(queue_s)
        grouped[(setup, c)]["prefill"].append(prefill_s)
        grouped[(setup, c)]["decode"].append(decode_s)
        grouped[(setup, c)]["e2e"].append(e2e_s if e2e_s is not None else queue_s + prefill_s + decode_s)
        grouped[(setup, c)]["off_tok_total"].append(off_tok)
        if n_req > 0:
            grouped[(setup, c)]["off_tok_per_req"].append(off_tok / n_req)

    out = {}
    for k, lists in grouped.items():
        d = {kk: (mean(vv) if vv else 0.0) for kk, vv in lists.items() if kk != "off_tok_total"}
        d["off_tok_total_sum"] = sum(lists["off_tok_total"])
        out[k] = d
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    ap.add_argument("--tier", default="full", choices=("full", "offload>50%"),
                    help="which CSV slice to plot (default: full run)")
    ap.add_argument("--out",  type=Path, default=None)
    args = ap.parse_args()

    csv_path = args.root / "tier12_summary_per_run.csv"
    if not csv_path.exists():
        raise SystemExit(f"missing: {csv_path}")
    means = load_means(csv_path, tier=args.tier)

    concs  = sorted({c for (_, c) in means.keys()})
    setups = ["hybrid-cpu", "hybrid-mtier"]
    setup_label = {"hybrid-cpu": "cpu", "hybrid-mtier": "mtier"}

    bar_w = 0.36
    x = np.arange(len(concs))

    fig, (ax_abs, ax_tot) = plt.subplots(1, 2, figsize=(17, 6))

    # ── LEFT: absolute clock-time stacked bars ────────────────────────────────
    for i, setup in enumerate(setups):
        offset = (-0.5 + i) * bar_w
        bottoms = np.zeros(len(concs))
        for phase_key, phase_label, color in _PHASES:
            vals = np.array([means.get((setup, c), {}).get(phase_key, 0.0) for c in concs])
            lbl = phase_label if i == 0 else None
            ax_abs.bar(x + offset, vals, bar_w, bottom=bottoms,
                       color=color, label=lbl, edgecolor="white", linewidth=0.5)
            bottoms += vals
        for j, c in enumerate(concs):
            ax_abs.text(x[j] + offset, -0.6, setup_label[setup],
                        ha="center", va="top", fontsize=7.5,
                        color=_SETUP_COLOR[setup])
    ax_abs.set_xticks(x); ax_abs.set_xticklabels([f"C={c}" for c in concs])
    ax_abs.set_xlabel("Concurrency")
    ax_abs.set_ylabel("Mean per-request time (s)")
    ax_abs.set_title("E2E breakdown — absolute clock time")
    ax_abs.grid(axis="y", alpha=0.3)
    ax_abs.legend(loc="upper left", fontsize=10)

    # ── RIGHT: TB transferred TOTAL (across all iters at this C) ─────────────
    bars = []  # (offset, tb_total array, setup)
    for i, setup in enumerate(setups):
        offset = (-0.5 + i) * bar_w
        tb_total = np.array([
            means.get((setup, c), {}).get("off_tok_total_sum", 0.0) * _KV_BYTES_PER_TOKEN / 1e12
            for c in concs
        ])
        bars.append((offset, tb_total, setup))

    y_max_tot = max(float(b[1].max()) for b in bars) if bars else 1.0
    ax_tot.set_ylim(0, y_max_tot * 1.18)
    for offset, tb_total, setup in bars:
        ax_tot.bar(x + offset, tb_total, bar_w,
                   color=_SETUP_COLOR[setup], label=setup_label[setup],
                   edgecolor="white", linewidth=0.5)
        for j, c in enumerate(concs):
            if tb_total[j] > 0:
                ax_tot.text(x[j] + offset, tb_total[j] + y_max_tot * 0.015,
                            f"{tb_total[j]:.1f}", ha="center", va="bottom",
                            fontsize=8, color=_SETUP_COLOR[setup])
        for j, c in enumerate(concs):
            ax_tot.text(x[j] + offset, -y_max_tot * 0.03, setup_label[setup],
                        ha="center", va="top", fontsize=7.5,
                        color=_SETUP_COLOR[setup])
    ax_tot.set_xticks(x); ax_tot.set_xticklabels([f"C={c}" for c in concs])
    ax_tot.set_xlabel("Concurrency")
    ax_tot.set_ylabel("TB transferred from offload, total across all iters")
    ax_tot.set_title("Offload data volume — total across all iters at this C")
    ax_tot.grid(axis="y", alpha=0.3)
    ax_tot.legend(loc="upper left", fontsize=10)

    regime = "full run" if args.tier == "full" else "offload-dominant windows (cache-hit > 50%)"
    fig.suptitle(f"Per-request time decomposition  +  offload data volume  —  mtier vs CPU\n"
                 f"({regime})",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    default_name = ("phase_breakdown.png" if args.tier == "full"
                    else "phase_breakdown_offload.png")
    out = args.out or args.root / default_name
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


if __name__ == "__main__":
    main()
