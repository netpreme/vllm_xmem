#!/usr/bin/env python3
"""
Per-turn TTFT vs prefill workload, two-panel view + tier-3 fit.

Prefill cost decomposes as:
  TTFT ≈ β₀ + γ·isl_cached + α·isl_new + δ·isl_new·isl
          └ overhead └ cache load └─ FFN ─┘ └── attention ──┘

Left:  TTFT vs isl_new           — FFN-only view (cannot predict TTFT alone)
Right: TTFT vs (isl_new · isl)   — attention-work view, dominates for long ctx

Points colored by isl (total context size in tokens). A 4-parameter OLS fit
is reported in the suptitle along with the crossover ISL at which the
attention term overtakes the FFN term (= α / δ). A third panel shows TTFT
vs cache_hit_rate so the cache-hit distribution and its TTFT effect are
both visible.

Filters:
  - category != "empty"
  - ttft_ms finite and > 0
  - isl > 0

Usage:
  python3 plot_ttft_prefill.py \
      --run-dir runs/20260513_175826 \
      --out ~/analysis_ttft_prefill.png \
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


def load_turns(run_dir: Path) -> list[dict]:
    out = []
    for f in sorted((run_dir / "per_problem").glob("*.csv")):
        with f.open() as fh:
            out.extend(csv.DictReader(fh))
    return out


def fit_tier3(ttft, isl_new, isl_cached, isl):
    """OLS: TTFT = β₀ + γ·isl_cached + α·isl_new + δ·isl_new·isl. Returns (coef, pred, R²)."""
    X = np.column_stack([
        np.ones_like(ttft),
        isl_cached,
        isl_new,
        isl_new * isl,
    ])
    coef, *_ = np.linalg.lstsq(X, ttft, rcond=None)
    pred = X @ coef
    ss_res = np.sum((ttft - pred) ** 2)
    ss_tot = np.sum((ttft - ttft.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coef, pred, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title-suffix", default="")
    args = ap.parse_args()

    rows = load_turns(args.run_dir)

    ttft, isl_new, isl_cached, isl = [], [], [], []
    for r in rows:
        if r.get("category") == "empty":
            continue
        t = num(r, "ttft_ms")
        i = num(r, "isl")
        i_new = num(r, "isl_new")
        i_cached = num(r, "isl_cached")
        if not all(np.isfinite(x) for x in (t, i, i_new, i_cached)):
            continue
        if t <= 0 or i <= 0:
            continue
        ttft.append(t); isl.append(i)
        isl_new.append(i_new); isl_cached.append(i_cached)

    ttft = np.array(ttft); isl = np.array(isl)
    isl_new = np.array(isl_new); isl_cached = np.array(isl_cached)
    print(f"plotting {len(ttft)} turns (after filtering)")

    coef, pred, r2 = fit_tier3(ttft, isl_new, isl_cached, isl)
    b0, gamma, alpha, delta = coef
    # Crossover ISL: where δ·isl_new·isl overtakes α·isl_new  →  isl = α / δ
    crossover = alpha / delta if delta > 0 else float("nan")

    # Scaled units in the equation for readability (ms vs tokens vs token²).
    eq = (
        f"TTFT ≈ {b0:.1f} "
        f"+ {gamma*1e3:.3f}·(isl_cached/1k) "
        f"+ {alpha*1e3:.3f}·(isl_new/1k) "
        f"+ {delta*1e6:.4f}·(isl_new·isl/1M)   [ms]"
    )
    print(eq)
    print(f"R² = {r2:.3f}    crossover ISL (FFN→attn dominant) ≈ {crossover:,.0f} tokens")

    cache_hit_rate = isl_cached / isl

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(
        f"Per-turn TTFT vs prefill workload{suffix}\n"
        f"{eq}\n"
        f"R²={r2:.3f}   crossover ISL ≈ {crossover:,.0f} tokens "
        f"(above this, attention term > FFN term)",
        fontsize=10,
    )

    panels = [
        (axes[0], isl_new,            "isl_new (tokens)",
         "TTFT vs isl_new   (FFN-only view — color shows what it misses)"),
        (axes[1], isl_new * isl,      "isl_new · isl (token²)",
         "TTFT vs isl_new·isl   (attention-work view)"),
        (axes[2], cache_hit_rate,     "cache_hit_rate (isl_cached / isl)",
         "TTFT vs cache_hit_rate   (cache-effect view)"),
    ]
    sc = None
    for ax, x, xlabel, title in panels:
        sc = ax.scatter(x, ttft, c=isl, cmap="viridis",
                        alpha=0.35, s=10, edgecolors="none")
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel("ttft_ms")
        ax.grid(True, ls="--", alpha=0.3)
    axes[2].set_xlim(0, 1)

    # Inset histogram of cache_hit_rate on the third panel, top edge.
    p50 = float(np.median(cache_hit_rate))
    p10 = float(np.percentile(cache_hit_rate, 10))
    axes[2].axvline(p50, color="black", ls="--", lw=1.2,
                    label=f"median = {p50:.2f}")
    axes[2].axvline(p10, color="black", ls=":", lw=1.0,
                    label=f"p10 = {p10:.2f}")
    axes[2].legend(loc="upper left", fontsize=8, framealpha=0.95)

    cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("isl (total context tokens)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
