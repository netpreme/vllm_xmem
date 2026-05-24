"""Per-turn TTFT vs prefill workload, three-panel view + tier-3 fit.

Prefill cost decomposes as:
    TTFT ≈ β₀ + γ·isl_cached + α·isl_new + δ·isl_new·isl
              └ overhead └ cache load └─ FFN ─┘ └── attention ──┘

  Panel 1: TTFT vs isl_new           — color = isl
  Panel 2: TTFT vs cache_hit_rate    — cache effect
  Panel 3: decode_ms vs ttft_ms      — prefill vs decode dominance (log-log)

Filters: drop empty turns, drop ttft<=0 / isl<=0.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import load_data


def fit_tier3(ttft: np.ndarray, isl_new: np.ndarray,
              isl_cached: np.ndarray, isl: np.ndarray
              ) -> tuple[np.ndarray, float]:
    """OLS: TTFT = β₀ + γ·isl_cached + α·isl_new + δ·isl_new·isl. (coef, R²)."""
    X = np.column_stack([np.ones_like(ttft), isl_cached, isl_new, isl_new * isl])
    coef, *_ = np.linalg.lstsq(X, ttft, rcond=None)
    pred = X @ coef
    ss_res = np.sum((ttft - pred) ** 2)
    ss_tot = np.sum((ttft - ttft.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coef, r2


def _scatter_ttft(ax, x, ttft, isl, *, xlabel: str, title: str):
    sc = ax.scatter(x, ttft, c=isl, cmap="viridis",
                    alpha=0.35, s=10, edgecolors="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ttft_ms")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.3)
    return sc


def _decode_vs_ttft(ax, ttft, decode, isl):
    mask = (decode > 0) & np.isfinite(decode)
    ax.scatter(ttft[mask], decode[mask], c=isl[mask], cmap="viridis",
               alpha=0.35, s=10, edgecolors="none")
    lo, hi = (max(1.0, min(ttft[mask].min(), decode[mask].min())),
              max(ttft[mask].max(), decode[mask].max()))
    ax.plot([lo, hi], [lo, hi], color="#ef4444", lw=0.8, alpha=0.7,
            label="ttft = decode")
    ax.set(xscale="log", yscale="log",
           xlim=(lo * 0.85, hi * 1.15), ylim=(lo * 0.85, hi * 1.15),
           xlabel="ttft_ms  (prefill time)",
           ylabel="decode_ms  (decode time)")
    ax.set_title("decode_ms vs ttft_ms", fontsize=10, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",      required=True, type=Path)
    ap.add_argument("--out",          required=True, type=Path)
    ap.add_argument("--title-suffix", default="")
    args = ap.parse_args()

    t = load_data(args.run_dir)
    real = t[(t["osl"] > 0) & (t["ttft_ms"] > 0) & (t["isl"] > 0)]
    print(f"plotting {len(real)} turns (after filtering)")

    ttft, isl   = real["ttft_ms"].astype(float),   real["isl"].astype(float)
    isl_new     = real["isl_new"].astype(float)
    isl_cached  = real["isl_cached"].astype(float)
    decode      = real["decode_ms"].astype(float)
    cache_hit   = isl_cached / isl

    coef, r2 = fit_tier3(ttft, isl_new, isl_cached, isl)
    b0, gamma, alpha_, delta = coef
    crossover = alpha_ / delta if delta > 0 else float("nan")

    eq = (f"TTFT ≈ {b0:.1f} "
          f"+ {gamma*1e3:.3f}·(isl_cached/1k) "
          f"+ {alpha_*1e3:.3f}·(isl_new/1k) "
          f"+ {delta*1e6:.4f}·(isl_new·isl/1M)   [ms]")
    print(eq)
    print(f"R² = {r2:.3f}    crossover ISL (FFN→attn dominant) ≈ {crossover:,.0f} tokens")

    fig, axes = plt.subplots(1, 3, figsize=(22, 7), constrained_layout=True)
    axes[0].sharey(axes[1])
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(
        f"Per-turn TTFT vs prefill workload{suffix}\n{eq}\n"
        f"R²={r2:.3f}   crossover ISL ≈ {crossover:,.0f} tokens "
        f"(above this, attention term > FFN term)",
        fontsize=10,
    )
    sc = _scatter_ttft(axes[0], isl_new,   ttft, isl,
                       xlabel="isl_new (tokens)",
                       title="TTFT vs isl_new")
    _scatter_ttft(axes[1], cache_hit, ttft, isl,
                  xlabel="cache_hit_rate (isl_cached / isl)",
                  title="TTFT vs cache_hit_rate")
    axes[1].set_xlim(-0.01, 1.01)
    y_hi = ttft.max() * 1.05
    axes[0].set_ylim(-y_hi * 0.01, y_hi)
    _decode_vs_ttft(axes[2], ttft, decode, isl)

    cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("isl (total context tokens)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
