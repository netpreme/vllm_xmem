#!/usr/bin/env python3
"""
Per-turn decode-latency views vs context size (ISL).

Left  panel: ITL  vs ISL — per-token decode latency. Color = response category
             (text_only / tool_only / mixed); marker size ∝ log(osl); linear
             fit on log-x overlaid.
Right panel: decode_ms vs ISL — total per-turn decode wall time. Color = osl.
             3-parameter OLS fit decode_ms ≈ β·osl + γ·osl·isl + δ·osl² is
             reported in the suptitle.

Filters applied:
  - category != "empty"
  - finite ttft / decode / isl / osl
  - itl-panel only: osl >= 5  (per-token mean is too noisy below that)

Usage:
  python3 plot_itl_vs_isl.py \
      --run-dir runs/20260513_175826 \
      --out analysis/analysis_itl_vs_isl.png \
      --title-suffix "claude × Verified"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import load_all_rows, num

CATEGORY_COLORS = {
    "text_only": "#3b82f6",
    "tool_only": "#22c55e",
    "mixed":     "#ec4899",
}


def itl_panel(ax, xs, ys, cats, osls) -> None:
    s = np.clip(8 + np.log1p(osls) * 6, 8, 80)
    for cat in ["tool_only", "mixed", "text_only"]:
        mask = cats == cat
        if not mask.any():
            continue
        ax.scatter(xs[mask], ys[mask], s=s[mask], alpha=0.32,
                   color=CATEGORY_COLORS[cat], edgecolors="white", linewidths=0.2,
                   label=f"{cat} (n={int(mask.sum())})")
    keep = (xs > 0) & np.isfinite(ys)
    if keep.sum() >= 10:
        lx = np.log10(xs[keep])
        m, b = np.polyfit(lx, ys[keep], 1)
        xx = np.logspace(np.log10(max(xs[keep].min(), 1)),
                         np.log10(xs[keep].max()), 200)
        ax.plot(xx, m * np.log10(xx) + b, color="black", lw=1.6,
                label=f"fit: itl ≈ {m:.2f}·log10(x) + {b:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("isl (tokens, log scale)")
    ax.set_ylabel("itl_ms  (per-token decode latency)")
    ax.set_title("ITL vs ISL (starting KV size)", fontsize=11, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.annotate("marker size ∝ log(osl)",
                xy=(0.98, 0.02), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=8, color="#6b7280")


def fit_decode(decode_ms, osl, isl):
    """OLS: decode_ms = β·osl + γ·osl·isl + δ·osl². Returns (coef, R²)."""
    X = np.column_stack([osl, osl * isl, osl * osl])
    coef, *_ = np.linalg.lstsq(X, decode_ms, rcond=None)
    pred = X @ coef
    ss_res = np.sum((decode_ms - pred) ** 2)
    ss_tot = np.sum((decode_ms - decode_ms.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coef, r2


def decode_panel(ax, isl, decode_ms, osl) -> tuple:
    coef, r2 = fit_decode(decode_ms, osl, isl)
    beta, gamma, delta = coef
    crossover = beta / gamma if gamma > 0 else float("nan")
    sc = ax.scatter(isl, decode_ms, c=osl, cmap="viridis",
                    alpha=0.35, s=10, edgecolors="none")
    ax.set_xlabel("isl (tokens) — starting KV size")
    ax.set_ylabel("decode_ms  (per-turn decode wall time)")
    ax.set_title("Total decode_ms vs ISL", fontsize=11, fontweight="bold")
    ax.grid(True, ls="--", alpha=0.3)
    return sc, (beta, gamma, delta, r2, crossover)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title-suffix", default="")
    ap.add_argument("--min-osl", type=int, default=5)
    args = ap.parse_args()

    rows = load_all_rows(args.run_dir)

    # ITL panel needs: itl_ms, osl >= min, isl, osl, category.
    # Decode panel needs: decode_ms > 0, osl > 0, isl > 0.
    # Build both filtered sets from the same row stream.
    itl_isl, itl_osl, itl_val, itl_cat = [], [], [], []
    dec_isl, dec_osl, dec_val = [], [], []
    for r in rows:
        if r.get("category") == "empty":
            continue
        isl = num(r, "isl"); osl = num(r, "osl")
        itl = num(r, "itl_ms"); decode = num(r, "decode_ms")
        if not (np.isfinite(isl) and np.isfinite(osl)) or isl <= 0 or osl <= 0:
            continue
        if np.isfinite(decode) and decode > 0:
            dec_isl.append(isl); dec_osl.append(osl); dec_val.append(decode)
        if np.isfinite(itl) and itl > 0 and osl >= args.min_osl:
            itl_isl.append(isl); itl_osl.append(osl); itl_val.append(itl)
            itl_cat.append(r.get("category", "?"))
    itl_isl = np.array(itl_isl); itl_osl = np.array(itl_osl)
    itl_val = np.array(itl_val); itl_cat = np.array(itl_cat)
    dec_isl = np.array(dec_isl); dec_osl = np.array(dec_osl)
    dec_val = np.array(dec_val)
    print(f"itl panel:    {len(itl_val)} turns")
    print(f"decode panel: {len(dec_val)} turns")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""

    itl_panel(axes[0], itl_isl, itl_val, itl_cat, itl_osl)
    sc, (beta, gamma, delta, r2, crossover) = decode_panel(
        axes[1], dec_isl, dec_val, dec_osl)
    eq = (
        f"decode_ms ≈ {beta:.3f}·osl + {gamma*1e3:.4f}·osl·(isl/1k) "
        f"+ {delta*1e3:.4f}·(osl²/1k)   [ms]"
    )
    print(eq)
    print(f"R² = {r2:.3f}    base ITL (isl→0) ≈ {beta:.2f} ms/token   "
          f"crossover ISL ≈ {crossover:,.0f} tokens")

    fig.suptitle(
        f"Per-turn decode latency vs ISL{suffix}\n"
        f"right panel — {eq}   R²={r2:.3f}   "
        f"base ITL ≈ {beta:.2f} ms/tok   crossover ISL ≈ {crossover:,.0f} tok",
        fontsize=10,
    )
    cbar = fig.colorbar(sc, ax=axes[1], fraction=0.04, pad=0.02)
    cbar.set_label("osl (output tokens)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
