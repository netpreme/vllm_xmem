"""Per-turn decode latency vs context size.

Left  : ITL_ms vs ISL — per-token decode latency, colored by response type.
Right : decode_ms vs ISL — total per-turn decode wall time, with the OLS fit
        `decode_ms ≈ β·osl + γ·osl·isl + δ·osl²` reported in the title.

Filters: drop empty turns, drop non-positive isl/osl. ITL panel additionally
requires osl >= 5 (per-token mean is too noisy below that).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import load_data

CATEGORY_COLORS = {
    "text_only": "#3b82f6",
    "tool_only": "#22c55e",
    "mixed":     "#ec4899",
}


def fit_decode(decode_ms: np.ndarray, osl: np.ndarray, isl: np.ndarray
               ) -> tuple[np.ndarray, float]:
    """OLS: decode_ms = β·osl + γ·osl·isl + δ·osl². Returns (coef, R²)."""
    X = np.column_stack([osl, osl * isl, osl * osl])
    coef, *_ = np.linalg.lstsq(X, decode_ms, rcond=None)
    pred = X @ coef
    ss_res = np.sum((decode_ms - pred) ** 2)
    ss_tot = np.sum((decode_ms - decode_ms.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coef, r2


def itl_panel(ax, isl, itl, cat, osl) -> None:
    sizes = np.clip(8 + np.log1p(osl) * 6, 8, 80)
    for category in ("tool_only", "mixed", "text_only"):
        mask = cat == category
        if not mask.any():
            continue
        ax.scatter(isl[mask], itl[mask], s=sizes[mask], alpha=0.32,
                   color=CATEGORY_COLORS[category],
                   edgecolors="white", linewidths=0.2,
                   label=f"{category} (n={int(mask.sum())})")

    # Linear fit on log(isl).
    keep = (isl > 0) & np.isfinite(itl)
    if keep.sum() >= 10:
        m, b = np.polyfit(np.log10(isl[keep]), itl[keep], 1)
        xx = np.logspace(np.log10(isl[keep].min()), np.log10(isl[keep].max()), 200)
        ax.plot(xx, m * np.log10(xx) + b, color="black", lw=1.6,
                label=f"fit: itl ≈ {m:.2f}·log10(isl) + {b:.2f}")

    ax.set_xscale("log")
    ax.set_xlabel("isl (tokens, log scale)")
    ax.set_ylabel("itl_ms  (per-token decode latency)")
    ax.set_title("ITL vs ISL (starting KV size)", fontsize=11, fontweight="bold")
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.95)
    ax.annotate("marker size ∝ log(osl)", xy=(0.98, 0.02),
                xycoords="axes fraction", ha="right", va="bottom",
                fontsize=8, color="#6b7280")


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
    return sc, beta, gamma, delta, r2, crossover


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",      required=True, type=Path)
    ap.add_argument("--out",          required=True, type=Path)
    ap.add_argument("--title-suffix", default="")
    ap.add_argument("--min-osl",      type=int, default=5,
                    help="minimum osl for ITL panel (per-token mean noise floor)")
    args = ap.parse_args()

    t = load_data(args.run_dir)
    real = t[(t["osl"] > 0) & (t["isl"] > 0) & (t["osl"] > 0)]
    dec = real[real["decode_ms"] > 0]
    itl = real[(real["itl_ms"] > 0) & (real["osl"] >= args.min_osl)
               & np.isfinite(real["itl_ms"])]
    print(f"itl panel:    {len(itl)} turns")
    print(f"decode panel: {len(dec)} turns")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""

    itl_panel(axes[0],
              itl["isl"].astype(float), itl["itl_ms"].astype(float),
              itl["category"], itl["osl"].astype(float))
    sc, beta, gamma, delta, r2, crossover = decode_panel(
        axes[1],
        dec["isl"].astype(float), dec["decode_ms"].astype(float),
        dec["osl"].astype(float))

    eq = (f"decode_ms ≈ {beta:.3f}·osl + {gamma*1e3:.4f}·osl·(isl/1k) "
          f"+ {delta*1e3:.4f}·(osl²/1k)   [ms]")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
