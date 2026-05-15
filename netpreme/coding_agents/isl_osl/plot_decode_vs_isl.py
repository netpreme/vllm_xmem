#!/usr/bin/env python3
"""
Per-turn total decode time vs context size, two-panel view + model fit.

Decode cost decomposes (analogous to prefill):
  decode_ms ≈ β·osl + γ·osl·isl + δ·osl²
              └ FFN/proj per generated token (no isl dependence)
                       └ attend each new token to the cached prefix
                                └ attend each new token to prior new tokens

Left:  decode_ms vs isl              — starting KV size
Right: decode_ms vs (isl + osl/2)    — average KV size during decode

Points colored by osl (number of output tokens). A 3-parameter OLS fit (no
intercept — decoding zero tokens takes zero time) is reported in the suptitle,
along with the per-decoded-token base latency at isl=0 (= β).

Filters:
  - category != "empty"
  - decode_ms, osl, isl all finite and > 0

Usage:
  python3 plot_decode_vs_isl.py \
      --run-dir runs/20260513_175826 \
      --out analysis/analysis_decode_vs_isl.png \
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


def fit_decode(decode_ms, osl, isl):
    """OLS: decode_ms = β·osl + γ·osl·isl + δ·osl². Returns (coef, pred, R²)."""
    X = np.column_stack([osl, osl * isl, osl * osl])
    coef, *_ = np.linalg.lstsq(X, decode_ms, rcond=None)
    pred = X @ coef
    ss_res = np.sum((decode_ms - pred) ** 2)
    ss_tot = np.sum((decode_ms - decode_ms.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return coef, pred, r2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title-suffix", default="")
    args = ap.parse_args()

    rows = load_turns(args.run_dir)

    decode_ms, osl, isl = [], [], []
    for r in rows:
        if r.get("category") == "empty":
            continue
        d = num(r, "decode_ms")
        o = num(r, "osl")
        i = num(r, "isl")
        if not all(np.isfinite(x) for x in (d, o, i)):
            continue
        if d <= 0 or o <= 0 or i <= 0:
            continue
        decode_ms.append(d); osl.append(o); isl.append(i)
    decode_ms = np.array(decode_ms); osl = np.array(osl); isl = np.array(isl)
    print(f"plotting {len(decode_ms)} turns (after filtering)")

    coef, pred, r2 = fit_decode(decode_ms, osl, isl)
    beta, gamma, delta = coef
    # Crossover: where γ·isl overtakes β  →  isl = β / γ
    crossover = beta / gamma if gamma > 0 else float("nan")

    eq = (
        f"decode_ms ≈ {beta:.3f}·osl "
        f"+ {gamma*1e3:.4f}·osl·(isl/1k) "
        f"+ {delta*1e3:.4f}·(osl²/1k)   [ms]"
    )
    print(eq)
    print(f"R² = {r2:.3f}    base ITL (isl→0) ≈ {beta:.2f} ms/token   "
          f"crossover ISL ≈ {crossover:,.0f} tokens")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True,
                             constrained_layout=True)
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(
        f"Per-turn total decode time vs context size{suffix}\n"
        f"{eq}\n"
        f"R²={r2:.3f}   base ITL ≈ {beta:.2f} ms/tok   "
        f"crossover ISL ≈ {crossover:,.0f} tokens",
        fontsize=10,
    )

    panels = [
        (axes[0], isl,            "isl (tokens) — starting KV",
         "decode_ms vs isl"),
        (axes[1], isl + osl / 2,  "isl + osl/2 (tokens) — avg KV during decode",
         "decode_ms vs (isl + osl/2)"),
    ]
    sc = None
    for ax, x, xlabel, title in panels:
        sc = ax.scatter(x, decode_ms, c=osl, cmap="viridis",
                        alpha=0.35, s=10, edgecolors="none")
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_ylabel("decode_ms")
        ax.grid(True, ls="--", alpha=0.3)

    cbar = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("osl (output tokens)")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
