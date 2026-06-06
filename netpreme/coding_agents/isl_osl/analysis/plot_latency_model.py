"""Three-panel latency model for the ISL/OSL simulator (pools 1+ runs).

  1. itl_ms vs isl         — decode per-token cost vs TOTAL context
                             (isl = isl_cached + isl_new, the full KV cache the
                             decode attends over). itl ≈ a + b·isl.
  2. prefill_ms vs isl_new — prefill cost vs RECOMPUTED (uncached) tokens.
                             Floor + per-token; cached tokens are loaded free.
  3. decode/prefill vs turn — per-turn ratio over turn index (faint points +
                             mean-per-turn trend). Shows decode increasingly
                             dominates as the prefix cache fills.

Usage:
    python analysis/plot_latency_model.py --save-dir RUN1 [RUN2 ...] --out fig.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import ScalarFormatter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import load_data  # noqa: E402


def _binned_median(x, y, n=14):
    """Median (x, y) per log-spaced x-bin with >=5 points — the central
    trend, used as the fit target so every decade gets equal weight (a plain
    least-squares fit is swamped by the dense low-isl_new region)."""
    edges = np.logspace(np.log10(max(x.min(), 1)), np.log10(x.max()), n)
    bx, by = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        b = (x >= lo) & (x < hi)
        if b.sum() >= 5:
            bx.append(np.median(x[b]))
            by.append(np.median(y[b]))
    return np.array(bx), np.array(by)


def _fit_floor_power(x, y, floor):
    """Fit y ≈ floor + a·x^p to the binned-median trend, minimizing log-space
    RMS (so low- and high-isl_new fit equally well). The pure linear
    floor+slope·x model is too shallow at low isl_new and too steep at high;
    prefill is concave in recompute, so a power term p<1 fits far better.
    Grid-search p, closed-form least-squares a at each p."""
    bx, by = _binned_median(x, y)
    best = None
    for p in np.linspace(0.40, 1.0, 121):
        xp = bx**p
        a = float(np.sum((by - floor) * xp) / np.sum(xp * xp))
        if a <= 0:
            continue
        err = np.sqrt(np.mean((np.log(floor + a * xp) - np.log(by)) ** 2))
        if best is None or err < best[0]:
            best = (err, a, float(p))
    return best[1], best[2]  # a, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", required=True, nargs="+", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--title-suffix", default="")
    args = ap.parse_args()

    t = np.concatenate([load_data(d) for d in args.save_dir])
    isl = t["isl"].astype(float)
    isl_new = t["isl_new"].astype(float)
    osl = t["osl"].astype(float)
    prefill = t["prefill_ms"].astype(float)
    decode = t["decode_ms"].astype(float)
    itl = t["itl_ms"].astype(float)
    turn = t["turn"].astype(int)

    # The latency model needs vLLM timing fields. Backends without them (e.g.
    # the Anthropic API, which reports only token usage) have all-NaN timings —
    # nothing to fit, so skip this figure rather than crash the analysis pass.
    if not np.isfinite(itl).any() or not np.isfinite(prefill).any():
        print(f"skip latency model: no timing data in {args.save_dir}")
        return 0

    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    title = f"ISL/OSL latency model  (pooled {len(t):,} turns){suffix}"

    # One figure, 2×2: top row = decode + prefill latency models;
    # bottom row = avg prefill/decode per turn + the decode/prefill ratio.
    fig, axg = plt.subplots(2, 2, figsize=(15, 13))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    # ---- 1: itl vs (total isl + uncached isl) -----------------------------
    a = axg[0, 1]
    m = (itl > 0) & np.isfinite(itl) & (osl >= 5) & (isl > 0)
    x1 = isl + isl_new  # total context + the uncached (tool-call-result) tokens
    af, bf = np.polyfit(x1[m], itl[m], 1)[::-1]
    sc = a.scatter(
        x1[m],
        itl[m],
        c=osl[m],
        cmap="turbo",
        norm=LogNorm(vmin=1e1, vmax=1e4),
        s=10,
        alpha=0.5,
    )
    xx = np.linspace(x1[m].min(), x1[m].max(), 100)
    a.plot(xx, af + bf * xx, "k", lw=2, label=f"ITL ≈ {af:.2f} + {bf*1e3:.4f}·x/1k")
    a.set(
        xlabel="ISL + uncached ISL (tool-call results)  [tokens]",
        ylabel="ITL [ms]",
        title="Decode scales linearly with ISL",
    )
    a.ticklabel_format(axis="x", style="sci", scilimits=(3, 3), useMathText=True)
    a.legend(loc="upper left")
    a.grid(alpha=0.3)
    fig.colorbar(sc, ax=a, label="OSL")

    # ---- 2: prefill vs isl_new --------------------------------------------
    a = axg[0, 0]
    m = (prefill > 0) & (isl > 0)
    # floor + a·x^p : floor = the genuine lower bound on prefill (fixed
    # per-call cost at ~zero recompute, the true minimum); the concave power
    # term (p<1) tracks the falling per-token cost as recompute grows.
    floor = float(prefill[m].min())
    a_pow, p_pow = _fit_floor_power(isl_new[m], prefill[m], floor)
    c2 = isl + isl_new  # ISL + ISL uncached (matches panel 1's x)
    sc = a.scatter(
        isl_new[m],
        prefill[m],
        c=c2[m],
        cmap="turbo",
        norm=Normalize(vmin=2e4, vmax=1.5e5),
        s=10,
        alpha=0.45,
    )
    xx = np.logspace(
        np.log10(isl_new[m].clip(1).min()), np.log10(isl_new[m].max()), 100
    )
    a.plot(
        xx,
        floor + a_pow * xx**p_pow,
        "k-",
        lw=2,
        label=f"prefill_ms = {floor:.0f} + {a_pow:.3f}·x^{p_pow:.2f}",
    )
    a.set(
        xscale="log",
        yscale="log",
        xlabel="ISL uncached [tokens]",
        ylabel="Prefill [ms]",
        title="Prefill time dependent on new unseen tokens",
    )
    a.legend(loc="upper left")
    a.grid(alpha=0.3, which="both")
    cb_fmt = ScalarFormatter(useMathText=True)
    cb_fmt.set_powerlimits((3, 3))
    fig.colorbar(sc, ax=a, label="ISL + ISL uncached", format=cb_fmt)

    m = (decode > 0) & (prefill > 0)
    r = decode[m] / prefill[m]
    tr = turn[m]
    pf, dc = prefill[m], decode[m]
    tmax = 40
    tx = np.arange(1, tmax + 1)

    # ---- 3: avg prefill (bottom bar) + avg decode (top bar) per turn ------
    a = axg[1, 0]
    mean_pf = np.array([pf[tr == k].mean() if (tr == k).any() else 0.0 for k in tx])
    mean_dc = np.array([dc[tr == k].mean() if (tr == k).any() else 0.0 for k in tx])
    a.bar(tx, mean_pf, color="#4c78a8", label="avg prefill")
    a.bar(tx, mean_dc, bottom=mean_pf, color="#f58518", label="avg decode")
    a.set(
        xlabel="Turn", ylabel="avg latency [ms]", title="avg prefill + decode per turn"
    )
    a.legend(loc="upper left")
    a.grid(alpha=0.3, axis="y")

    # ---- 4: decode/prefill ratio vs turn index ----------------------------
    a = axg[1, 1]
    mean_r = np.array([r[tr == k].mean() if (tr == k).any() else np.nan for k in tx])
    median_r = np.array(
        [np.median(r[tr == k]) if (tr == k).any() else np.nan for k in tx]
    )
    a.scatter(tr[tr <= tmax], r[tr <= tmax], s=6, alpha=0.15, color="#2ca02c")
    a.plot(tx, mean_r, "-o", color="#117733", lw=2, ms=4, label="mean ratio / turn")
    a.plot(tx, median_r, "-s", color="#cc6677", lw=2, ms=4, label="median ratio / turn")
    a.axhline(
        np.median(r),
        color="k",
        ls="--",
        alpha=0.6,
        label=f"overall median {np.median(r):.1f}×",
    )
    a.axhline(
        r.mean(),
        color="b",
        ls=":",
        lw=1.5,
        alpha=0.7,
        label=f"overall mean {r.mean():.1f}×",
    )
    a.set(
        xlabel="Turn",
        ylabel="Decode / prefill ratio",
        title="Decode dominates with prefix caching (HBM cached)",
    )
    a.set_ylim(0, np.nanpercentile(r[tr <= tmax], 97))
    a.legend(loc="upper left")
    a.grid(alpha=0.3)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out, dpi=120, bbox_inches="tight")
    print(f"wrote {args.out}")
    print(
        f"itl ≈ {af:.3f} + {bf*1e3:.4f}·isl/1k   "
        f"prefill ≈ {floor:.0f} + {a_pow:.3f}·isl_new^{p_pow:.2f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
