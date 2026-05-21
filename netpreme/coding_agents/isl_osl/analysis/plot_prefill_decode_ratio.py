"""Average e2e latency stacked into (prefill + decode) by turn index.

  x = 1-based turn index within a problem (after dropping empty turns)
  y = avg(ttft_ms) stacked over avg(decode_ms); the top line is avg e2e

Filters: drop empty turns, drop turns missing timing.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import load_data


def per_turn_means(t: np.ndarray, min_samples: int
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each substantive turn index ti, return mean(ttft_ms), mean(decode_ms)
    aggregated across all problems. Per-problem `turn` numbers are re-ranked
    after dropping empty turns so ti=1 is the first substantive turn."""
    ttfts, decs, idx = [], [], []
    for iid in np.unique(t["instance_id"]):
        rows = t[t["instance_id"] == iid]
        ti = 0
        for r in rows:
            if (r["category"] == "empty"
                    or r["isl"] <= 0 or r["osl"] <= 0
                    or r["ttft_ms"] + r["decode_ms"] <= 0):
                continue
            ti += 1
            ttfts.append(float(r["ttft_ms"]))
            decs.append(float(r["decode_ms"]))
            idx.append(ti)
    ttfts = np.array(ttfts); decs = np.array(decs); idx = np.array(idx)

    xs, mean_ttft, mean_dec = [], [], []
    for ti in range(1, int(idx.max()) + 1 if len(idx) else 1):
        m = idx == ti
        if int(m.sum()) < min_samples:
            continue
        xs.append(ti)
        mean_ttft.append(float(ttfts[m].mean()))
        mean_dec.append(float(decs[m].mean()))
    return np.array(xs), np.array(mean_ttft), np.array(mean_dec)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir",      required=True, type=Path)
    ap.add_argument("--out",          required=True, type=Path)
    ap.add_argument("--title-suffix", default="")
    ap.add_argument("--min-samples",  type=int, default=10,
                    help="truncate at turn indices with fewer samples than this")
    args = ap.parse_args()

    t = load_data(args.run_dir)
    xs, mean_ttft, mean_dec = per_turn_means(t, args.min_samples)
    print(f"per-turn axis: turns 1..{xs.max() if len(xs) else 0} "
          f"with ≥{args.min_samples} samples")

    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""
    fig.suptitle(
        f"Average e2e latency = prefill + decode, by turn index{suffix}",
        fontsize=12,
    )
    ax.stackplot(xs, [mean_ttft, mean_dec],
                 labels=["prefill (ttft)", "decode"],
                 colors=["#3b82f6", "#22c55e"], alpha=0.85)
    ax.plot(xs, mean_ttft + mean_dec, color="black", lw=1.2, label="e2e (sum)")
    ax.set_xlabel("turn index within problem")
    ax.set_ylabel("average latency (ms)")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
    ax.grid(True, ls="--", alpha=0.3)
    if len(xs):
        ax.set_xlim(xs.min(), xs.max())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
