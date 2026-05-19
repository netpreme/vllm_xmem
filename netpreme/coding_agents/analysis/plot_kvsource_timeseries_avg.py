#!/usr/bin/env python3
"""
For each concurrency C in --concurrencies (default 12..16), spin Prometheus
over every finalized snapshot, extract time-series of HBM-hit / Offload-hit /
Recompute % (of prompt tokens served by each source) and HBM KV-cache %, and
plot the mean across iters per setup.

Output: one figure per C, two columns (hybrid-mtier left, hybrid-cpu right),
showing four lines (HBM hit %, Offload hit %, Recompute %, HBM KV util %) as a
function of time-from-level-start, with a light shaded ±1σ envelope.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

ROOT_DEFAULT = Path("/tmp/nodeterm_results")

_NVIDIA_GREEN = "#76B900"
_GRAY = "#b0b0b0"
_RED = "#d62728"
_COLORS = {
    "hybrid-mtier": {"hbm": _NVIDIA_GREEN, "offload": "#5e3c99", "recompute": _RED, "kv": _GRAY},
    "hybrid-cpu":   {"hbm": _NVIDIA_GREEN, "offload": "#e66101", "recompute": _RED, "kv": _GRAY},
}


def start_prom(snapshot_path: Path, port: int) -> subprocess.Popen:
    cfg = Path(tempfile.mkstemp(prefix="prom_ts_", suffix=".yml")[1])
    cfg.write_text("global:\n  scrape_interval: 60s\n")
    proc = subprocess.Popen(
        ["prometheus", f"--config.file={cfg}",
         f"--storage.tsdb.path={snapshot_path}",
         f"--web.listen-address=:{port}",
         "--storage.tsdb.retention.time=10y"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    t0 = time.monotonic()
    while True:
        try:
            if requests.get(f"http://localhost:{port}/-/ready", timeout=1).status_code == 200:
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError("prom died")
        if time.monotonic() - t0 > 30:
            proc.terminate()
            raise RuntimeError("prom startup timeout")
        time.sleep(0.3)


def stop_prom(proc: subprocess.Popen) -> None:
    try: os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError: pass
    try: proc.wait(timeout=3)
    except subprocess.TimeoutExpired: pass


def q_range(url: str, query: str, t0: float, t1: float, step: str) -> pd.Series:
    r = requests.get(f"{url}/api/v1/query_range",
                     params={"query": query, "start": t0, "end": t1, "step": step},
                     timeout=30).json()
    if r.get("status") != "success" or not r["data"]["result"]:
        return pd.Series(dtype=float)
    s = pd.Series(
        {float(t): float(v) for t, v in r["data"]["result"][0]["values"] if v not in ("NaN","+Inf","-Inf")},
        dtype=float)
    return s


def extract_setup(url: str, t0: float, t1: float, step: str, setup: str,
                  win: str = "2m") -> pd.DataFrame:
    """Return DataFrame indexed by seconds-from-start with columns:
       hbm, offload, recompute (% of prompt tokens), kv_util (% of HBM KV pool)."""
    sel = f'{{setup="{setup}"}}'
    def by_src(src):
        return (f'sum(rate(vllm:prompt_tokens_by_source_total{{setup="{setup}",source="{src}"}}[{win}]))')
    tot = f'sum(rate(vllm:prompt_tokens_by_source_total{sel}[{win}]))'
    queries = {
        "hbm":       f"{by_src('local_cache_hit')} / clamp_min({tot},1e-9) * 100",
        "offload":   f"{by_src('external_kv_transfer')} / clamp_min({tot},1e-9) * 100",
        "recompute": f"{by_src('local_compute')} / clamp_min({tot},1e-9) * 100",
        "kv_util":   f"avg_over_time(vllm:kv_cache_usage_perc{sel}[{win}]) * 100",
    }
    df = pd.DataFrame({k: q_range(url, q, t0, t1, step) for k, q in queries.items()})
    if df.empty:
        return df
    df = df.sort_index().ffill().fillna(0.0)
    df.index = (df.index - t0).round().astype(int)
    df = df.groupby(df.index).mean()
    return df


def process_concurrency(root: Path, c: int, step: str, port: int) -> dict[str, list[pd.DataFrame]]:
    """For one concurrency: collect per-iter DataFrames per setup."""
    snaps = sorted(p for p in root.glob(f"bench_sweep_*/c{c:03d}")
                   if (p / "config.json").exists() and (p / "prom_snapshot").exists())
    snaps = [s for s in snaps if "t_end_unix" in json.loads((s/"config.json").read_text())]
    print(f"  C={c}: {len(snaps)} finalized snapshots")
    per_setup: dict[str, list[pd.DataFrame]] = {"hybrid-mtier": [], "hybrid-cpu": []}
    for i, snap in enumerate(snaps):
        cfg = json.loads((snap / "config.json").read_text())
        t0 = float(cfg["t_start_unix"]); t1 = float(cfg["t_end_unix"])
        prom = start_prom(snap / "prom_snapshot", port)
        try:
            url = f"http://localhost:{port}"
            for setup in ("hybrid-mtier", "hybrid-cpu"):
                df = extract_setup(url, t0, t1, step, setup)
                if not df.empty:
                    per_setup[setup].append(df)
        finally:
            stop_prom(prom)
        if (i+1) % 5 == 0:
            print(f"    [{i+1}/{len(snaps)}] done", flush=True)
    return per_setup


def aggregate(frames: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stack frames, align by index (truncate to min length), compute mean and std."""
    if not frames: return pd.DataFrame(), pd.DataFrame()
    # Take a common index = intersection of all indices (most conservative)
    common_max = min(f.index.max() for f in frames)
    aligned = [f[f.index <= common_max] for f in frames]
    # Now reindex to a uniform grid (every step)
    cols = ("hbm", "offload", "recompute", "kv_util")
    means = pd.DataFrame(index=aligned[0].index, columns=cols, dtype=float)
    stds  = pd.DataFrame(index=aligned[0].index, columns=cols, dtype=float)
    for col in cols:
        # Build a wide matrix [time, iter]
        wide = pd.concat([f[col] for f in aligned], axis=1)
        means[col] = wide.mean(axis=1)
        stds[col]  = wide.std(axis=1)
    return means, stds


def plot_one_c(c: int, per_setup: dict[str, list[pd.DataFrame]], out_path: Path) -> None:
    setups = ("hybrid-mtier", "hybrid-cpu")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True, squeeze=False)
    for col, setup in enumerate(setups):
        ax = axes[0][col]
        frames = per_setup.get(setup, [])
        if not frames:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue
        means, stds = aggregate(frames)
        x = means.index
        cs = _COLORS[setup]
        for key, label, color, lw in (
            ("hbm",       "HBM hit %",     cs["hbm"],     2.0),
            ("offload",   "Offload hit %", cs["offload"], 2.0),
            ("recompute", "Recompute %",   cs["recompute"], 1.6),
            ("kv_util",   "HBM KV util %", cs["kv"],      1.6),
        ):
            mean = means[key].values
            std  = stds[key].values
            ax.plot(x, mean, color=color, linewidth=lw, label=label)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        ax.set_title(f"C = {c}   —   {setup}   ({len(frames)} iters)", fontsize=11)
        ax.set_xlabel("Time from level start (s)")
        if col == 0:
            ax.set_ylabel("Percent (%)")
        ax.set_ylim(0, 105)
        ax.grid(alpha=0.3)
        ax.legend(loc="center right", fontsize=9)
    fig.suptitle(f"KV source mix over time — pooled across iters  (shaded = ±1σ across iters)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    ap.add_argument("--concurrencies", type=int, nargs="+", default=[12,13,14,15,16])
    ap.add_argument("--step", default="5s")
    ap.add_argument("--port-base", type=int, default=10000)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="output directory (default: --root)")
    args = ap.parse_args()

    out_dir = args.out_dir or args.root
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, c in enumerate(args.concurrencies):
        per_setup = process_concurrency(args.root, c, args.step, args.port_base + i)
        out_path = out_dir / f"kvsource_timeseries_c{c:03d}_nodeterm.png"
        plot_one_c(c, per_setup, out_path)


if __name__ == "__main__":
    main()
