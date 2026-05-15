#!/usr/bin/env python3
"""
Plot every snapshot of one concurrency level (e.g. C=10) as its own subplot.

Layout: rows = runs (sorted by timestamp), columns = setups (mtier left, cpu
right).  Each cell shows the same three lines the Grafana dashboard does:

    HBM hit %      = rate(prefix_cache_hits[2m])      / rate(prefix_cache_queries[2m])           * 100
    Offload hit %  = rate(external_cache_hits[2m])    / rate(external_cache_queries[2m])         * 100   ← LOCAL denom
    HBM KV util %  = vllm:kv_cache_usage_perc * 100                                                     ← dashed

The 2-minute rate window makes the curves smooth (same as Grafana).

Usage:
    python3 analyze_level_aggregate.py --level 10
"""
import argparse
import atexit
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
import pandas as pd
import requests

ROOT_DEFAULT = Path(__file__).resolve().parents[1] / "benchmarks" / "results_benchmarks"

# NVIDIA-green HBM hit, per-setup offload colour, red recompute, light gray KV util.
_NVIDIA_GREEN = "#76B900"
_GRAY         = "#b0b0b0"
_RED          = "#d62728"
_COLORS = {
    "hybrid-mtier": {
        "hbm":       _NVIDIA_GREEN,
        "offload":   "#5e3c99",   # purple
        "recompute": _RED,
        "kv":        _GRAY,
    },
    "hybrid-cpu": {
        "hbm":       _NVIDIA_GREEN,
        "offload":   "#e66101",   # orange
        "recompute": _RED,
        "kv":        _GRAY,
    },
}


# ── throwaway Prometheus over a snapshot ──────────────────────────────────────

def start_local_prom(snapshot_path: Path, port: int) -> subprocess.Popen:
    cfg = Path(tempfile.mkstemp(prefix="prom_agg_", suffix=".yml")[1])
    cfg.write_text("global:\n  scrape_interval: 60s\n")
    log = Path(f"/tmp/prom_agg_{port}.log")
    proc = subprocess.Popen(
        [
            "prometheus",
            f"--config.file={cfg}",
            f"--storage.tsdb.path={snapshot_path}",
            f"--web.listen-address=:{port}",
            "--storage.tsdb.retention.time=10y",
        ],
        stdout=open(log, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    t0 = time.monotonic()
    while True:
        try:
            if requests.get(f"http://localhost:{port}/-/ready", timeout=2).status_code == 200:
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"Prometheus died (see {log})")
        if time.monotonic() - t0 > 30:
            proc.terminate()
            raise RuntimeError(f"Prometheus startup timed out (see {log})")
        time.sleep(0.5)


def stop_local_prom(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass


# ── PromQL query_range → pandas Series ────────────────────────────────────────

def query_range(prom_url: str, query: str, t_start: float, t_end: float, step: str) -> pd.Series:
    r = requests.get(
        f"{prom_url}/api/v1/query_range",
        params={"query": query, "start": t_start, "end": t_end, "step": step},
        timeout=60,
    )
    j = r.json()
    if j.get("status") != "success":
        return pd.Series(dtype=float)
    results = j["data"]["result"]
    if not results:
        return pd.Series(dtype=float)
    series = [
        pd.Series({float(t): float(v) for t, v in res["values"]
                   if v not in ("NaN", "+Inf", "-Inf")}, dtype=float)
        for res in results
    ]
    if len(series) == 1:
        return series[0]
    return pd.concat(series, axis=1).mean(axis=1)


def extract_run_metrics(prom_url: str, t0: float, t1: float, step: str,
                        setup: str, rate_window: str = "2m") -> pd.DataFrame:
    """Return DataFrame with hbm, offload, recompute, kv_util — all in percent.

    Uses the authoritative `vllm:prompt_tokens_by_source_total` counter which
    breaks down every prompt token by where it was served from:
        local_compute        → recompute %
        local_cache_hit      → HBM hit %
        external_kv_transfer → offload hit %
    The three by-source rates sum to the same denominator (total prompt tokens
    processed in the window), so the percentages sum to 100% by construction.

    Defined in vllm/v1/metrics/stats.py: PromptTokenStats.ALL_SOURCES and
    vllm/v1/metrics/loggers.py:601 (counter_prompt_tokens_by_source).
    """
    sel_base = '{setup="%s"}' % setup
    # Use sum() to drop the `source` label so the division matches by setup.
    # Without sum(), the LHS has source="..." and the RHS has source labels too
    # but different — PromQL vector matching returns empty.
    def pct(source: str) -> str:
        return (
            f'sum(rate(vllm:prompt_tokens_by_source_total{{setup="{setup}",source="{source}"}}[{rate_window}]))'
            f' / clamp_min(sum(rate(vllm:prompt_tokens_by_source_total{{setup="{setup}"}}[{rate_window}])), 1e-9)'
            f' * 100'
        )
    hbm       = query_range(prom_url, pct("local_cache_hit"),       t0, t1, step)
    offload   = query_range(prom_url, pct("external_kv_transfer"),  t0, t1, step)
    recompute = query_range(prom_url, pct("local_compute"),         t0, t1, step)
    kv        = query_range(prom_url,
                            f"avg_over_time(vllm:kv_cache_usage_perc{sel_base}[{rate_window}]) * 100",
                            t0, t1, step)

    df = pd.concat({"hbm": hbm, "offload": offload, "recompute": recompute, "kv_util": kv},
                   axis=1).sort_index().ffill().fillna(0.0)
    # Index → seconds offset from level start
    df.index = (df.index - t0).round().astype(int)
    df = df.groupby(df.index).mean()
    return df


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level", type=int, default=10)
    ap.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--step", default="2s")
    ap.add_argument("--rate-window", default="2m",
                    help="PromQL rate() lookback window (default 2m, matches Grafana)")
    ap.add_argument("--port-base", type=int, default=9100)
    args = ap.parse_args()

    level_pattern = f"c{args.level:03d}"
    out_path = args.out or (args.root / f"per_run_{level_pattern}.png")

    level_dirs: list[Path] = sorted(
        p for p in args.root.glob(f"bench_*/{level_pattern}")
        if (p / "config.json").exists() and (p / "prom_snapshot").exists()
    )
    if not level_dirs:
        sys.exit(f"No {level_pattern}/ snapshot dirs found under {args.root}")

    print(f"  Found {len(level_dirs)} run(s) of {level_pattern}:")
    for d in level_dirs:
        print(f"    {d.parent.name}/{d.name}")
    print(f"  Output: {out_path}")

    setups = ["hybrid-mtier", "hybrid-cpu"]
    # rows = runs, cols = setups
    n_rows = len(level_dirs)
    n_cols = len(setups)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(7 * n_cols, 3.2 * n_rows),
                             sharex="col", sharey=True, squeeze=False)

    for i, level_dir in enumerate(level_dirs):
        cfg = json.loads((level_dir / "config.json").read_text())
        t0  = float(cfg["t_start_unix"])
        t1  = float(cfg["t_end_unix"])
        snap = level_dir / "prom_snapshot"
        port = args.port_base + i
        run_label = level_dir.parent.name
        print(f"\n  [{i+1}/{len(level_dirs)}] {run_label}  window={t1 - t0:.0f}s  port={port}")
        prom = start_local_prom(snap, port)
        atexit.register(lambda p=prom: stop_local_prom(p))
        try:
            prom_url = f"http://localhost:{port}"
            for j, setup in enumerate(setups):
                ax = axes[i][j]
                df = extract_run_metrics(prom_url, t0, t1, args.step, setup,
                                         rate_window=args.rate_window)
                if df.empty:
                    ax.text(0.5, 0.5, f"no data for {setup}",
                            ha="center", va="center", transform=ax.transAxes)
                else:
                    colors = _COLORS[setup]
                    ax.plot(df.index, df["hbm"].values,
                            label="HBM hit %", color=colors["hbm"], linewidth=2.0)
                    ax.plot(df.index, df["offload"].values,
                            label="Offload hit %", color=colors["offload"], linewidth=2.0)
                    ax.plot(df.index, df["recompute"].values,
                            label="Recompute %", color=colors["recompute"], linewidth=2.0)
                    ax.plot(df.index, df["kv_util"].values,
                            label="HBM KV util %", color=colors["kv"],
                            linewidth=1.8)
                ax.set_ylim(0, 105)
                ax.grid(alpha=0.3)
                ax.set_title(f"{run_label}   —   {setup}", fontsize=10)
                if j == 0:
                    ax.set_ylabel("Percent (%)")
                if i == 0:
                    ax.legend(loc="center right", fontsize=8)
                if i == n_rows - 1:
                    ax.set_xlabel("Time (s, from level start)")
                print(f"      {setup}: {len(df)} samples")
        finally:
            stop_local_prom(prom)

    fig.suptitle(f"C={args.level}  —  per-run cache & KV util  (rate window: {args.rate_window})",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
