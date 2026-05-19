#!/usr/bin/env python3
"""
Analyze one concurrency-level snapshot produced by bench_concurrent_users.py.

Input:  a level directory containing  config.json  and  prom_snapshot/
Output: two figures inside an analysis/ subdir.

Figure 1 — fig1_timeseries.png
    Stacked area: HBM hit fraction, offload (CPU/MTier) hit fraction,
    recompute fraction (sums to 1).
    Overlay: HBM KV-cache occupancy (vllm:kv_cache_usage_perc).
    Second panel: GPU SM utilization (avg across visible GPUs).

Figure 2 — fig2_offload_dominant.png
    Filter to time windows where offload-share-of-cache-hits > 50%
    (i.e. offload tier served more bytes/tokens than HBM among prefix hits).
    Plot: TTFT p50/p95, E2E p50/p95, output tok/s, overall cache hit rate
    versus time, with the offload-dominant region shaded.

Usage:
    python3 analyze_snapshot.py path/to/c016/
    python3 analyze_snapshot.py path/to/c016/ --port 9099 --step 1s

Requires: prometheus binary, pandas, matplotlib, requests.
"""
import argparse
import atexit
import json
import os
import shutil
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


# ── throwaway Prometheus over the snapshot ────────────────────────────────────

def start_local_prom(snapshot_path: Path, port: int) -> subprocess.Popen:
    """Launch `prometheus` reading the snapshot TSDB on a free port."""
    cfg = Path(tempfile.mkstemp(prefix="prom_analyze_", suffix=".yml")[1])
    cfg.write_text("global:\n  scrape_interval: 60s\n")
    log = Path(f"/tmp/prom_analyze_{port}.log")
    print(f"  [prom] Starting on :{port}  tsdb={snapshot_path}  log={log}", flush=True)
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
            r = requests.get(f"http://localhost:{port}/-/ready", timeout=2)
            if r.status_code == 200:
                print(f"  [prom] Ready ({time.monotonic()-t0:.0f}s)  PID={proc.pid}",
                      flush=True)
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"Prometheus died during startup (see {log})")
        if time.monotonic() - t0 > 30:
            proc.terminate()
            raise RuntimeError(f"Prometheus startup timed out (see {log})")
        time.sleep(0.5)


def stop_local_prom(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass


# ── query_range helpers ───────────────────────────────────────────────────────

def query_range(prom_url: str, query: str, t_start: float, t_end: float,
                step: str) -> pd.Series:
    """Return a pandas Series indexed by Unix timestamp (float seconds)."""
    r = requests.get(
        f"{prom_url}/api/v1/query_range",
        params={"query": query, "start": t_start, "end": t_end, "step": step},
        timeout=60,
    )
    r.raise_for_status()
    j = r.json()
    if j.get("status") != "success":
        raise RuntimeError(f"PromQL error for {query!r}: {j}")
    results = j["data"]["result"]
    if not results:
        return pd.Series(dtype=float)
    # If multiple series come back (e.g. per-GPU), average them
    series_list = []
    for res in results:
        vals = res["values"]
        s = pd.Series(
            {float(t): float(v) for t, v in vals if v not in ("NaN", "+Inf", "-Inf")},
            dtype=float,
        )
        series_list.append(s)
    if len(series_list) == 1:
        return series_list[0]
    return pd.concat(series_list, axis=1).mean(axis=1)


def query_avg_by_gpu(prom_url: str, query: str, t_start: float, t_end: float,
                     step: str) -> dict[str, pd.Series]:
    """Per-GPU series. Used to plot one line per GPU when there are few."""
    r = requests.get(
        f"{prom_url}/api/v1/query_range",
        params={"query": query, "start": t_start, "end": t_end, "step": step},
        timeout=60,
    )
    r.raise_for_status()
    j = r.json()
    out: dict[str, pd.Series] = {}
    for res in j.get("data", {}).get("result", []):
        label = res["metric"].get("gpu", "?")
        s = pd.Series(
            {float(t): float(v) for t, v in res["values"]
             if v not in ("NaN", "+Inf", "-Inf")},
            dtype=float,
        )
        out[label] = s
    return out


# ── plotting ──────────────────────────────────────────────────────────────────

def _discover_setups(prom_url: str) -> list[str]:
    """Return all `setup` label values present in the snapshot."""
    try:
        r = requests.get(f"{prom_url}/api/v1/label/setup/values", timeout=10)
        vals = r.json().get("data", [])
        return [v for v in vals if v]
    except Exception:
        return []


# Per-setup colours match the Grafana dashboard:
#   HBM hit = NVIDIA green   Offload = purple (mtier) / orange (cpu)
#   Recompute = red          KV util = light gray (solid)
_NVIDIA_GREEN = "#76B900"
_GRAY         = "#b0b0b0"
_RED          = "#d62728"
_SETUP_COLORS = {
    "hybrid-mtier": {"hbm": _NVIDIA_GREEN, "offload": "#5e3c99", "recompute": _RED, "kv": _GRAY},
    "hybrid-cpu":   {"hbm": _NVIDIA_GREEN, "offload": "#e66101", "recompute": _RED, "kv": _GRAY},
}


def _hit_rate_series(prom_url: str, t0: float, t1: float, step: str,
                     setup_filter: str, rate_window: str = "2m") -> pd.DataFrame:
    """Per-setup HBM hit % / Offload hit % / Recompute % from the authoritative
    `vllm:prompt_tokens_by_source_total` counter. Each source rate is divided by
    the total of all three (sum-aggregated to drop the source label), so the
    three percentages sum to exactly 100%. Smoothed over `rate_window`."""
    def pct(source: str) -> str:
        return (
            f'sum(rate(vllm:prompt_tokens_by_source_total'
            f'{{setup="{setup_filter}",source="{source}"}}[{rate_window}])) / '
            f'clamp_min(sum(rate(vllm:prompt_tokens_by_source_total'
            f'{{setup="{setup_filter}"}}[{rate_window}])), 1e-9) * 100'
        )
    hbm       = query_range(prom_url, pct("local_cache_hit"),      t0, t1, step)
    offload   = query_range(prom_url, pct("external_kv_transfer"), t0, t1, step)
    recompute = query_range(prom_url, pct("local_compute"),        t0, t1, step)
    df = pd.concat({"hbm": hbm, "offload": offload, "recompute": recompute},
                   axis=1).sort_index().ffill().fillna(0.0)
    return df  # already in percent, sums to ~100% by construction


def fig1_timeseries(prom_url: str, cfg: dict, t0: float, t1: float, step: str,
                    out_path: Path, rate_window: str = "30s") -> None:
    """Per-setup HBM hit % / Offload hit % / Recompute % using global denominator
    (offload-hit = external_hits / total_queries). Hit rate is computed from raw
    counter deltas so each timestep reflects only the traffic in that interval.

    Each setup is drawn on its OWN subplot (stacked vertically) so the lines
    don't overlap."""
    setups = sorted(_discover_setups(prom_url) or [cfg.get("setup", "hybrid-mtier")])
    n = len(setups)

    fig, axes = plt.subplots(n, 1, figsize=(12, 4 * n), sharex=True, squeeze=False)
    axes = axes.flatten()

    for i, setup_label in enumerate(setups):
        ax = axes[i]
        colors = _SETUP_COLORS.get(setup_label, {
            "hbm": "#2ca02c", "offload": "#ff7f0e",
            "recompute": "#d62728", "kv": "#1f77b4",
        })
        df = _hit_rate_series(prom_url, t0, t1, step, setup_label)
        if df.empty:
            ax.text(0.5, 0.5, f"no data for {setup_label}",
                    ha="center", va="center", transform=ax.transAxes)
        else:
            rel = (df.index - t0)
            ax.plot(rel, df["hbm"].values,       label="HBM hit",     color=colors["hbm"],       linewidth=1.8)
            ax.plot(rel, df["offload"].values,   label="Offload hit", color=colors["offload"],   linewidth=1.8)
            ax.plot(rel, df["recompute"].values, label="Recompute",   color=colors["recompute"], linewidth=1.8)
            # HBM KV-cache occupancy — wrap in avg_over_time(...[rate_window]) so
            # it gets the same smoothing window as the rate()-based hit-rate
            # lines. Without this it's a raw 1s gauge → jagged.
            kv = query_range(
                prom_url,
                f'avg_over_time(vllm:kv_cache_usage_perc{{setup="{setup_label}"}}[{rate_window}]) * 100',
                t0, t1, step,
            )
            if not kv.empty:
                ax.plot((kv.index - t0), kv.values,
                        label="HBM KV util", color=colors["kv"],
                        linewidth=1.8)
        ax.set_ylabel("Percent (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"{setup_label}  —  C={cfg.get('concurrency')}")
        ax.legend(loc="center right", fontsize=9)
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (s, from level start)")
    fig.suptitle(f"C={cfg.get('concurrency')}  "
                 f"({cfg.get('sustained_mins')}min, model={cfg.get('model')})",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  [fig1] {out_path}", flush=True)


def fig2_offload_dominant(prom_url: str, cfg: dict, t0: float, t1: float,
                          step: str, out_path: Path,
                          rate_window: str = "30s",
                          threshold: float = 0.5,
                          setup_filter: str | None = None) -> None:
    """Time series of TTFT/E2E/throughput/hit-rate, with offload-dominant region shaded.
    If setup_filter is provided, all queries are filtered to that setup label."""
    sel = ('{setup="%s"}' % setup_filter) if setup_filter else ""
    # Offload share of cache hits = offload / (hbm + offload)
    off_share = query_range(
        prom_url,
        f"rate(vllm:external_prefix_cache_hits_total{sel}[{rate_window}])"
        f" / clamp_min(rate(vllm:prefix_cache_hits_total{sel}[{rate_window}])"
        f" + rate(vllm:external_prefix_cache_hits_total{sel}[{rate_window}]), 1e-9)",
        t0, t1, step,
    )
    ttft_p50 = query_range(
        prom_url,
        f"histogram_quantile(0.5, sum by (le) "
        f"(rate(vllm:time_to_first_token_seconds_bucket{sel}[{rate_window}]))) * 1000",
        t0, t1, step,
    )
    ttft_p95 = query_range(
        prom_url,
        f"histogram_quantile(0.95, sum by (le) "
        f"(rate(vllm:time_to_first_token_seconds_bucket{sel}[{rate_window}]))) * 1000",
        t0, t1, step,
    )
    e2e_p50 = query_range(
        prom_url,
        f"histogram_quantile(0.5, sum by (le) "
        f"(rate(vllm:e2e_request_latency_seconds_bucket{sel}[{rate_window}]))) * 1000",
        t0, t1, step,
    )
    e2e_p95 = query_range(
        prom_url,
        f"histogram_quantile(0.95, sum by (le) "
        f"(rate(vllm:e2e_request_latency_seconds_bucket{sel}[{rate_window}]))) * 1000",
        t0, t1, step,
    )
    out_tps = query_range(
        prom_url,
        f"sum(rate(vllm:generation_tokens_total{sel}[{rate_window}]))",
        t0, t1, step,
    )
    # "Total cache hit rate" = fraction of prompt tokens that came from EITHER
    # HBM or the offload tier (i.e. not recomputed). Computed from the
    # authoritative `vllm:prompt_tokens_by_source_total` counter so it matches
    # the per-run aggregate plot. Sum() drops the `source` label so the
    # division by total works.
    _setup_filter = setup_filter or '*'
    if setup_filter:
        _ps_sel = f'{{setup="{setup_filter}"}}'
        _src    = lambda s: f'{{setup="{setup_filter}",source="{s}"}}'
    else:
        _ps_sel = '{}'
        _src    = lambda s: f'{{source="{s}"}}'
    hit_total = query_range(
        prom_url,
        f'(sum(rate(vllm:prompt_tokens_by_source_total{_src("local_cache_hit")}[{rate_window}]))'
        f' + sum(rate(vllm:prompt_tokens_by_source_total{_src("external_kv_transfer")}[{rate_window}])))'
        f' / clamp_min(sum(rate(vllm:prompt_tokens_by_source_total{_ps_sel}[{rate_window}])), 1e-9)',
        t0, t1, step,
    )

    df = pd.concat({
        "off_share": off_share,
        "ttft_p50":  ttft_p50,
        "ttft_p95":  ttft_p95,
        "e2e_p50":   e2e_p50,
        "e2e_p95":   e2e_p95,
        "out_tps":   out_tps,
        "hit_total": hit_total,
    }, axis=1)
    # Forward-fill very small gaps so masks/lines don't blink
    df = df.sort_index().ffill(limit=2)

    mask = df["off_share"] > threshold

    rel = lambda idx: (idx - t0)  # noqa: E731

    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)

    # Helper to shade offload-dominant intervals on each axis
    def _shade(ax):
        if not mask.any():
            return
        m = mask.astype(int).values
        t = rel(df.index).values
        # Find contiguous True runs
        starts, ends = [], []
        in_run = False
        run_start = 0.0
        for i, val in enumerate(m):
            if val and not in_run:
                in_run = True
                run_start = t[i]
            elif not val and in_run:
                in_run = False
                ends.append(t[i])
                starts.append(run_start)
        if in_run:
            ends.append(t[-1])
            starts.append(run_start)
        for s, e in zip(starts, ends):
            ax.axvspan(s, e, color="orange", alpha=0.15)

    # Panel 1: offload share + threshold line
    ax = axes[0]
    ax.plot(rel(df.index), df["off_share"].values, color="#ff7f0e",
            linewidth=1.3, label="Offload share of cache hits")
    ax.axhline(threshold, color="black", linestyle="--", linewidth=0.8,
               label=f"threshold = {threshold:.0%}")
    ax.plot(rel(df.index), df["hit_total"].values, color="#2ca02c",
            linewidth=1.0, alpha=0.7, label="Total cache hit rate")
    _shade(ax)
    ax.set_ylabel("Fraction")
    ax.set_ylim(0, 1.05)
    _title_setup = setup_filter or cfg.get('setup', '?')
    ax.set_title(f"{_title_setup} — C={cfg.get('concurrency')}  "
                 f"offload-share-of-hits  +  total cache hit rate")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # Panel 2: TTFT
    ax = axes[1]
    ax.plot(rel(df.index), df["ttft_p50"].values, label="TTFT p50", linewidth=1.2)
    ax.plot(rel(df.index), df["ttft_p95"].values, label="TTFT p95", linewidth=1.0, alpha=0.8)
    _shade(ax)
    ax.set_ylabel("TTFT (ms)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # Panel 3: E2E
    ax = axes[2]
    ax.plot(rel(df.index), df["e2e_p50"].values, label="E2E p50", linewidth=1.2)
    ax.plot(rel(df.index), df["e2e_p95"].values, label="E2E p95", linewidth=1.0, alpha=0.8)
    _shade(ax)
    ax.set_ylabel("E2E latency (ms)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    # Panel 4: output throughput
    ax = axes[3]
    ax.plot(rel(df.index), df["out_tps"].values, color="#1f77b4",
            linewidth=1.2, label="Output tok/s")
    _shade(ax)
    ax.set_ylabel("Output tok/s")
    ax.set_xlabel("Time (s, from level start)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  [fig2] {out_path}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("level_dir", type=Path,
                    help="Directory containing config.json + prom_snapshot/")
    ap.add_argument("--port", type=int, default=9099)
    ap.add_argument("--step", default="1s",
                    help="query_range step (default: 1s; bigger = faster, coarser)")
    ap.add_argument("--rate-window", default="30s",
                    help="PromQL rate() lookback window (default: 30s)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Offload-share threshold for Figure 2 (default: 0.5)")
    args = ap.parse_args()

    level_dir = args.level_dir.resolve()
    cfg_path  = level_dir / "config.json"
    snap_path = level_dir / "prom_snapshot"
    if not cfg_path.exists():
        sys.exit(f"ERROR: {cfg_path} not found")
    if not snap_path.exists():
        sys.exit(f"ERROR: {snap_path} not found")

    cfg = json.loads(cfg_path.read_text())
    t0  = float(cfg["t_start_unix"])
    t1  = float(cfg["t_end_unix"])
    print(f"  Snapshot : {snap_path}")
    print(f"  Window   : {t0:.0f} → {t1:.0f}  ({t1-t0:.0f}s)")
    print(f"  Setup    : {cfg.get('setup')}  C={cfg.get('concurrency')}")

    out_dir = level_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    prom = start_local_prom(snap_path, args.port)
    atexit.register(lambda p=prom: stop_local_prom(p))

    try:
        prom_url = f"http://localhost:{args.port}"

        fig1_timeseries(prom_url, cfg, t0, t1, args.step,
                        out_dir / "timeseries.png",
                        rate_window=args.rate_window)

        # One offload-dominant plot per detected setup
        setups = _discover_setups(prom_url) or [cfg.get("setup")]
        for s in sorted(filter(None, setups)):
            suffix = s.replace("hybrid-", "")  # "mtier" or "cpu"
            fig2_offload_dominant(prom_url, cfg, t0, t1, args.step,
                                  out_dir / f"offload_dominant_{suffix}.png",
                                  rate_window=args.rate_window,
                                  threshold=args.threshold,
                                  setup_filter=s)
    finally:
        stop_local_prom(prom)


if __name__ == "__main__":
    main()
