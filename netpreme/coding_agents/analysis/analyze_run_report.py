#!/usr/bin/env python3
"""
Detailed per-run report for one snapshot directory.

Produces, per setup (mtier + cpu):
  - Time-series figure with TTFT, decode/ITL, E2E, output tok/s, KV breakdown
  - Stats JSON with mean / p50 / p99 for each metric, for two windows:
        full   : the whole 20-min run
        offload: only timestamps where offload-hit % > 50%

KV breakdown is the absolute tokens/sec from each source
(`vllm:prompt_tokens_by_source_total`), so you can see how many tokens were
served from HBM vs offload vs recompute, not just percentages.

Usage:
    python3 analyze_run_report.py <level_dir>
    python3 analyze_run_report.py results_benchmarks/bench_sweep_<ts>/c016/
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
import numpy as np
import pandas as pd
import requests

_NVIDIA_GREEN = "#76B900"
_GRAY         = "#b0b0b0"
_RED          = "#d62728"
_COLORS = {
    "hybrid-mtier": {"hbm": _NVIDIA_GREEN, "offload": "#5e3c99", "recompute": _RED, "kv": _GRAY},
    "hybrid-cpu":   {"hbm": _NVIDIA_GREEN, "offload": "#e66101", "recompute": _RED, "kv": _GRAY},
}


# ── throwaway Prometheus over a snapshot ──────────────────────────────────────

def start_local_prom(snapshot_path: Path, port: int) -> subprocess.Popen:
    cfg = Path(tempfile.mkstemp(prefix="prom_rpt_", suffix=".yml")[1])
    cfg.write_text("global:\n  scrape_interval: 60s\n")
    log = Path(f"/tmp/prom_rpt_{port}.log")
    proc = subprocess.Popen(
        ["prometheus", f"--config.file={cfg}",
         f"--storage.tsdb.path={snapshot_path}",
         f"--web.listen-address=:{port}",
         "--storage.tsdb.retention.time=10y"],
        stdout=open(log, "w"), stderr=subprocess.STDOUT, start_new_session=True,
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
    r = requests.get(f"{prom_url}/api/v1/query_range",
                     params={"query": query, "start": t_start, "end": t_end, "step": step},
                     timeout=60)
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


# ── extract per-setup time series for one run ─────────────────────────────────

def extract_setup_metrics(prom_url: str, t0: float, t1: float, step: str,
                          setup: str, win: str = "2m") -> pd.DataFrame:
    """Per-setup time series. All columns indexed by seconds-from-start."""
    sel = f'{{setup="{setup}"}}'
    def by_src(s):
        return (
            f'sum(rate(vllm:prompt_tokens_by_source_total{{setup="{setup}",source="{s}"}}[{win}]))'
        )
    rate_tot = f'sum(rate(vllm:prompt_tokens_by_source_total{{setup="{setup}"}}[{win}]))'

    queries = {
        # latency p50 / p99 in milliseconds
        "ttft_p50":   f'histogram_quantile(0.50, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{sel}[{win}]))) * 1000',
        "ttft_p99":   f'histogram_quantile(0.99, sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{sel}[{win}]))) * 1000',
        "decode_p50": f'histogram_quantile(0.50, sum by (le) (rate(vllm:inter_token_latency_seconds_bucket{sel}[{win}]))) * 1000',
        "decode_p99": f'histogram_quantile(0.99, sum by (le) (rate(vllm:inter_token_latency_seconds_bucket{sel}[{win}]))) * 1000',
        "e2e_p50":    f'histogram_quantile(0.50, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket{sel}[{win}]))) * 1000',
        "e2e_p99":    f'histogram_quantile(0.99, sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket{sel}[{win}]))) * 1000',
        # throughput
        "out_tps":    f'sum(rate(vllm:generation_tokens_total{sel}[{win}]))',
        # KV breakdown ABSOLUTE tok/s
        "hbm_tps":     by_src("local_cache_hit"),
        "offload_tps": by_src("external_kv_transfer"),
        "recompute_tps": by_src("local_compute"),
        # KV breakdown percentages (for offload>50% mask)
        "hbm_pct":     f'{by_src("local_cache_hit")} / clamp_min({rate_tot}, 1e-9) * 100',
        "offload_pct": f'{by_src("external_kv_transfer")} / clamp_min({rate_tot}, 1e-9) * 100',
        "recompute_pct": f'{by_src("local_compute")} / clamp_min({rate_tot}, 1e-9) * 100',
        # HBM occupancy
        "kv_util":   f'avg_over_time(vllm:kv_cache_usage_perc{sel}[{win}]) * 100',
    }
    df = pd.DataFrame({k: query_range(prom_url, q, t0, t1, step) for k, q in queries.items()})
    df = df.sort_index().ffill().fillna(0.0)
    df.index = (df.index - t0).round().astype(int)
    df = df.groupby(df.index).mean()
    return df


# ── stats helpers ─────────────────────────────────────────────────────────────

def summarize(df: pd.DataFrame, mask: pd.Series | None = None) -> dict:
    """Compute mean/p50/p99 for each numeric column. Optionally over a mask."""
    sub = df if mask is None else df[mask]
    out = {}
    for col in df.columns:
        s = sub[col].replace(0, np.nan).dropna() if col in ("ttft_p50", "ttft_p99",
                                                              "decode_p50", "decode_p99",
                                                              "e2e_p50", "e2e_p99") else sub[col].dropna()
        if s.empty:
            out[col] = {"n": 0}
        else:
            out[col] = {
                "n":    int(s.size),
                "mean": float(s.mean()),
                "p50":  float(s.quantile(0.50)),
                "p99":  float(s.quantile(0.99)),
                "min":  float(s.min()),
                "max":  float(s.max()),
            }
    return out


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_run_report(per_setup: dict[str, pd.DataFrame], out_path: Path, cfg: dict) -> None:
    """5 rows × 2 cols (one column per setup): TTFT, Decode, E2E, Throughput, KV breakdown."""
    setups = [s for s in ["hybrid-mtier", "hybrid-cpu"] if not per_setup[s].empty]
    n_cols = len(setups)
    if n_cols == 0:
        return
    rows = ["TTFT", "Decode/ITL", "E2E latency", "Output throughput", "KV breakdown (absolute)"]
    fig, axes = plt.subplots(len(rows), n_cols, figsize=(7.5 * n_cols, 3 * len(rows)),
                             sharex=True, squeeze=False)

    for col, setup in enumerate(setups):
        df = per_setup[setup]
        c = _COLORS[setup]
        rel = df.index
        # Shade where offload > 50%
        mask_off = df["offload_pct"] > 50
        def shade(ax):
            if not mask_off.any():
                return
            run_start = None
            for i, v in enumerate(mask_off.values):
                if v and run_start is None:
                    run_start = rel[i]
                elif not v and run_start is not None:
                    ax.axvspan(run_start, rel[i], color=c["offload"], alpha=0.10)
                    run_start = None
            if run_start is not None:
                ax.axvspan(run_start, rel[-1], color=c["offload"], alpha=0.10)

        # Row 0: TTFT
        ax = axes[0][col]
        ax.plot(rel, df["ttft_p50"], label="p50", color=c["offload"], linewidth=1.6)
        ax.plot(rel, df["ttft_p99"], label="p99", color=c["offload"], linewidth=1.2, linestyle="--")
        shade(ax)
        ax.set_ylabel("TTFT (ms)")
        ax.set_title(f"{setup}")
        ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

        # Row 1: Decode / ITL
        ax = axes[1][col]
        ax.plot(rel, df["decode_p50"], label="p50", color=c["offload"], linewidth=1.6)
        ax.plot(rel, df["decode_p99"], label="p99", color=c["offload"], linewidth=1.2, linestyle="--")
        shade(ax)
        ax.set_ylabel("ITL (ms)")
        ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

        # Row 2: E2E
        ax = axes[2][col]
        ax.plot(rel, df["e2e_p50"], label="p50", color=c["offload"], linewidth=1.6)
        ax.plot(rel, df["e2e_p99"], label="p99", color=c["offload"], linewidth=1.2, linestyle="--")
        shade(ax)
        ax.set_ylabel("E2E latency (ms)")
        ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

        # Row 3: Output throughput
        ax = axes[3][col]
        ax.plot(rel, df["out_tps"], color=c["offload"], linewidth=1.8, label="Output tok/s")
        shade(ax)
        ax.set_ylabel("Output tok/s")
        ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

        # Row 4: KV breakdown absolute
        ax = axes[4][col]
        ax.plot(rel, df["hbm_tps"],       label="HBM hit",   color=c["hbm"],       linewidth=1.6)
        ax.plot(rel, df["offload_tps"],   label="Offload",   color=c["offload"],   linewidth=1.6)
        ax.plot(rel, df["recompute_tps"], label="Recompute", color=c["recompute"], linewidth=1.6)
        shade(ax)
        ax.set_ylabel("Source tok/s")
        ax.set_xlabel("Time (s, from level start)")
        ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)

    fig.suptitle(f"Per-run report  —  C={cfg.get('concurrency')}  "
                 f"({cfg.get('sustained_mins')}min)   "
                 f"shaded = offload hit > 50%",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"  [report] {out_path}", flush=True)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("level_dir", type=Path,
                    help="results_benchmarks/bench_<run>/c<NNN>/ directory")
    ap.add_argument("--port",        type=int, default=9099)
    ap.add_argument("--step",        default="2s")
    ap.add_argument("--rate-window", default="2m")
    args = ap.parse_args()

    level_dir = args.level_dir.resolve()
    cfg_path  = level_dir / "config.json"
    snap_path = level_dir / "prom_snapshot"
    if not cfg_path.exists() or not snap_path.exists():
        sys.exit(f"ERROR: missing config.json or prom_snapshot in {level_dir}")

    cfg = json.loads(cfg_path.read_text())
    t0 = float(cfg["t_start_unix"])
    t1 = float(cfg["t_end_unix"])
    print(f"  Snapshot:  {snap_path}")
    print(f"  Window:    {t1 - t0:.0f}s")
    setups_cfg = cfg.get('setups', [])
    setup_names = [s['setup'] if isinstance(s, dict) else s for s in setups_cfg]
    print(f"  C={cfg.get('concurrency')}  setups={setup_names}")

    out_dir = level_dir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    prom = start_local_prom(snap_path, args.port)
    atexit.register(lambda p=prom: stop_local_prom(p))
    try:
        prom_url = f"http://localhost:{args.port}"
        per_setup = {}
        for setup in ("hybrid-mtier", "hybrid-cpu"):
            df = extract_setup_metrics(prom_url, t0, t1, args.step, setup, win=args.rate_window)
            per_setup[setup] = df
            print(f"    {setup}: {len(df)} samples")

        plot_run_report(per_setup, out_dir / "run_report.png", cfg)

        # Stats JSON: full-run + offload-dominant (>50%) window
        stats = {}
        for setup, df in per_setup.items():
            if df.empty:
                stats[setup] = {"note": "no data"}
                continue
            mask = df["offload_pct"] > 50
            stats[setup] = {
                "n_samples_total":    int(df.shape[0]),
                "n_samples_offload":  int(mask.sum()),
                "offload_fraction":   float(mask.mean()),
                "full_run":           summarize(df, None),
                "offload_dominant":   summarize(df, mask),
            }
        (out_dir / "run_stats.json").write_text(json.dumps(stats, indent=2))
        print(f"  [stats]  {out_dir/'run_stats.json'}")
    finally:
        stop_local_prom(prom)


if __name__ == "__main__":
    main()
