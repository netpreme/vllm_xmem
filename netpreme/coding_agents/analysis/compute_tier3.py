#!/usr/bin/env python3
"""
Tier 3 — matched-workload comparison of mtier vs cpu.

For every snapshot, bin the run into uniform `--bin` second windows and, using
endpoint-delta on cumulative Prometheus counters in each bin, compute:

    n_req              =  Δ(ttft_count)                       requests completed in bin
    mean_ttft_ms       =  Δ(ttft_sum)     / n_req * 1000
    mean_prefill_s     =  Δ(prefill_sum)  / Δ(prefill_count)
    off_per_req        =  Δ(prompt_tokens_by_source{external_kv_transfer}) / n_req
    hbm_per_req        =  Δ(prompt_tokens_by_source{local_cache_hit})      / n_req
    rec_per_req        =  Δ(prompt_tokens_by_source{local_compute})        / n_req

Then bucket bins by `off_per_req` (default 5000-tok wide buckets) and compute
the request-weighted mean TTFT in each bucket, separately for mtier and cpu.
This answers the question:

    "At the same offload-tokens-loaded-per-request, what TTFT does each
     system deliver?"

— i.e. it controls for the volume of offload work and isolates the system's
response to that work. This addresses the concern that mtier might do more
transfer in the same window.

Outputs:
  tier3_per_bin.csv                  one row per (run, setup, bin)
  tier3_bucketed_per_concurrency.csv per (concurrency, setup, bucket): n_req-weighted means
  tier3_matched_workload.png         per-concurrency: TTFT vs off_per_req, both setups
"""
import argparse
import csv
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
import requests

ROOT_DEFAULT = Path(__file__).resolve().parents[1] / "benchmarks" / "results_benchmarks"

_COLORS = {"hybrid-mtier": "#5e3c99", "hybrid-cpu": "#e66101"}

SOURCES = {
    "hbm":       "local_cache_hit",
    "offload":   "external_kv_transfer",
    "recompute": "local_compute",
}


def start_local_prom(snapshot_path: Path, port: int) -> subprocess.Popen:
    cfg = Path(tempfile.mkstemp(prefix="prom_t3_", suffix=".yml")[1])
    cfg.write_text("global:\n  scrape_interval: 60s\n")
    log = Path(f"/tmp/prom_t3_{port}.log")
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
            raise RuntimeError("Prometheus startup timed out")
        time.sleep(0.3)


def stop_local_prom(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        pass


def q_range(url: str, query: str, t0: float, t1: float, step: str) -> list[tuple[float, float]]:
    r = requests.get(f"{url}/api/v1/query_range",
                     params={"query": query, "start": t0, "end": t1, "step": step},
                     timeout=60)
    j = r.json()
    if j.get("status") != "success" or not j["data"]["result"]:
        return []
    series_by_ts: dict[float, float] = {}
    for res in j["data"]["result"]:
        for ts, v in res["values"]:
            if v in ("NaN", "+Inf", "-Inf"):
                continue
            series_by_ts[float(ts)] = series_by_ts.get(float(ts), 0.0) + float(v)
    return sorted(series_by_ts.items())


def endpoint_delta(samples: list[tuple[float, float]], t_a: float, t_b: float) -> float | None:
    if not samples:
        return None
    lo = next(((ts, v) for ts, v in samples if ts >= t_a - 0.5), None)
    hi = next(((ts, v) for ts, v in reversed(samples) if ts <= t_b + 0.5), None)
    if lo is None or hi is None or hi[0] <= lo[0]:
        return None
    return hi[1] - lo[1]


def process_snapshot(level_dir: Path, port: int, step: str, bin_s: float) -> list[dict]:
    cfg = json.loads((level_dir / "config.json").read_text())
    t0 = float(cfg["t_start_unix"])
    t1 = float(cfg["t_end_unix"])
    snap = level_dir / "prom_snapshot"
    rows: list[dict] = []
    prom = start_local_prom(snap, port)
    try:
        url = f"http://localhost:{port}"
        for setup in ("hybrid-mtier", "hybrid-cpu"):
            sel = f'{{setup="{setup}"}}'
            samples = {
                "ttft_sum":    q_range(url, f'sum(vllm:time_to_first_token_seconds_sum{sel})',   t0, t1, step),
                "ttft_count":  q_range(url, f'sum(vllm:time_to_first_token_seconds_count{sel})', t0, t1, step),
                "pre_sum":     q_range(url, f'sum(vllm:request_prefill_time_seconds_sum{sel})',  t0, t1, step),
                "pre_count":   q_range(url, f'sum(vllm:request_prefill_time_seconds_count{sel})', t0, t1, step),
                "off":         q_range(url, f'vllm:prompt_tokens_by_source_total{{setup="{setup}",source="external_kv_transfer"}}', t0, t1, step),
                "hbm":         q_range(url, f'vllm:prompt_tokens_by_source_total{{setup="{setup}",source="local_cache_hit"}}',      t0, t1, step),
                "rec":         q_range(url, f'vllm:prompt_tokens_by_source_total{{setup="{setup}",source="local_compute"}}',        t0, t1, step),
            }
            if not samples["ttft_count"]:
                continue

            # Build uniform time grid t0, t0+bin, ..., t1
            grid = [t0]
            while grid[-1] + bin_s <= t1:
                grid.append(grid[-1] + bin_s)
            if grid[-1] < t1:
                grid.append(t1)

            for i in range(len(grid) - 1):
                a, b = grid[i], grid[i + 1]
                n_req = endpoint_delta(samples["ttft_count"], a, b) or 0.0
                if n_req < 1:
                    continue
                ttft_sum = endpoint_delta(samples["ttft_sum"], a, b) or 0.0
                pre_sum  = endpoint_delta(samples["pre_sum"],  a, b) or 0.0
                pre_cnt  = endpoint_delta(samples["pre_count"], a, b) or 0.0
                off      = endpoint_delta(samples["off"], a, b) or 0.0
                hbm      = endpoint_delta(samples["hbm"], a, b) or 0.0
                rec      = endpoint_delta(samples["rec"], a, b) or 0.0
                rows.append({
                    "concurrency": cfg["concurrency"],
                    "run_id":      level_dir.parent.name,
                    "setup":       setup,
                    "t_bin_start": a - t0,
                    "duration_s":  b - a,
                    "n_req":       n_req,
                    "mean_ttft_ms":  (ttft_sum / n_req * 1000) if n_req > 0 else None,
                    "mean_prefill_s": (pre_sum / pre_cnt)      if pre_cnt > 0 else None,
                    "off_per_req":   off / n_req,
                    "hbm_per_req":   hbm / n_req,
                    "rec_per_req":   rec / n_req,
                })
    finally:
        stop_local_prom(prom)
    return rows


def bucket_per_concurrency(rows: list[dict], min_req: int, bucket_size: int,
                           min_bucket_req: int) -> list[dict]:
    """Bucket bins by off_per_req, compute request-weighted TTFT/prefill per bucket."""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        if r["n_req"] < min_req: continue
        if r["mean_ttft_ms"] is None: continue
        bkt = int(r["off_per_req"] // bucket_size) * bucket_size
        groups.setdefault((r["concurrency"], r["setup"], bkt), []).append(r)
    out = []
    for (c, setup, bkt), rs in sorted(groups.items()):
        n_req_total = sum(r["n_req"] for r in rs)
        if n_req_total < min_bucket_req: continue
        ttft_w = sum(r["mean_ttft_ms"] * r["n_req"] for r in rs) / n_req_total
        pf_rs = [r for r in rs if r["mean_prefill_s"] is not None]
        if pf_rs:
            pf_w = (sum(r["mean_prefill_s"] * r["n_req"] for r in pf_rs)
                    / sum(r["n_req"] for r in pf_rs))
        else:
            pf_w = None
        off_mean = sum(r["off_per_req"] * r["n_req"] for r in rs) / n_req_total
        out.append({
            "concurrency":      c,
            "setup":            setup,
            "bucket_lo":        bkt,
            "bucket_hi":        bkt + bucket_size,
            "n_bins":           len(rs),
            "n_req":            n_req_total,
            "off_per_req_mean": off_mean,
            "mean_ttft_ms":     ttft_w,
            "mean_prefill_s":   pf_w,
        })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows: return
    cols = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows: w.writerow(r)


def plot_matched(buckets: list[dict], out_path: Path) -> None:
    """Per-concurrency: TTFT (and prefill) vs off_per_req, mtier vs cpu overlaid."""
    concs = sorted({r["concurrency"] for r in buckets if r["concurrency"] >= 13})
    if not concs: return
    n = len(concs)
    fig, axes = plt.subplots(2, n, figsize=(4.5 * n, 7.5), sharex="col", squeeze=False)
    for col, c in enumerate(concs):
        ax_t, ax_p = axes[0][col], axes[1][col]
        for setup in ("hybrid-mtier", "hybrid-cpu"):
            rs = [r for r in buckets if r["concurrency"] == c and r["setup"] == setup]
            rs.sort(key=lambda r: r["off_per_req_mean"])
            if not rs: continue
            xs = np.array([r["off_per_req_mean"] for r in rs])
            ys_t = np.array([r["mean_ttft_ms"]    for r in rs])
            sizes = np.clip(np.array([r["n_req"] for r in rs]) * 0.15, 15, 150)
            ax_t.plot(xs, ys_t, color=_COLORS[setup], linewidth=2, marker="o",
                      markersize=0, label=setup)
            ax_t.scatter(xs, ys_t, s=sizes, color=_COLORS[setup], alpha=0.6, edgecolor="none")
            ys_p = np.array([r["mean_prefill_s"] if r["mean_prefill_s"] is not None else np.nan
                             for r in rs])
            ax_p.plot(xs, ys_p, color=_COLORS[setup], linewidth=2, marker="o", markersize=0,
                      label=setup)
            ax_p.scatter(xs, ys_p, s=sizes, color=_COLORS[setup], alpha=0.6, edgecolor="none")
        ax_t.set_title(f"C = {c}")
        ax_t.grid(alpha=0.3)
        ax_p.grid(alpha=0.3)
        if col == 0:
            ax_t.set_ylabel("Mean TTFT (ms)")
            ax_p.set_ylabel("Mean prefill time (s)")
        ax_p.set_xlabel("Offload tokens loaded per request")
        if col == n - 1:
            ax_t.legend(loc="lower right", fontsize=9)
    fig.suptitle("Tier 3 — matched-workload comparison\n"
                 "At the same offload-tokens-per-request, what does each system deliver?\n"
                 "(point size = number of requests in that bucket)",
                 fontsize=11, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def print_table(buckets: list[dict]) -> None:
    """Side-by-side mtier vs cpu TTFT per off-per-req bucket, plus delta."""
    print()
    print("=" * 110)
    print("  Tier 3 — TTFT at matched offload-tokens-per-request bucket")
    print("=" * 110)
    print(f"  {'C':>3} {'off/req':>14} {'cpu_TTFT_ms':>12} {'mtier_TTFT_ms':>14} "
          f"{'Δ(mt-cpu)_ms':>13} {'cpu_n_req':>10} {'mtier_n_req':>11}")
    print("-" * 110)
    by_key: dict[tuple, dict[str, dict]] = {}
    for r in buckets:
        by_key.setdefault((r["concurrency"], r["bucket_lo"]), {})[r["setup"]] = r
    last_c = None
    for (c, lo), pair in sorted(by_key.items()):
        if last_c is not None and c != last_c:
            print()
        last_c = c
        cpu = pair.get("hybrid-cpu"); mtier = pair.get("hybrid-mtier")
        cpu_t   = f"{cpu['mean_ttft_ms']:.0f}"   if cpu else "  -"
        mt_t    = f"{mtier['mean_ttft_ms']:.0f}" if mtier else "  -"
        delta   = (f"{mtier['mean_ttft_ms'] - cpu['mean_ttft_ms']:+.0f}"
                   if cpu and mtier else "  -")
        cpu_n   = f"{int(cpu['n_req'])}"   if cpu else "-"
        mt_n    = f"{int(mtier['n_req'])}" if mtier else "-"
        print(f"  {c:>3} {f'{lo}-{lo+5000}':>14} {cpu_t:>12} {mt_t:>14} {delta:>13} "
              f"{cpu_n:>10} {mt_n:>11}")
    print("=" * 110)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    ap.add_argument("--step",      default="5s")
    ap.add_argument("--bin",       type=float, default=15.0,
                    help="seconds per bin (default 15)")
    ap.add_argument("--min-req",   type=int, default=3,
                    help="drop bins with fewer than this many completed requests (default 3)")
    ap.add_argument("--bucket-size",   type=int, default=5000,
                    help="off_per_req bucket width in tokens (default 5000)")
    ap.add_argument("--min-bucket-req", type=int, default=20,
                    help="drop buckets with fewer than this many requests total (default 20)")
    ap.add_argument("--port-base", type=int, default=9500)
    ap.add_argument("--out-prefix", default="tier3")
    ap.add_argument("--reuse-csv", action="store_true",
                    help="skip re-extracting from snapshots; load existing tier3_per_bin.csv")
    args = ap.parse_args()

    per_bin_path = args.root / f"{args.out_prefix}_per_bin.csv"

    if args.reuse_csv and per_bin_path.exists():
        per_bin = []
        for r in csv.DictReader(open(per_bin_path)):
            for k in ("concurrency", "n_req"):
                r[k] = float(r[k])
            for k in ("mean_ttft_ms", "mean_prefill_s",
                      "off_per_req", "hbm_per_req", "rec_per_req"):
                r[k] = float(r[k]) if r[k] not in ("", "None") else None
            per_bin.append(r)
        print(f"  Loaded per-bin CSV: {per_bin_path}  ({len(per_bin)} rows)")
    else:
        snaps = sorted(p for p in args.root.glob("bench_sweep_*/c*")
                       if (p / "config.json").exists() and (p / "prom_snapshot").exists())
        if not snaps:
            sys.exit(f"No snapshots under {args.root}")
        print(f"  Found {len(snaps)} snapshot(s)")

        per_bin = []
        for i, level_dir in enumerate(snaps):
            print(f"  [{i+1}/{len(snaps)}] {level_dir.parent.name}/{level_dir.name}", flush=True)
            try:
                per_bin.extend(process_snapshot(level_dir, args.port_base + (i % 50),
                                                args.step, args.bin))
            except Exception as e:
                print(f"    ERROR: {e}")
        write_csv(per_bin_path, per_bin)
        print(f"\n  Wrote per-bin:    {per_bin_path}  ({len(per_bin)} rows)")

    buckets = bucket_per_concurrency(per_bin, args.min_req,
                                     args.bucket_size, args.min_bucket_req)
    bkt_path = args.root / f"{args.out_prefix}_bucketed_per_concurrency.csv"
    write_csv(bkt_path, buckets)
    print(f"  Wrote bucketed:   {bkt_path}  ({len(buckets)} rows)")

    plot_path = args.root / f"{args.out_prefix}_matched_workload.png"
    plot_matched(buckets, plot_path)
    print(f"  Wrote plot:       {plot_path}")

    print_table(buckets)


if __name__ == "__main__":
    main()
