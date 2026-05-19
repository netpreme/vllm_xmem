#!/usr/bin/env python3
"""
Endpoint-delta summary of TTFT / ITL / E2E / queue / prefill / decode for every
snapshot, in two regimes:

  Tier 1: the entire snapshot duration [t_start_unix, t_end_unix].
  Tier 2: only the offload-dominant windows — contiguous intervals where the
          per-bin offload-token share is >50% and lasts at least 30s.

How averages are computed:
  For each histogram (vllm:..._sum, vllm:..._count) we take the *cumulative*
  counter value at the two endpoints of an interval and divide:

      mean(metric) over [a,b]  =  (sum_b - sum_a) / (count_b - count_a)

  This is the exact request-weighted mean over those requests, with no rate()
  smoothing window. For Tier 2 we sum the (sum_b - sum_a) and (count_b - count_a)
  across all qualifying windows, then divide — i.e. pooled mean.

Outputs:
  results_benchmarks/tier12_summary_per_run.csv         one row per (run, setup, tier)
  results_benchmarks/tier12_summary_per_concurrency.csv pooled across iters
  also prints the per-concurrency table to stdout.
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

import requests

ROOT_DEFAULT = Path(__file__).resolve().parents[1] / "benchmarks" / "results_benchmarks"

# Histograms we want endpoint deltas for. Each entry: (out_name, prom_root, unit_scale)
# unit_scale converts the metric's natural unit to the display unit.
HISTOGRAMS = [
    ("ttft_ms",        "vllm:time_to_first_token_seconds",   1000.0),
    ("itl_ms",         "vllm:inter_token_latency_seconds",   1000.0),
    ("e2e_s",          "vllm:e2e_request_latency_seconds",      1.0),
    ("queue_ms",       "vllm:request_queue_time_seconds",    1000.0),
    ("prefill_s",      "vllm:request_prefill_time_seconds",     1.0),
    ("decode_s",       "vllm:request_decode_time_seconds",      1.0),
]

# Per-source prompt-token counters (for offload share + Tier-2 mask).
SOURCES = {
    "hbm":       "local_cache_hit",
    "offload":   "external_kv_transfer",
    "recompute": "local_compute",
}


# ── throwaway Prometheus ──────────────────────────────────────────────────────

def start_local_prom(snapshot_path: Path, port: int) -> subprocess.Popen:
    cfg = Path(tempfile.mkstemp(prefix="prom_tier_", suffix=".yml")[1])
    cfg.write_text("global:\n  scrape_interval: 60s\n")
    log = Path(f"/tmp/prom_tier_{port}.log")
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


# ── querying ───────────────────────────────────────────────────────────────────

def q_range(url: str, query: str, t0: float, t1: float, step: str) -> list[tuple[float, float]]:
    """Return [(unix_ts, value)] sorted by ts, possibly empty."""
    r = requests.get(f"{url}/api/v1/query_range",
                     params={"query": query, "start": t0, "end": t1, "step": step},
                     timeout=60)
    j = r.json()
    if j.get("status") != "success" or not j["data"]["result"]:
        return []
    # If multiple series come back, sum them (we only ever ask for one label set).
    series_by_ts: dict[float, float] = {}
    for res in j["data"]["result"]:
        for ts, v in res["values"]:
            if v in ("NaN", "+Inf", "-Inf"):
                continue
            series_by_ts[float(ts)] = series_by_ts.get(float(ts), 0.0) + float(v)
    return sorted(series_by_ts.items())


def endpoint_delta(samples: list[tuple[float, float]], t_a: float, t_b: float) -> float | None:
    """Return value at the sample nearest >= t_a and the sample nearest <= t_b,
    differenced. None if we can't bracket the interval."""
    if not samples:
        return None
    # Closest sample >= t_a
    lo = None
    for ts, v in samples:
        if ts >= t_a - 0.5:
            lo = (ts, v); break
    # Closest sample <= t_b
    hi = None
    for ts, v in reversed(samples):
        if ts <= t_b + 0.5:
            hi = (ts, v); break
    if lo is None or hi is None or hi[0] <= lo[0]:
        return None
    return hi[1] - lo[1]


def value_near(samples: list[tuple[float, float]], t: float, side: str) -> float | None:
    """Return value at the sample bracketing t. side='lo' (>= t) or 'hi' (<= t)."""
    if not samples:
        return None
    if side == "lo":
        for ts, v in samples:
            if ts >= t - 0.5:
                return v
        return None
    else:
        for ts, v in reversed(samples):
            if ts <= t + 0.5:
                return v
        return None


# ── per-snapshot work ─────────────────────────────────────────────────────────

def process_snapshot(level_dir: Path, port: int, step: str, bin_s: float,
                     offload_threshold: float, min_window_s: float) -> list[dict]:
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

            # Fetch per-source prompt-token cumulative series (used for both share + Tier-2 mask)
            src_samples: dict[str, list[tuple[float, float]]] = {}
            for nice, src_label in SOURCES.items():
                src_samples[nice] = q_range(
                    url,
                    f'vllm:prompt_tokens_by_source_total{{setup="{setup}",source="{src_label}"}}',
                    t0, t1, step)
            tot_samples = q_range(url, f'sum(vllm:prompt_tokens_by_source_total{sel})', t0, t1, step)
            if not tot_samples:
                continue

            # Fetch histogram cumulative _sum / _count series
            hist_samples: dict[str, dict[str, list]] = {}
            for nice, root, _scale in HISTOGRAMS:
                hist_samples[nice] = {
                    "sum":   q_range(url, f'sum({root}_sum{sel})',   t0, t1, step),
                    "count": q_range(url, f'sum({root}_count{sel})', t0, t1, step),
                }

            # ── Tier 1: full-run endpoint delta ────────────────────────────────
            row = {"level_dir": str(level_dir), "concurrency": cfg["concurrency"],
                   "run_id": level_dir.parent.name, "setup": setup, "tier": "full",
                   "n_windows": 1, "duration_s": t1 - t0}
            total_delta = endpoint_delta(tot_samples, t0, t1) or 0.0
            for nice in SOURCES:
                d = endpoint_delta(src_samples[nice], t0, t1) or 0.0
                row[f"{nice}_share"] = (d / total_delta) if total_delta > 0 else 0.0
                row[f"{nice}_tokens"] = d
            row["total_prompt_tokens"] = total_delta
            for nice, _root, scale in HISTOGRAMS:
                ds = endpoint_delta(hist_samples[nice]["sum"], t0, t1) or 0.0
                dc = endpoint_delta(hist_samples[nice]["count"], t0, t1) or 0.0
                row[f"{nice}_n_req"] = dc
                row[nice] = (ds / dc * scale) if dc > 0 else None
            rows.append(row)

            # ── Tier 2: offload-dominant windows ──────────────────────────────
            # Build per-bin offload share by stepping bin_s seconds at a time.
            t_grid = []
            cur = t0
            while cur <= t1 + 1e-6:
                t_grid.append(cur); cur += bin_s
            if t_grid[-1] < t1:
                t_grid.append(t1)

            # Per-bin endpoint deltas → instantaneous share
            offload_series = src_samples["offload"]
            in_window = []
            for i in range(len(t_grid) - 1):
                a, b = t_grid[i], t_grid[i + 1]
                d_off = endpoint_delta(offload_series, a, b) or 0.0
                d_tot = endpoint_delta(tot_samples,    a, b) or 0.0
                share = (d_off / d_tot) if d_tot > 0 else 0.0
                in_window.append(share > offload_threshold)

            # Group contiguous Trues, drop short runs
            windows: list[tuple[float, float]] = []
            i = 0
            while i < len(in_window):
                if in_window[i]:
                    j = i
                    while j + 1 < len(in_window) and in_window[j + 1]:
                        j += 1
                    w_a, w_b = t_grid[i], t_grid[j + 1]
                    if (w_b - w_a) >= min_window_s:
                        windows.append((w_a, w_b))
                    i = j + 1
                else:
                    i += 1

            if windows:
                row2 = {"level_dir": str(level_dir), "concurrency": cfg["concurrency"],
                        "run_id": level_dir.parent.name, "setup": setup, "tier": "offload>50%",
                        "n_windows": len(windows),
                        "duration_s": sum(b - a for a, b in windows)}
                # Per-source token deltas pooled across windows
                pooled_total = 0.0
                pooled_src = {k: 0.0 for k in SOURCES}
                for a, b in windows:
                    pooled_total += endpoint_delta(tot_samples, a, b) or 0.0
                    for nice in SOURCES:
                        pooled_src[nice] += endpoint_delta(src_samples[nice], a, b) or 0.0
                row2["total_prompt_tokens"] = pooled_total
                for nice in SOURCES:
                    row2[f"{nice}_tokens"] = pooled_src[nice]
                    row2[f"{nice}_share"] = (pooled_src[nice] / pooled_total) if pooled_total > 0 else 0.0
                # Histogram pooled deltas
                for nice, _root, scale in HISTOGRAMS:
                    pooled_sum = 0.0
                    pooled_cnt = 0.0
                    for a, b in windows:
                        pooled_sum += endpoint_delta(hist_samples[nice]["sum"],   a, b) or 0.0
                        pooled_cnt += endpoint_delta(hist_samples[nice]["count"], a, b) or 0.0
                    row2[f"{nice}_n_req"] = pooled_cnt
                    row2[nice] = (pooled_sum / pooled_cnt * scale) if pooled_cnt > 0 else None
                rows.append(row2)
            else:
                rows.append({"level_dir": str(level_dir),
                             "concurrency": cfg["concurrency"],
                             "run_id": level_dir.parent.name,
                             "setup": setup, "tier": "offload>50%",
                             "n_windows": 0, "duration_s": 0.0,
                             "total_prompt_tokens": 0.0,
                             **{f"{k}_tokens": 0.0 for k in SOURCES},
                             **{f"{k}_share":  0.0 for k in SOURCES},
                             **{f"{nice}_n_req": 0.0 for nice, _, _ in HISTOGRAMS},
                             **{nice: None for nice, _, _ in HISTOGRAMS}})
    finally:
        stop_local_prom(prom)
    return rows


# ── aggregation: pooled across iters per (concurrency, setup, tier) ───────────

def pooled_summary(per_run_rows: list[dict]) -> list[dict]:
    """Re-pool means across runs by adding the (sum_delta, count_delta) again.
    But per-run rows already collapsed to means — to re-pool we need per-run
    sums and counts. We DO have per-metric n_req per row, and we have the mean
    value. The pooled mean across runs is then:
        Σ (mean_i * n_req_i) / Σ n_req_i
    which is exactly the request-weighted mean.
    """
    out_groups: dict[tuple, list[dict]] = {}
    for r in per_run_rows:
        key = (r["concurrency"], r["setup"], r["tier"])
        out_groups.setdefault(key, []).append(r)

    out_rows = []
    for (c, setup, tier), rs in sorted(out_groups.items()):
        agg = {"concurrency": c, "setup": setup, "tier": tier,
               "n_iters": len(rs),
               "n_iters_with_window": sum(1 for r in rs if r.get("n_windows", 0) > 0),
               "total_duration_s": sum(r.get("duration_s", 0) or 0.0 for r in rs)}
        # Token shares: pooled across all iters
        tot = sum(r.get("total_prompt_tokens", 0.0) or 0.0 for r in rs)
        for nice in SOURCES:
            agg[f"{nice}_share"] = (sum(r.get(f"{nice}_tokens", 0.0) or 0.0 for r in rs) / tot
                                    if tot > 0 else None)
        # Histogram metrics: request-weighted mean
        for nice, _root, _scale in HISTOGRAMS:
            num = 0.0; den = 0.0
            for r in rs:
                if r.get(nice) is None: continue
                n = r.get(f"{nice}_n_req", 0.0) or 0.0
                num += float(r[nice]) * n
                den += n
            agg[nice]            = (num / den) if den > 0 else None
            agg[f"{nice}_n_req"] = den
        out_rows.append(agg)
    return out_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r.keys()))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def print_summary_table(rows: list[dict]) -> None:
    rows = sorted(rows, key=lambda r: (r["concurrency"], r["tier"], r["setup"]))
    print()
    print("=" * 140)
    print(f"  Pooled (request-weighted) means across iterations per (concurrency, setup, tier)")
    print("=" * 140)
    hdr = (f"{'C':>3} {'tier':>12} {'setup':>13} {'iters':>5} {'dur_s':>7} "
           f"{'TTFT_ms':>9} {'ITL_ms':>8} {'E2E_s':>7} {'queue_ms':>9} "
           f"{'prefill_s':>10} {'decode_s':>9} "
           f"{'hbm%':>6} {'off%':>6} {'rec%':>6} {'n_req':>7}")
    print(hdr)
    print("-" * 140)
    def fmt(v, digits=1):
        if v is None: return "    -"
        return f"{v:>.{digits}f}"
    last_c = None; last_tier = None
    for r in rows:
        if last_c is not None and r["concurrency"] != last_c:
            print()
        elif last_tier is not None and r["tier"] != last_tier:
            pass
        last_c = r["concurrency"]; last_tier = r["tier"]
        # n_req: use TTFT count as the canonical
        nreq = r.get("ttft_ms_n_req", 0.0) or 0.0
        print(f"{r['concurrency']:>3} {r['tier']:>12} {r['setup']:>13} "
              f"{r['n_iters']:>5} {r['total_duration_s']:>7.0f} "
              f"{fmt(r.get('ttft_ms')):>9} {fmt(r.get('itl_ms')):>8} "
              f"{fmt(r.get('e2e_s'),2):>7} {fmt(r.get('queue_ms')):>9} "
              f"{fmt(r.get('prefill_s'),2):>10} {fmt(r.get('decode_s'),2):>9} "
              f"{fmt((r.get('hbm_share') or 0)*100):>6} "
              f"{fmt((r.get('offload_share') or 0)*100):>6} "
              f"{fmt((r.get('recompute_share') or 0)*100):>6} "
              f"{nreq:>7.0f}")
    print("=" * 140)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT_DEFAULT)
    ap.add_argument("--step",      default="5s", help="prometheus query_range step (default 5s)")
    ap.add_argument("--bin",       type=float, default=10.0,
                    help="seconds per bin used to classify offload-dominant intervals (default 10)")
    ap.add_argument("--threshold", type=float, default=0.50,
                    help="offload-share threshold (default 0.50)")
    ap.add_argument("--min-window-s", type=float, default=30.0,
                    help="minimum contiguous offload-dominant window length (default 30s)")
    ap.add_argument("--port-base", type=int, default=9200)
    ap.add_argument("--out-prefix", default="tier12_summary")
    args = ap.parse_args()

    snapshots: list[Path] = sorted(
        p for p in args.root.glob("bench_sweep_*/c*")
        if (p / "config.json").exists() and (p / "prom_snapshot").exists()
    )
    if not snapshots:
        sys.exit(f"No snapshots under {args.root}")
    print(f"  Found {len(snapshots)} snapshot(s) under {args.root}")

    per_run: list[dict] = []
    for i, level_dir in enumerate(snapshots):
        print(f"  [{i+1}/{len(snapshots)}] {level_dir.parent.name}/{level_dir.name}", flush=True)
        try:
            per_run.extend(process_snapshot(
                level_dir, args.port_base + (i % 50),
                args.step, args.bin, args.threshold, args.min_window_s,
            ))
        except Exception as e:
            print(f"    ERROR: {e}", flush=True)

    per_run_path = args.root / f"{args.out_prefix}_per_run.csv"
    write_csv(per_run_path, per_run)
    print(f"\n  Wrote per-run: {per_run_path}  ({len(per_run)} rows)")

    pooled = pooled_summary(per_run)
    pooled_path = args.root / f"{args.out_prefix}_per_concurrency.csv"
    write_csv(pooled_path, pooled)
    print(f"  Wrote pooled:  {pooled_path}  ({len(pooled)} rows)")

    print_summary_table(pooled)


if __name__ == "__main__":
    main()
