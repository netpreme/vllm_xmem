#!/usr/bin/env python3
"""Extract inference metrics from a replay (or capture) level directory.

Reads:
  <level_dir>/config.json          — concurrency, setups[], t_start/end_unix, per_setup
  <level_dir>/prom_snapshot/       — Prometheus TSDB

Starts a throwaway Prometheus on the snapshot, queries per-setup metrics
(TTFT, ITL, E2E, queue-time avg + p50/p95/p99; HBM hit, offload hit,
HBM use, recompute), then writes one CSV row per setup.

Usage:
    extract_metrics.py --level-dir <path> --output metrics.csv [--append]
"""
import argparse
import csv
import json
import os
import signal
import subprocess
import tempfile
import time
from pathlib import Path

import requests


_PROM_FIELDS = [
    "concurrency", "setup", "vllm_port", "gpus",
    "kind", "capture_dir",
    "t_start_unix", "t_end_unix", "duration_s",
    "sessions_total", "sessions_completed", "turns_attempted", "turns_ok", "turns_error",
    "n_requests",
    "avg_ttft_ms",  "p50_ttft_ms",  "p95_ttft_ms",  "p99_ttft_ms",
    "avg_itl_ms",   "p50_itl_ms",   "p95_itl_ms",   "p99_itl_ms",
    "avg_e2e_s",    "p50_e2e_s",    "p95_e2e_s",    "p99_e2e_s",
    "avg_queue_ms", "p50_queue_ms", "p95_queue_ms", "p99_queue_ms",
    "avg_out_tps", "hbm_use_pct", "hbm_hit_pct", "offload_hit_pct",
    "offload_active_pct",
    "offload_gb_total",
    "offload_gb_gpu_to_cpu",
    "offload_gb_cpu_to_gpu",
    "level_dir",
]


def start_local_prom(snapshot_path: Path, port: int) -> subprocess.Popen:
    cfg = Path(tempfile.mkstemp(prefix="prom_extract_", suffix=".yml")[1])
    cfg.write_text("global:\n  scrape_interval: 60s\n")
    log = Path(f"/tmp/prom_extract_{port}.log")
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
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"Prometheus died during startup (see {log})")
        if time.monotonic() - t0 > 30:
            try: os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception: pass
            raise RuntimeError("Prometheus startup timed out")
        time.sleep(0.3)


def stop_local_prom(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=5)
    except Exception:
        try: proc.kill()
        except Exception: pass


def _q_instant(prom_url: str, query: str, at_time: float) -> float | None:
    """Single scalar query at a given Unix time. Returns float or None."""
    try:
        r = requests.get(f"{prom_url}/api/v1/query",
                         params={"query": query, "time": at_time},
                         timeout=15)
        d = r.json().get("data", {}).get("result", [])
        if not d:
            return None
        v = d[0]["value"][1]
        if v in ("NaN", "+Inf", "-Inf"):
            return None
        return float(v)
    except Exception:
        return None


def _metric_block(prom_url: str, sel: str, window: str, at_time: float) -> dict:
    """Pull TTFT/ITL/E2E/queue avg+p50/p95/p99 + cache/util/throughput for one setup."""
    out: dict = {}

    # Histogram metrics: avg = sum/count; quantiles from bucket increase.
    for label, hist in [
        ("ttft",  "vllm:time_to_first_token_seconds"),
        ("itl",   "vllm:inter_token_latency_seconds"),
        ("e2e",   "vllm:e2e_request_latency_seconds"),
        ("queue", "vllm:request_queue_time_seconds"),
    ]:
        avg = _q_instant(prom_url,
            f'increase({hist}_sum{sel}[{window}]) / clamp_min(increase({hist}_count{sel}[{window}]), 1)',
            at_time)
        p50 = _q_instant(prom_url,
            f'histogram_quantile(0.5,  sum by (le) (increase({hist}_bucket{sel}[{window}])))',
            at_time)
        p95 = _q_instant(prom_url,
            f'histogram_quantile(0.95, sum by (le) (increase({hist}_bucket{sel}[{window}])))',
            at_time)
        p99 = _q_instant(prom_url,
            f'histogram_quantile(0.99, sum by (le) (increase({hist}_bucket{sel}[{window}])))',
            at_time)
        out[f"avg_{label}"] = avg
        out[f"p50_{label}"] = p50
        out[f"p95_{label}"] = p95
        out[f"p99_{label}"] = p99

    # Total /v1/messages requests served (any path; vLLM doesn't split by path).
    out["n_requests"] = _q_instant(prom_url,
        f'sum(increase(vllm:e2e_request_latency_seconds_count{sel}[{window}]))',
        at_time)

    # Output token throughput (tokens/sec averaged over window).
    out["avg_out_tps"] = _q_instant(prom_url,
        f'rate(vllm:generation_tokens_total{sel}[{window}])',
        at_time)

    # KV cache utilization (avg over window, as a fraction 0-1).
    out["hbm_use"] = _q_instant(prom_url,
        f'avg_over_time(vllm:kv_cache_usage_perc{sel}[{window}])',
        at_time)

    # Prefix-cache hit rates (HBM and external/offload), as fractions.
    out["hbm_hit"] = _q_instant(prom_url,
        f'increase(vllm:prefix_cache_hits_total{sel}[{window}]) / clamp_min(increase(vllm:prefix_cache_queries_total{sel}[{window}]), 1)',
        at_time)
    out["off_hit"] = _q_instant(prom_url,
        f'increase(vllm:external_prefix_cache_hits_total{sel}[{window}]) / clamp_min(increase(vllm:prefix_cache_queries_total{sel}[{window}]), 1)',
        at_time)

    # Offload bandwidth: cumulative bytes by transfer_type. Metric name has
    # OpenMetrics _total suffix; labels use CPU_to_GPU / GPU_to_CPU (case sensitive).
    sel_g2c = sel[:-1] + ',transfer_type="GPU_to_CPU"}'
    sel_c2g = sel[:-1] + ',transfer_type="CPU_to_GPU"}'
    out["off_gb_g2c"] = _q_instant(prom_url,
        f'sum(increase(vllm:kv_offload_total_bytes_total{sel_g2c}[{window}])) / 1e9',
        at_time)
    out["off_gb_c2g"] = _q_instant(prom_url,
        f'sum(increase(vllm:kv_offload_total_bytes_total{sel_c2g}[{window}])) / 1e9',
        at_time)
    out["off_gb_total"] = _q_instant(prom_url,
        f'sum(increase(vllm:kv_offload_total_bytes_total{sel}[{window}])) / 1e9',
        at_time)

    # "Percent of time using offload tier": fraction of 10s buckets in the
    # window during which the offload had nonzero bandwidth (either direction).
    out["off_active"] = _q_instant(prom_url,
        f'avg_over_time((sum(rate(vllm:kv_offload_total_bytes_total{sel}[10s])) > bool 0)[{window}:10s])',
        at_time)

    return out


def extract_for_level(level_dir: Path, port: int = 9097) -> list[dict]:
    """Return one dict per setup with all metrics filled in."""
    cfg = json.loads((level_dir / "config.json").read_text())
    snapshot = level_dir / "prom_snapshot"
    if not snapshot.exists():
        raise FileNotFoundError(f"No prom_snapshot/ under {level_dir}")

    t_start = float(cfg["t_start_unix"])
    t_end   = float(cfg["t_end_unix"])
    window  = f"{int(t_end - t_start)}s"
    concurrency = int(cfg.get("concurrency", 0))
    kind = cfg.get("kind", "sweep")
    capture_dir = cfg.get("capture_dir", "")
    setups = cfg.get("setups", [])
    per_setup = cfg.get("per_setup", {}) if isinstance(cfg.get("per_setup"), dict) else {}

    proc = start_local_prom(snapshot, port)
    prom_url = f"http://localhost:{port}"
    try:
        rows: list[dict] = []
        for s in setups:
            setup_name = s["setup"]
            vllm_port  = s["port"]
            gpus       = s["gpus"]
            sel = '{instance="localhost:%s"}' % vllm_port
            m = _metric_block(prom_url, sel, window, t_end)

            ps = per_setup.get(setup_name, {})
            # In capture (non-replay) runs, per-setup info lives in `setups` dict-style.
            cap_setups_dict = cfg.get("setups")
            if isinstance(cap_setups_dict, dict):
                ps_alt = cap_setups_dict.get(setup_name, {})
            else:
                ps_alt = {}

            def _ms(v): return round(v * 1000, 2) if v is not None else None
            def _s(v):  return round(v, 3) if v is not None else None
            def _pct(v): return round(v * 100, 2) if v is not None else None

            row = {
                "concurrency":        concurrency,
                "setup":              setup_name,
                "vllm_port":          vllm_port,
                "gpus":               gpus,
                "kind":               kind,
                "capture_dir":        capture_dir,
                "t_start_unix":       t_start,
                "t_end_unix":         t_end,
                "duration_s":         round(t_end - t_start, 2),
                "sessions_total":     ps.get("n_sessions") or ps_alt.get("n_tasks_started") or 0,
                "sessions_completed": ps.get("n_completed") or ps_alt.get("n_tasks_completed") or 0,
                "turns_attempted":    ps.get("n_turns", 0),
                "turns_ok":           ps.get("n_ok", 0),
                "turns_error":        ps.get("n_error", 0),
                "n_requests":         int(m["n_requests"]) if m["n_requests"] is not None else None,
                "avg_ttft_ms":  _ms(m["avg_ttft"]),
                "p50_ttft_ms":  _ms(m["p50_ttft"]),
                "p95_ttft_ms":  _ms(m["p95_ttft"]),
                "p99_ttft_ms":  _ms(m["p99_ttft"]),
                "avg_itl_ms":   _ms(m["avg_itl"]),
                "p50_itl_ms":   _ms(m["p50_itl"]),
                "p95_itl_ms":   _ms(m["p95_itl"]),
                "p99_itl_ms":   _ms(m["p99_itl"]),
                "avg_e2e_s":    _s(m["avg_e2e"]),
                "p50_e2e_s":    _s(m["p50_e2e"]),
                "p95_e2e_s":    _s(m["p95_e2e"]),
                "p99_e2e_s":    _s(m["p99_e2e"]),
                "avg_queue_ms": _ms(m["avg_queue"]),
                "p50_queue_ms": _ms(m["p50_queue"]),
                "p95_queue_ms": _ms(m["p95_queue"]),
                "p99_queue_ms": _ms(m["p99_queue"]),
                "avg_out_tps":  round(m["avg_out_tps"], 2) if m["avg_out_tps"] is not None else None,
                "hbm_use_pct":  _pct(m["hbm_use"]),
                "hbm_hit_pct":  _pct(m["hbm_hit"]),
                "offload_hit_pct": _pct(m["off_hit"]),
                "offload_active_pct":    _pct(m.get("off_active")),
                "offload_gb_total":      round(m["off_gb_total"], 3) if m.get("off_gb_total") is not None else None,
                "offload_gb_gpu_to_cpu": round(m["off_gb_g2c"], 3) if m.get("off_gb_g2c") is not None else None,
                "offload_gb_cpu_to_gpu": round(m["off_gb_c2g"], 3) if m.get("off_gb_c2g") is not None else None,
                "level_dir":    str(level_dir),
            }
            rows.append(row)
        return rows
    finally:
        stop_local_prom(proc)


def write_csv(rows: list[dict], path: Path, append: bool = False) -> None:
    new_file = not path.exists() or path.stat().st_size == 0
    mode = "a" if (append and not new_file) else "w"
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=_PROM_FIELDS)
        if mode == "w":
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in _PROM_FIELDS})


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--level-dir", required=True,
        help="Path to a per-level dir containing config.json + prom_snapshot/")
    ap.add_argument("--output", required=True, help="CSV path to write")
    ap.add_argument("--append", action="store_true",
        help="Append rows (skip header) if file exists; otherwise overwrite")
    ap.add_argument("--port", type=int, default=9097,
        help="Local port for the throwaway Prometheus")
    args = ap.parse_args()

    level_dir = Path(args.level_dir).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = extract_for_level(level_dir, port=args.port)
    write_csv(rows, out, append=args.append)
    print(f"Wrote {len(rows)} row(s) → {out}")
    def _f(v, suffix="", width=6):
        s = "-" if v is None else str(v)
        return f"{s:>{width}}{suffix}"
    for r in rows:
        print(f"  c={r['concurrency']:>3} {r['setup']:<14} "
              f"ttft={_f(r['avg_ttft_ms'],'ms',7)} p95={_f(r['p95_ttft_ms'],'ms',8)}  "
              f"queue={_f(r['avg_queue_ms'],'ms',6)} p95={_f(r['p95_queue_ms'],'ms',7)}  "
              f"e2e={_f(r['avg_e2e_s'],'s',5)}  "
              f"out={_f(r['avg_out_tps'],'t/s',5)}  "
              f"hbm_hit={_f(r['hbm_hit_pct'],'%',5)}  off_hit={_f(r['offload_hit_pct'],'%',5)}  "
              f"off_active={_f(r['offload_active_pct'],'%',5)}  "
              f"off_GB={_f(r['offload_gb_total'],'',6)}")


if __name__ == "__main__":
    main()
