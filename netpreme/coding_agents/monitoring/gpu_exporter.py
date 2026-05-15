#!/usr/bin/env python3
"""
Prometheus exporter for per-GPU utilization via `nvidia-smi`.

Polls nvidia-smi every --interval seconds and exposes one labeled time series
per GPU (label: gpu="0", "1", ...). Metrics:

  gpu_utilization_pct        — SM (compute) utilization, 0-100
  gpu_memory_utilization_pct — DRAM bandwidth utilization, 0-100
  gpu_memory_used_mib        — used HBM (MiB)
  gpu_memory_total_mib       — total HBM (MiB)
  gpu_memory_used_pct        — gpu_memory_used / gpu_memory_total * 100
  gpu_power_watts            — current board power draw
  gpu_temperature_c          — die temperature

Default port: 9092 (kv_exporter uses 9091).

Usage:
    python3 gpu_exporter.py [--port 9092] [--interval 1.0]
"""
import argparse
import subprocess
import time

from prometheus_client import Gauge, start_http_server

# ── metrics ───────────────────────────────────────────────────────────────────
g_util_sm   = Gauge("gpu_utilization_pct",        "SM utilization (%)",          ["gpu"])
g_util_mem  = Gauge("gpu_memory_utilization_pct", "Memory BW utilization (%)",   ["gpu"])
g_mem_used  = Gauge("gpu_memory_used_mib",        "Used HBM (MiB)",              ["gpu"])
g_mem_total = Gauge("gpu_memory_total_mib",       "Total HBM (MiB)",             ["gpu"])
g_mem_pct   = Gauge("gpu_memory_used_pct",        "Used HBM as % of total",      ["gpu"])
g_power     = Gauge("gpu_power_watts",            "Board power draw (W)",        ["gpu"])
g_temp      = Gauge("gpu_temperature_c",          "Die temperature (C)",         ["gpu"])


NVSMI_QUERY = (
    "index,utilization.gpu,utilization.memory,"
    "memory.used,memory.total,power.draw,temperature.gpu"
)


def poll_once() -> None:
    """One nvidia-smi snapshot → update all gauges."""
    out = subprocess.run(
        ["nvidia-smi", f"--query-gpu={NVSMI_QUERY}",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=5,
    )
    if out.returncode != 0:
        return
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 7:
            continue
        try:
            idx       = parts[0]
            sm_pct    = float(parts[1])
            mem_pct   = float(parts[2])
            mem_used  = float(parts[3])
            mem_total = float(parts[4])
            power_w   = float(parts[5]) if parts[5] not in ("[N/A]", "") else 0.0
            temp_c    = float(parts[6]) if parts[6] not in ("[N/A]", "") else 0.0
        except ValueError:
            continue
        g_util_sm  .labels(gpu=idx).set(sm_pct)
        g_util_mem .labels(gpu=idx).set(mem_pct)
        g_mem_used .labels(gpu=idx).set(mem_used)
        g_mem_total.labels(gpu=idx).set(mem_total)
        g_mem_pct  .labels(gpu=idx).set(mem_used / mem_total * 100 if mem_total else 0.0)
        g_power    .labels(gpu=idx).set(power_w)
        g_temp     .labels(gpu=idx).set(temp_c)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port",     type=int,   default=9092)
    ap.add_argument("--interval", type=float, default=1.0,
                    help="Seconds between nvidia-smi polls (default: 1.0)")
    args = ap.parse_args()

    start_http_server(args.port)
    print(f"GPU exporter listening on :{args.port}  interval={args.interval}s")
    while True:
        try:
            poll_once()
        except Exception as e:
            print(f"poll error: {e}")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
