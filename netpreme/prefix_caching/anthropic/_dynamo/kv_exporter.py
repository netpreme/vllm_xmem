#!/usr/bin/env python3
"""
Prometheus exporter for vLLM KV transfer metrics.

Tails the dynamo worker log and exposes KV offload metrics
(bytes transferred CPU↔GPU, request TTFT) on port 9091.

Usage:
    python3 kv_exporter.py [--log /tmp/dynamo_worker_8000.log] [--port 9091]
"""
import argparse
import glob
import re
import time
from threading import Thread

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# ── Prometheus metrics ────────────────────────────────────────────
# KV transfer byte totals (from "KV Transfer metrics:" log lines)
kv_gpu_to_cpu_bytes = Counter(
    "vllm_kv_gpu_to_cpu_bytes_total",
    "Total bytes transferred GPU→CPU (offload to CPU/MTier tier)",
)
kv_cpu_to_gpu_bytes = Counter(
    "vllm_kv_cpu_to_gpu_bytes_total",
    "Total bytes transferred CPU→GPU (recall from CPU/MTier tier)",
)
kv_num_offloaded_blocks = Gauge(
    "vllm_kv_offloaded_blocks",
    "Number of KV blocks currently stored in CPU/MTier tier",
)
kv_num_loaded_blocks = Gauge(
    "vllm_kv_loaded_blocks",
    "Number of KV blocks loaded back to GPU in the last metric interval",
)

# Derived bandwidth gauges (computed from delta over last interval)
kv_gpu_to_cpu_bw_gbps = Gauge(
    "vllm_kv_gpu_to_cpu_bandwidth_gbps",
    "Recent GPU→CPU offload bandwidth in GB/s",
)
kv_cpu_to_gpu_bw_gbps = Gauge(
    "vllm_kv_cpu_to_gpu_bandwidth_gbps",
    "Recent CPU→GPU recall bandwidth in GB/s",
)

# TTFT histogram observed from client-side (if populated via --ttft-log)
ttft_histogram = Histogram(
    "vllm_request_ttft_seconds",
    "Time to first token (seconds) observed by stress test",
    buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0],
)

# Log parse health
log_parse_errors = Counter(
    "vllm_kv_exporter_parse_errors_total",
    "Number of log lines that failed to parse",
)
log_bytes_read = Counter(
    "vllm_kv_exporter_log_bytes_read_total",
    "Total bytes read from the worker log",
)

# ── Log pattern ───────────────────────────────────────────────────
KV_PATTERN = re.compile(r"KV Transfer metrics:\s+(.+?)(?:\n|$)")
KV_KV_PATTERN = re.compile(r"(\w+)\s*=\s*([0-9.eE+\-]+)")


def parse_kv_line(line: str) -> dict[str, float]:
    result = {}
    for k, v in KV_KV_PATTERN.findall(line):
        try:
            result[k] = float(v)
        except ValueError:
            pass
    return result


def tail_log(log_path: str, poll_interval: float = 2.0):
    """
    Tail the worker log, parse KV Transfer metrics lines,
    and update Prometheus metrics incrementally.
    """
    offset = 0
    prev_g2c = 0.0
    prev_c2g = 0.0
    prev_ts = time.time()

    print(f"[kv_exporter] Tailing {log_path} ...")

    while True:
        try:
            with open(log_path, "r", errors="replace") as f:
                f.seek(0, 2)
                eof = f.tell()
                if offset > eof:
                    # log was rotated / truncated
                    offset = 0
                f.seek(offset)
                text = f.read()
                offset = f.tell()

            log_bytes_read.inc(len(text))

            for m in KV_PATTERN.finditer(text):
                kv = parse_kv_line(m.group(1))
                if not kv:
                    log_parse_errors.inc()
                    continue

                g2c = kv.get("GPU_to_CPU_total_bytes", 0.0)
                c2g = kv.get("CPU_to_GPU_total_bytes", 0.0)

                # Increment counters by delta (log values are cumulative totals)
                delta_g2c = max(0.0, g2c - prev_g2c)
                delta_c2g = max(0.0, c2g - prev_c2g)

                if delta_g2c > 0:
                    kv_gpu_to_cpu_bytes.inc(delta_g2c)
                if delta_c2g > 0:
                    kv_cpu_to_gpu_bytes.inc(delta_c2g)

                # Bandwidth: bytes/sec → GB/s
                now = time.time()
                dt = max(0.1, now - prev_ts)
                kv_gpu_to_cpu_bw_gbps.set(delta_g2c / dt / 1e9)
                kv_cpu_to_gpu_bw_gbps.set(delta_c2g / dt / 1e9)

                prev_g2c = g2c
                prev_c2g = c2g
                prev_ts = now

                # Optional block counts
                if "num_offloaded_blocks" in kv:
                    kv_num_offloaded_blocks.set(kv["num_offloaded_blocks"])
                if "num_loaded_blocks" in kv:
                    kv_num_loaded_blocks.set(kv["num_loaded_blocks"])

        except FileNotFoundError:
            pass  # log not created yet, keep waiting
        except Exception as e:
            print(f"[kv_exporter] Error: {e}")
            log_parse_errors.inc()

        time.sleep(poll_interval)


def watch_multiple_logs(pattern: str, poll_interval: float = 2.0):
    """Watch all logs matching a glob pattern (e.g. /tmp/dynamo_worker_*.log)."""
    watched: set[str] = set()
    while True:
        for path in glob.glob(pattern):
            if path not in watched:
                print(f"[kv_exporter] Discovered log: {path}")
                t = Thread(target=tail_log, args=(path, poll_interval), daemon=True)
                t.start()
                watched.add(path)
        time.sleep(5.0)


def main():
    parser = argparse.ArgumentParser(description="vLLM KV offload Prometheus exporter")
    parser.add_argument(
        "--log",
        default="/tmp/dynamo_worker_8000.log",
        help="Worker log path or glob (e.g. /tmp/dynamo_worker_*.log)",
    )
    parser.add_argument("--port", type=int, default=9091, help="Prometheus metrics port")
    parser.add_argument("--poll", type=float, default=2.0, help="Log poll interval (seconds)")
    args = parser.parse_args()

    print(f"[kv_exporter] Starting Prometheus exporter on :{args.port}")
    start_http_server(args.port)

    if "*" in args.log or "?" in args.log:
        watch_multiple_logs(args.log, args.poll)
    else:
        tail_log(args.log, args.poll)


if __name__ == "__main__":
    main()
