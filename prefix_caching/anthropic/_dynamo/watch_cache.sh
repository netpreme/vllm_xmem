#!/usr/bin/env bash
set -euo pipefail

# Watch GPU + CPU KV cache metrics in real time.
#
# Two data sources:
#   1. Dynamo /metrics endpoint (http://localhost:8000/metrics)
#      - dynamo_frontend_input_sequence_tokens_sum  → total input tokens (cumulative)
#      - dynamo_frontend_cached_tokens_sum          → GPU+CPU cache hits (cumulative)
#      - dynamo_frontend_model_total_kv_blocks      → GPU block count
#
#   2. /tmp/dynamo_worker.log (vLLM worker log — not proxied by Dynamo)
#      loggers.log lines → rolling 1000-request window hit rates:
#        "Prefix cache hit rate: X%"          → GPU cache (local)
#        "External prefix cache hit rate: X%" → CPU/connector cache
#        "GPU KV cache usage: X%"
#      metrics.log lines → per-interval CPU offload bytes:
#        "KV Transfer metrics: GPU_to_CPU_total_bytes=X, ..."
#
# Why two sources?
#   vllm:kv_offload_total_bytes is never proxied through the Dynamo frontend.
#   vllm:prefix_cache_queries/_hits are also absent from the Dynamo endpoint.
#   The worker log has both, so we parse it directly.
#
# Usage:
#   ./watch_cache.sh [interval_seconds] [worker_log_path]
#   ./watch_cache.sh 2                           # default log path
#   ./watch_cache.sh 2 /tmp/dynamo_worker.log

API_BASE="${API_BASE:-http://127.0.0.1:8000}"
INTERVAL="${1:-2}"
WORKER_LOG="${2:-/tmp/dynamo_worker.log}"

echo "Watching GPU + CPU KV cache (every ${INTERVAL}s)"
echo "  metrics endpoint : $API_BASE/metrics"
echo "  worker log       : $WORKER_LOG"
echo "──────────────────────────────────────────────────────────────"

python3 - "$API_BASE" "$INTERVAL" "$WORKER_LOG" <<'PYEOF'
import sys
import time
import re
import urllib.request

api_base  = sys.argv[1]
interval  = float(sys.argv[2])
log_path  = sys.argv[3]
url       = f"{api_base}/metrics"

# ── helpers ────────────────────────────────────────────────────────────

def fetch_metrics():
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return r.read().decode()
    except Exception:
        return ""

def parse_gauge(text, name):
    m = re.search(
        r'^' + re.escape(name) + r'(?:\{[^}]*\})?\s+([\d.e+\-]+)',
        text, re.MULTILINE
    )
    return float(m.group(1)) if m else None

def read_log_since(path, offset):
    """Return (new_text, new_offset). Handles rotation/truncation."""
    try:
        with open(path, 'r', errors='replace') as f:
            f.seek(0, 2)
            eof = f.tell()
            if offset > eof:   # file was truncated / rotated
                offset = 0
            f.seek(offset)
            text = f.read()
            return text, f.tell()
    except FileNotFoundError:
        return "", offset

# loggers.log pattern:
# "... GPU KV cache usage: 8.5%, Prefix cache hit rate: 74.9%, External prefix cache hit rate: 0.0%"
_LOG_STATS = re.compile(
    r'GPU KV cache usage:\s*([\d.]+)%.*?'
    r'Prefix cache hit rate:\s*([\d.]+)%.*?'
    r'External prefix cache hit rate:\s*([\d.]+)%'
)
# metrics.log pattern:
# "KV Transfer metrics: GPU_to_CPU_total_bytes=37748736, GPU_to_CPU_total_time=0.001..."
_LOG_XFER = re.compile(r'KV Transfer metrics:\s+(.+?)(?:\n|$)')

def parse_log(text):
    """
    Returns:
      latest_stats : dict  — most-recent loggers.log snapshot (rolling window)
      delta_xfer   : dict  — sum of ALL kv-transfer entries in text (per-interval → accumulate)
    """
    latest = {}
    for m in _LOG_STATS.finditer(text):
        latest['gpu_usage_pct']    = float(m.group(1))
        latest['prefix_hit_rate']  = float(m.group(2))
        latest['ext_hit_rate']     = float(m.group(3))

    delta = {}
    for m in _LOG_XFER.finditer(text):
        for kv in m.group(1).split(','):
            kv = kv.strip()
            if '=' in kv:
                k, v = kv.split('=', 1)
                try:
                    delta[k.strip()] = delta.get(k.strip(), 0.0) + float(v.strip())
                except ValueError:
                    pass
    return latest, delta

def fmt_bytes(b):
    if b >= 1e9:  return f"{b/1e9:.2f} GB"
    if b >= 1e6:  return f"{b/1e6:.1f} MB"
    if b >= 1e3:  return f"{b/1e3:.1f} KB"
    return f"{b:.0f} B"

def fmt_rate(b, t):
    return f"@ {fmt_bytes(b/t)}/s" if t > 0 else ""

# ── state ──────────────────────────────────────────────────────────────

prev_input  = 0.0
prev_cached = 0.0

# Cumulative CPU offload bytes (summed from per-interval log lines)
cum_g2c_bytes = 0.0
cum_c2g_bytes = 0.0
cum_g2c_time  = 0.0
cum_c2g_time  = 0.0

# Start log offset at 0 to capture full history since server start
log_offset = 0
first_poll = True

# Read full log history on startup
init_text, log_offset = read_log_since(log_path, 0)
if init_text:
    _, delta = parse_log(init_text)
    cum_g2c_bytes += delta.get('GPU_to_CPU_total_bytes', 0.0)
    cum_c2g_bytes += delta.get('CPU_to_GPU_total_bytes', 0.0)
    cum_g2c_time  += delta.get('GPU_to_CPU_total_time', 0.0)
    cum_c2g_time  += delta.get('CPU_to_GPU_total_time', 0.0)

# ── main loop ──────────────────────────────────────────────────────────

while True:
    # 1. Dynamo /metrics endpoint
    raw = fetch_metrics()
    if not raw:
        print("  (waiting for server...)")
        time.sleep(interval)
        continue

    input_tokens  = parse_gauge(raw, "dynamo_frontend_input_sequence_tokens_sum") or 0.0
    cached_tokens = parse_gauge(raw, "dynamo_frontend_cached_tokens_sum")         or 0.0
    gpu_blocks    = parse_gauge(raw, "dynamo_frontend_model_total_kv_blocks")

    # 2. Worker log — new lines since last poll
    new_text, log_offset = read_log_since(log_path, log_offset)
    log_stats, delta_xfer = parse_log(new_text) if new_text else ({}, {})

    cum_g2c_bytes += delta_xfer.get('GPU_to_CPU_total_bytes', 0.0)
    cum_c2g_bytes += delta_xfer.get('CPU_to_GPU_total_bytes', 0.0)
    cum_g2c_time  += delta_xfer.get('GPU_to_CPU_total_time',  0.0)
    cum_c2g_time  += delta_xfer.get('CPU_to_GPU_total_time',  0.0)

    # Skip if nothing changed
    if (input_tokens == prev_input and
        not delta_xfer and
        not log_stats and
        not first_poll):
        time.sleep(interval)
        continue

    first_poll = False

    # Deltas (this turn)
    d_input  = input_tokens  - prev_input
    d_cached = cached_tokens - prev_cached
    d_miss   = max(d_input - d_cached, 0.0)

    # Cumulative hit rate (from Dynamo — all-time)
    cum_hit = 100 * cached_tokens / input_tokens if input_tokens > 0 else 0.0

    # This-turn hit rate
    turn_hit = 100 * d_cached / d_input if d_input > 0 else 0.0

    print()
    print(f"  ┌─ This Turn ───────────────────────────────────────────────")
    print(f"  │ Input tokens:          {d_input:.0f}")
    print(f"  │ Cache hits (GPU+CPU):  {d_cached:.0f} tokens  ({turn_hit:.1f}%)")
    print(f"  │ Computed (miss):       {d_miss:.0f} tokens")
    print(f"  ├─ Cumulative — all-time (Dynamo counters) ─────────────────")
    print(f"  │ Total input tokens:    {input_tokens:.0f}")
    print(f"  │ Total cache hits:      {cached_tokens:.0f}  ({cum_hit:.1f}%)")

    # Rolling window stats from vLLM log (last ~1000 requests)
    if log_stats:
        gpu_r  = log_stats.get('prefix_hit_rate', None)
        ext_r  = log_stats.get('ext_hit_rate', None)
        gpu_u  = log_stats.get('gpu_usage_pct', None)
        print(f"  ├─ Rolling window (last ~1000 reqs, from vLLM log) ─────────")
        if gpu_r  is not None: print(f"  │ GPU prefix hit rate:   {gpu_r:.1f}%")
        if ext_r  is not None: print(f"  │ CPU/ext hit rate:      {ext_r:.1f}%")
        if gpu_u  is not None: print(f"  │ GPU KV cache usage:    {gpu_u:.1f}%")
    elif gpu_blocks is not None:
        print(f"  ├─ GPU KV Cache ────────────────────────────────────────────")
        print(f"  │ Total GPU blocks:      {int(gpu_blocks)} (16 tokens/block, "
              f"{int(gpu_blocks)*16} tok capacity)")

    print(f"  ├─ CPU Offload Activity (cumulative since server start) ────")
    if cum_g2c_bytes == 0.0 and cum_c2g_bytes == 0.0:
        print(f"  │ (no KV Transfer metrics in log yet)")
    else:
        g2c_str = fmt_bytes(cum_g2c_bytes)
        c2g_str = fmt_bytes(cum_c2g_bytes)
        g2c_rate = fmt_rate(delta_xfer.get('GPU_to_CPU_total_bytes', 0),
                            delta_xfer.get('GPU_to_CPU_total_time',  0))
        c2g_rate = fmt_rate(delta_xfer.get('CPU_to_GPU_total_bytes', 0),
                            delta_xfer.get('CPU_to_GPU_total_time',  0))
        print(f"  │ GPU→CPU total:         {g2c_str}  {g2c_rate}")
        print(f"  │ CPU→GPU total:         {c2g_str}  {c2g_rate}")
        if delta_xfer.get('GPU_to_CPU_total_bytes', 0) > 0:
            print(f"  │ This interval GPU→CPU: {fmt_bytes(delta_xfer['GPU_to_CPU_total_bytes'])}")
        if delta_xfer.get('CPU_to_GPU_total_bytes', 0) > 0:
            print(f"  │ This interval CPU→GPU: {fmt_bytes(delta_xfer['CPU_to_GPU_total_bytes'])}")
    print(f"  └──────────────────────────────────────────────────────────")

    prev_input  = input_tokens
    prev_cached = cached_tokens
    time.sleep(interval)

PYEOF
