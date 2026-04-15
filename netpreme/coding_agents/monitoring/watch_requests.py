#!/usr/bin/env python3
"""
Per-request metrics watcher for Dynamo + vLLM (single user).

Polls Prometheus every 0.5 s. When a new request completes it prints one row.
All latency and cache metrics come from vLLM histogram counters via
delta(sum)/delta(count), giving the exact per-turn value (not a rolling average).

Columns:
  ISL   — input sequence length (tokens), from Dynamo _last_ gauge
  OSL   — avg output tokens this turn, from Dynamo router delta
  TTFT  — time to first token (ms), vLLM exact per-turn
  ITL   — avg inter-token latency (ms) across decode steps, vLLM exact per-turn
  E2E   — end-to-end latency (ms), vLLM exact per-turn
  GPU$  — GPU (HBM) prefix cache hit rate this turn, vLLM delta
  CPU$  — CPU/MTier prefix cache hit rate this turn, vLLM delta
  gCum  — cumulative GPU hit rate since startup
  cCum  — cumulative CPU/MTier hit rate since startup
  G→C   — GPU→CPU bytes offloaded this turn, from kv_exporter delta
  C→G   — CPU→GPU bytes recalled this turn, from kv_exporter delta

On Ctrl+C saves a 2x3 matplotlib chart to ./session_metrics.png.

Usage:
    python3 watch_requests.py [--prom http://localhost:9090] [--interval 0.5]
"""

import argparse
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests

# ── optional matplotlib ───────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── data ──────────────────────────────────────────────────────────────────────
@dataclass
class Turn:
    index: int
    ts: str
    isl: Optional[float]        # input sequence length (tokens)
    osl: Optional[float]        # output sequence length (tokens)
    ttft_ms: Optional[float]    # time to first token (ms) — vLLM exact per-turn
    itl_ms: Optional[float]     # avg inter-token latency (ms) — vLLM exact per-turn
    e2e_ms: Optional[float]     # end-to-end latency (ms) — vLLM exact per-turn
    gpu_hit: Optional[float]    # per-turn GPU (HBM) cache hit rate 0-1
    cpu_hit: Optional[float]    # per-turn CPU/MTier cache hit rate 0-1
    gpu_cumul: Optional[float]  # cumulative GPU cache hit rate 0-1
    cpu_cumul: Optional[float]  # cumulative CPU/MTier cache hit rate 0-1
    g2c_bytes: Optional[float]  # GPU→CPU bytes offloaded this turn
    c2g_bytes: Optional[float]  # CPU→GPU bytes recalled this turn


# ── prometheus helpers ────────────────────────────────────────────────────────
def prom_scalar(prom: str, q: str) -> Optional[float]:
    try:
        r = requests.get(f"{prom}/api/v1/query", params={"query": q}, timeout=2)
        results = r.json()["data"]["result"]
        if results:
            v = results[0]["value"][1]
            return None if v in ("NaN", "Inf", "-Inf") else float(v)
    except Exception:
        pass
    return None


def fmt_f(v: Optional[float], decimals: int = 1, suffix: str = "") -> str:
    return f"{v:.{decimals}f}{suffix}" if v is not None else "-"


def fmt_bytes(v: Optional[float]) -> str:
    """Human-readable byte count: 0, 456B, 12K, 3.4M, 1.2G"""
    if v is None or v == 0:
        return "0"
    if v >= 1e9:
        return f"{v/1e9:.1f}G"
    if v >= 1e6:
        return f"{v/1e6:.1f}M"
    if v >= 1e3:
        return f"{v/1e3:.0f}K"
    return f"{v:.0f}B"


def _delta_ms(new_sum, new_cnt, old_sum, old_cnt) -> Optional[float]:
    """Return delta(sum)/delta(count) in ms, or None if data is missing."""
    if None in (new_sum, new_cnt, old_sum, old_cnt):
        return None
    d_cnt = new_cnt - old_cnt
    if d_cnt <= 0:
        return None
    return (new_sum - old_sum) / d_cnt * 1000


# ── chart ─────────────────────────────────────────────────────────────────────
def save_chart(turns: list[Turn], path: str) -> None:
    if not HAS_MPL:
        print("matplotlib not installed — skipping chart")
        return
    if not turns:
        print("No data to chart")
        return

    idx       = [t.index for t in turns]
    isl       = [t.isl      or 0 for t in turns]
    osl       = [t.osl      or 0 for t in turns]
    ttft      = [t.ttft_ms  or 0 for t in turns]
    e2e       = [t.e2e_ms   or 0 for t in turns]
    gpu_hit   = [(t.gpu_hit  or 0) * 100 for t in turns]
    cpu_hit   = [(t.cpu_hit  or 0) * 100 for t in turns]
    gpu_cumul = [(t.gpu_cumul or 0) * 100 for t in turns]
    cpu_cumul = [(t.cpu_cumul or 0) * 100 for t in turns]
    g2c_mb    = [(t.g2c_bytes or 0) / 1e6 for t in turns]
    c2g_mb    = [(t.c2g_bytes or 0) / 1e6 for t in turns]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle("Session Metrics — Dynamo + vLLM", fontsize=14)

    def bar(ax, y, label, color, unit):
        ax.bar(idx, y, color=color, alpha=0.8)
        ax.set_title(label)
        ax.set_xlabel("Request #")
        ax.set_ylabel(unit)
        ax.set_xticks(idx)
        for i, v in zip(idx, y):
            if v:
                ax.text(i, v * 1.02, f"{v:.0f}", ha="center", fontsize=8)

    bar(axes[0][0], isl, "Input Sequence Length",  "#4e79a7", "tokens")
    bar(axes[0][1], osl, "Output Sequence Length", "#f28e2b", "tokens")

    # TTFT vs E2E — both on one axes so the decode cost (E2E - TTFT) is visible
    ax = axes[0][2]
    ax.plot(idx, ttft, marker="o", color="#e15759", linewidth=2, label="TTFT")
    ax.plot(idx, e2e,  marker="s", color="#b07aa1", linewidth=2, label="E2E")
    ax.set_title("Latency per Turn (vLLM exact)")
    ax.set_xlabel("Request #")
    ax.set_ylabel("ms")
    ax.set_xticks(idx)
    ax.legend(fontsize=8)

    # Cache hit rates — GPU vs CPU/MTier
    ax = axes[1][0]
    ax.plot(idx, gpu_hit,   marker="o", color="#4e79a7", linewidth=1.5,
            alpha=0.6, label="GPU per-turn")
    ax.plot(idx, gpu_cumul, marker="s", color="#4e79a7", linewidth=2,
            label="GPU cumul")
    ax.plot(idx, cpu_hit,   marker="o", color="#f28e2b", linewidth=1.5,
            alpha=0.6, label="CPU per-turn")
    ax.plot(idx, cpu_cumul, marker="s", color="#f28e2b", linewidth=2,
            label="CPU cumul")
    ax.fill_between(idx, gpu_cumul, alpha=0.10, color="#4e79a7")
    ax.fill_between(idx, cpu_cumul, alpha=0.10, color="#f28e2b")
    ax.set_ylim(0, 105)
    ax.set_title("Prefix Cache Hit Rate — GPU vs CPU/MTier")
    ax.set_xlabel("Request #")
    ax.set_ylabel("%")
    ax.set_xticks(idx)
    ax.legend(fontsize=8)

    bar(axes[1][1], g2c_mb, "GPU→CPU Offload (MB)", "#59a14f", "MB")
    bar(axes[1][2], c2g_mb, "CPU→GPU Recall (MB)",  "#76b7b2", "MB")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"\nChart saved → {path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prom",     default="http://localhost:9090")
    ap.add_argument("--interval", type=float, default=0.5,
                    help="Poll interval in seconds (default 0.5)")
    ap.add_argument("--chart",    default="session_metrics.png")
    args = ap.parse_args()

    prom = args.prom
    turns: list[Turn] = []

    # ── column header ─────────────────────────────────────────────────────────
    hdr = (f"{'#':>4}  {'time':>8}  {'ISL':>7}  {'OSL':>7}"
           f"  {'TTFT':>8}  {'ITL':>8}  {'E2E':>8}"
           f"  {'GPU$':>7}  {'CPU$':>7}"
           f"  {'gCum':>7}  {'cCum':>7}"
           f"  {'G→C':>8}  {'C→G':>8}")
    sep = "─" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    # ── state — all counters tracked for delta computation ────────────────────
    prev_total       = None
    prev_out_sum     = None
    prev_out_count   = None
    # vLLM latency histograms — exact per-turn via delta(sum)/delta(count)
    prev_ttft_sum    = None
    prev_ttft_count  = None
    prev_itl_sum     = None
    prev_itl_count   = None
    prev_e2e_sum     = None
    prev_e2e_count   = None
    # GPU-local prefix cache (blocks)
    prev_hits        = None
    prev_hit_queries = None
    # CPU/MTier external prefix cache (tokens)
    prev_ext_hits    = None
    prev_ext_queries = None
    # KV offload byte counters (from kv_exporter on port 9091)
    prev_g2c_bytes   = None
    prev_c2g_bytes   = None
    turn_idx         = 0

    def snapshot(q):
        return prom_scalar(prom, q)

    def on_exit(*_):
        print(f"\n{sep}")
        print(f"  {len(turns)} requests recorded")
        if turns:
            last = turns[-1]
            if last.gpu_cumul is not None:
                print(f"  cumulative GPU  cache hit rate : {last.gpu_cumul*100:.1f}%")
            if last.cpu_cumul is not None:
                print(f"  cumulative CPU  cache hit rate : {last.cpu_cumul*100:.1f}%")
        save_chart(turns, args.chart)
        sys.exit(0)

    signal.signal(signal.SIGINT,  on_exit)
    signal.signal(signal.SIGTERM, on_exit)

    print("Watching Prometheus for completed requests… (Ctrl+C to stop & save chart)\n")

    while True:
        total = snapshot("sum(dynamo_frontend_requests_total)")

        if total is not None:
            if prev_total is None:
                # First sample — seed all baselines, emit nothing
                prev_total       = total
                prev_out_sum     = snapshot("sum(dynamo_component_router_output_sequence_tokens_sum)")
                prev_out_count   = snapshot("sum(dynamo_component_router_output_sequence_tokens_count)")
                prev_ttft_sum    = snapshot("sum(vllm:time_to_first_token_seconds_sum)")
                prev_ttft_count  = snapshot("sum(vllm:time_to_first_token_seconds_count)")
                prev_itl_sum     = snapshot("sum(vllm:inter_token_latency_seconds_sum)")
                prev_itl_count   = snapshot("sum(vllm:inter_token_latency_seconds_count)")
                prev_e2e_sum     = snapshot("sum(vllm:e2e_request_latency_seconds_sum)")
                prev_e2e_count   = snapshot("sum(vllm:e2e_request_latency_seconds_count)")
                prev_hits        = snapshot("sum(vllm:prefix_cache_hits_total)")
                prev_hit_queries = snapshot("sum(vllm:prefix_cache_queries_total)")
                prev_ext_hits    = snapshot("sum(vllm:external_prefix_cache_hits_total)")
                prev_ext_queries = snapshot("sum(vllm:external_prefix_cache_queries_total)")
                prev_g2c_bytes   = snapshot("sum(vllm_kv_gpu_to_cpu_bytes_total)")
                prev_c2g_bytes   = snapshot("sum(vllm_kv_cpu_to_gpu_bytes_total)")
                if total > 0:
                    print(f"[seeded at request #{int(total)} — earlier requests not shown]")

            elif total > prev_total:
                new_requests = int(total - prev_total)

                # ── fetch all batch-level counters once ───────────────────────
                out_sum     = snapshot("sum(dynamo_component_router_output_sequence_tokens_sum)")
                out_count   = snapshot("sum(dynamo_component_router_output_sequence_tokens_count)")
                ttft_sum    = snapshot("sum(vllm:time_to_first_token_seconds_sum)")
                ttft_count  = snapshot("sum(vllm:time_to_first_token_seconds_count)")
                itl_sum     = snapshot("sum(vllm:inter_token_latency_seconds_sum)")
                itl_count   = snapshot("sum(vllm:inter_token_latency_seconds_count)")
                e2e_sum     = snapshot("sum(vllm:e2e_request_latency_seconds_sum)")
                e2e_count   = snapshot("sum(vllm:e2e_request_latency_seconds_count)")
                hits        = snapshot("sum(vllm:prefix_cache_hits_total)")
                hit_q       = snapshot("sum(vllm:prefix_cache_queries_total)")
                ext_hits    = snapshot("sum(vllm:external_prefix_cache_hits_total)")
                ext_q       = snapshot("sum(vllm:external_prefix_cache_queries_total)")
                g2c_bytes   = snapshot("sum(vllm_kv_gpu_to_cpu_bytes_total)")
                c2g_bytes   = snapshot("sum(vllm_kv_cpu_to_gpu_bytes_total)")

                # ── OSL: avg output tokens this turn ──────────────────────────
                # dynamo_frontend_output_sequence_tokens_sum is always 0 (Dynamo
                # bug); dynamo_component_router_output_sequence_tokens_sum is correct.
                osl = None
                if None not in (out_sum, out_count, prev_out_sum, prev_out_count):
                    d_cnt = out_count - prev_out_count
                    if d_cnt > 0:
                        osl = (out_sum - prev_out_sum) / d_cnt

                # ── latency: exact per-turn via delta(sum)/delta(count) ───────
                ttft_ms = _delta_ms(ttft_sum, ttft_count, prev_ttft_sum, prev_ttft_count)
                itl_ms  = _delta_ms(itl_sum,  itl_count,  prev_itl_sum,  prev_itl_count)
                e2e_ms  = _delta_ms(e2e_sum,  e2e_count,  prev_e2e_sum,  prev_e2e_count)

                # ── GPU (HBM) prefix cache hit rate this turn ─────────────────
                gpu_hit = None
                if None not in (hits, hit_q, prev_hits, prev_hit_queries):
                    d_q = hit_q - prev_hit_queries
                    if d_q > 0:
                        gpu_hit = (hits - prev_hits) / d_q
                gpu_cumul = hits / hit_q if (hits is not None and hit_q and hit_q > 0) else None

                # ── CPU/MTier external prefix cache hit rate this turn ─────────
                cpu_hit = None
                if None not in (ext_hits, ext_q, prev_ext_hits, prev_ext_queries):
                    d_ext_q = ext_q - prev_ext_queries
                    if d_ext_q > 0:
                        cpu_hit = (ext_hits - prev_ext_hits) / d_ext_q
                cpu_cumul = (
                    ext_hits / ext_q
                    if (ext_hits is not None and ext_q and ext_q > 0)
                    else None
                )

                # ── KV transfer bytes this turn ───────────────────────────────
                d_g2c = (
                    max(0.0, g2c_bytes - prev_g2c_bytes)
                    if None not in (g2c_bytes, prev_g2c_bytes) else None
                )
                d_c2g = (
                    max(0.0, c2g_bytes - prev_c2g_bytes)
                    if None not in (c2g_bytes, prev_c2g_bytes) else None
                )

                # ── one row per request in the batch ─────────────────────────
                # ISL still uses the Dynamo _last_ gauge — no vLLM equivalent.
                # All other per-turn values are already computed above from deltas.
                # If new_requests > 1 the ISL _last_ gauge is stale for earlier
                # requests, but latency/cache/byte columns are averaged across
                # the batch (sequential single-user means this rarely happens).
                for _ in range(new_requests):
                    turn_idx += 1
                    isl = snapshot("dynamo_frontend_worker_last_input_sequence_tokens")

                    ts = datetime.now().strftime("%H:%M:%S")
                    t  = Turn(
                        turn_idx, ts, isl, osl,
                        ttft_ms, itl_ms, e2e_ms,
                        gpu_hit, cpu_hit, gpu_cumul, cpu_cumul,
                        d_g2c, d_c2g,
                    )
                    turns.append(t)

                    print(
                        f"{turn_idx:>4}  {ts:>8}"
                        f"  {fmt_f(isl, 0):>7}"
                        f"  {fmt_f(osl, 0):>7}"
                        f"  {fmt_f(ttft_ms, 1, 'ms'):>8}"
                        f"  {fmt_f(itl_ms,  2, 'ms'):>8}"
                        f"  {fmt_f(e2e_ms,  1, 'ms'):>8}"
                        f"  {fmt_f(None if gpu_hit  is None else gpu_hit  * 100, 1, '%'):>7}"
                        f"  {fmt_f(None if cpu_hit  is None else cpu_hit  * 100, 1, '%'):>7}"
                        f"  {fmt_f(None if gpu_cumul is None else gpu_cumul * 100, 1, '%'):>7}"
                        f"  {fmt_f(None if cpu_cumul is None else cpu_cumul * 100, 1, '%'):>7}"
                        f"  {fmt_bytes(t.g2c_bytes):>8}"
                        f"  {fmt_bytes(t.c2g_bytes):>8}"
                    )

                # ── advance all state for next turn's delta ───────────────────
                prev_total       = total
                prev_out_sum     = out_sum    if out_sum    is not None else prev_out_sum
                prev_out_count   = out_count  if out_count  is not None else prev_out_count
                prev_ttft_sum    = ttft_sum   if ttft_sum   is not None else prev_ttft_sum
                prev_ttft_count  = ttft_count if ttft_count is not None else prev_ttft_count
                prev_itl_sum     = itl_sum    if itl_sum    is not None else prev_itl_sum
                prev_itl_count   = itl_count  if itl_count  is not None else prev_itl_count
                prev_e2e_sum     = e2e_sum    if e2e_sum    is not None else prev_e2e_sum
                prev_e2e_count   = e2e_count  if e2e_count  is not None else prev_e2e_count
                prev_hits        = hits       if hits       is not None else prev_hits
                prev_hit_queries = hit_q      if hit_q      is not None else prev_hit_queries
                prev_ext_hits    = ext_hits   if ext_hits   is not None else prev_ext_hits
                prev_ext_queries = ext_q      if ext_q      is not None else prev_ext_queries
                prev_g2c_bytes   = g2c_bytes  if g2c_bytes  is not None else prev_g2c_bytes
                prev_c2g_bytes   = c2g_bytes  if c2g_bytes  is not None else prev_c2g_bytes

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
