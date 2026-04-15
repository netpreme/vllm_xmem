#!/usr/bin/env python3
"""
Per-request metrics watcher for vLLM (no Dynamo, no Prometheus required).

Scrapes the vLLM /metrics endpoint directly every 0.5 s.
When a new request completes it prints one row.
All latency and cache metrics come from vLLM histogram counters via
delta(sum)/delta(count), giving the exact per-turn value.

The G→C / C→G byte columns use vllm:kv_offload_total_bytes_total exposed natively
by the OffloadingConnector — no kv_exporter sidecar is needed.

Columns:
  ISL   — input prompt tokens (per-turn avg from histogram delta)
  OSL   — output generation tokens (per-turn avg from histogram delta)
  TTFT  — time to first token (ms), exact per-turn via histogram delta
  ITL   — avg inter-token latency (ms) across decode steps, per-turn
  E2E   — end-to-end latency (ms), per-turn
  GPU$  — GPU (HBM) prefix cache hit rate this turn
  CPU$  — CPU/MTier prefix cache hit rate this turn (OffloadingConnector)
  gCum  — cumulative GPU hit rate since startup
  cCum  — cumulative CPU/MTier hit rate since startup
  G→C   — GPU→CPU bytes offloaded this turn (label transfer_type="GPU_to_CPU")
  C→G   — CPU→GPU bytes recalled this turn  (label transfer_type="CPU_to_GPU")

On Ctrl+C saves a 2×4 matplotlib chart to ./session_metrics_vllm.png.

Usage:
    python3 watch_vllm.py [--vllm http://localhost:8000] [--interval 0.1]
    python3 watch_vllm.py --vllm http://localhost:9000 --interval 1.0
"""

import argparse
import os
import re
import signal
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests

# ── optional matplotlib ────────────────────────────────────────────────────────
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ── data ───────────────────────────────────────────────────────────────────────
@dataclass
class Turn:
    index: int
    ts: str
    isl: Optional[float]        # avg input prompt tokens this turn
    osl: Optional[float]        # avg output generation tokens this turn
    ttft_ms: Optional[float]    # time to first token (ms) — exact per-turn
    itl_ms: Optional[float]     # avg inter-token latency (ms) — exact per-turn
    e2e_ms: Optional[float]     # end-to-end latency (ms) — exact per-turn
    gpu_hit: Optional[float]    # per-turn GPU (HBM) cache hit rate 0-1
    cpu_hit: Optional[float]    # per-turn CPU/MTier cache hit rate 0-1
    gpu_cumul: Optional[float]  # cumulative GPU cache hit rate 0-1
    cpu_cumul: Optional[float]  # cumulative CPU/MTier cache hit rate 0-1
    g2c_bytes: Optional[float]  # GPU→CPU bytes offloaded this turn
    c2g_bytes: Optional[float]  # CPU→GPU bytes recalled this turn
    # E2E breakdown — from vllm:request_{queue,prefill,decode}_time_seconds
    # queue   = arrival → first scheduled   (waiting in scheduler)
    # prefill = scheduled → first token     (first forward pass)
    # decode  = first token → last token    (all decode steps)
    # queue + prefill + decode == e2e exactly (same timestamps, no overhead)
    queue_ms: Optional[float]   # time spent waiting in scheduler queue (ms)
    prefill_ms: Optional[float] # time for prefill phase (ms)
    decode_ms: Optional[float]  # time for decode phase (ms)


# ── Prometheus text format parser ─────────────────────────────────────────────
# Matches:  metric_name{label="val",...}  <value>
# or:       metric_name  <value>
_LINE_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)'  # metric name
    r'(?:\{([^}]*)\})?'              # optional labels block
    r'\s+([+-]?(?:[0-9]*\.)?[0-9]+(?:[eE][+-]?[0-9]+)?|NaN|[+-]?Inf)'
)
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')

# Scraped type: metric_name -> [(labels_dict, float_value), ...]
Scraped = dict[str, list[tuple[dict[str, str], float]]]


def scrape(url: str) -> Scraped:
    """
    Fetch Prometheus text metrics from <url>/metrics.
    Returns a dict of metric_name -> [(labels, value), ...].
    """
    result: Scraped = {}
    try:
        r = requests.get(f"{url}/metrics", timeout=2)
        for line in r.text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            m = _LINE_RE.match(line)
            if not m:
                continue
            name, labels_str, value_str = m.groups()
            if value_str in ("NaN", "Inf", "+Inf", "-Inf"):
                continue
            try:
                value = float(value_str)
            except ValueError:
                continue
            labels: dict[str, str] = {}
            if labels_str:
                for kv in _LABEL_RE.finditer(labels_str):
                    labels[kv.group(1)] = kv.group(2)
            result.setdefault(name, []).append((labels, value))
    except Exception:
        pass
    return result


def sum_metric(s: Scraped, name: str) -> Optional[float]:
    """Sum all time series for the given metric name."""
    series = s.get(name)
    if not series:
        return None
    return sum(v for _, v in series)


def get_labeled(s: Scraped, name: str, **filters) -> Optional[float]:
    """Sum series whose labels match all supplied key=value filters."""
    series = s.get(name)
    if not series:
        return None
    total = 0.0
    found = False
    for labels, value in series:
        if all(labels.get(k) == v for k, v in filters.items()):
            total += value
            found = True
    return total if found else None


# ── helpers ────────────────────────────────────────────────────────────────────
def _delta_ms(new_sum, new_cnt, old_sum, old_cnt) -> Optional[float]:
    """delta(sum)/delta(count) in ms; None if data missing or no new samples."""
    if None in (new_sum, new_cnt, old_sum, old_cnt):
        return None
    d_cnt = new_cnt - old_cnt
    if d_cnt <= 0:
        return None
    return (new_sum - old_sum) / d_cnt * 1000


def _delta_avg(new_sum, new_cnt, old_sum, old_cnt) -> Optional[float]:
    """delta(sum)/delta(count) as plain float (no ms scaling)."""
    if None in (new_sum, new_cnt, old_sum, old_cnt):
        return None
    d_cnt = new_cnt - old_cnt
    if d_cnt <= 0:
        return None
    return (new_sum - old_sum) / d_cnt


def fmt_f(v: Optional[float], decimals: int = 1, suffix: str = "") -> str:
    return f"{v:.{decimals}f}{suffix}" if v is not None else "-"


def fmt_bytes(v: Optional[float]) -> str:
    """Human-readable: 0, 456B, 12K, 3.4M, 1.2G"""
    if v is None or v == 0:
        return "0"
    if v >= 1e9:
        return f"{v/1e9:.1f}G"
    if v >= 1e6:
        return f"{v/1e6:.1f}M"
    if v >= 1e3:
        return f"{v/1e3:.0f}K"
    return f"{v:.0f}B"


# ── chart ──────────────────────────────────────────────────────────────────────
def save_chart(turns: list[Turn], path: str) -> None:
    if not HAS_MPL:
        print("matplotlib not installed — skipping chart")
        return
    if not turns:
        print("No data to chart")
        return

    idx        = [t.index for t in turns]
    isl        = [t.isl       or 0 for t in turns]
    osl        = [t.osl       or 0 for t in turns]
    ttft       = [t.ttft_ms   or 0 for t in turns]
    itl        = [t.itl_ms    or 0 for t in turns]
    e2e        = [t.e2e_ms    or 0 for t in turns]
    gpu_hit    = [(t.gpu_hit   or 0) * 100 for t in turns]
    cpu_hit    = [(t.cpu_hit   or 0) * 100 for t in turns]
    gpu_cumul  = [(t.gpu_cumul or 0) * 100 for t in turns]
    cpu_cumul  = [(t.cpu_cumul or 0) * 100 for t in turns]
    g2c_mb     = [(t.g2c_bytes or 0) / 1e6 for t in turns]
    c2g_mb     = [(t.c2g_bytes or 0) / 1e6 for t in turns]
    queue_ms   = [t.queue_ms   or 0 for t in turns]
    prefill_ms = [t.prefill_ms or 0 for t in turns]
    decode_ms  = [t.decode_ms  or 0 for t in turns]

    fig, axes = plt.subplots(2, 4, figsize=(26, 8))
    fig.suptitle("Session Metrics — vLLM (CPU Offload)", fontsize=14)

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

    # ── TTFT / ITL / E2E line chart with exact value labels ──────────────────
    ax = axes[0][2]
    # Left axis: TTFT and E2E (hundreds of ms)
    l1, = ax.plot(idx, ttft, marker="o", color="#e15759", linewidth=2, label="TTFT")
    l2, = ax.plot(idx, e2e,  marker="s", color="#b07aa1", linewidth=2, label="E2E")
    ax.set_title("Latency per Turn (vLLM exact)")
    ax.set_xlabel("Request #")
    ax.set_ylabel("TTFT / E2E (ms)")
    ax.set_xticks(idx)

    # Right axis: ITL (typically much smaller — single-digit to tens of ms)
    ax2 = ax.twinx()
    l3, = ax2.plot(idx, itl, marker="^", color="#59a14f", linewidth=1.5,
                   linestyle="--", label="ITL")
    ax2.set_ylabel("ITL (ms)", color="#59a14f")
    ax2.tick_params(axis="y", labelcolor="#59a14f")

    # Combined legend for both axes
    ax.legend(handles=[l2, l1, l3], fontsize=8)

    # E2E (purple) — above left axis markers, alternating 20/34 pt
    for i, (x, y) in enumerate(zip(idx, e2e)):
        pts = 20 if i % 2 == 0 else 34
        ax.annotate(f"{y:.0f}", (x, y),
                    textcoords="offset points", xytext=(0, pts),
                    ha="center", fontsize=7, color="#b07aa1",
                    arrowprops=None)
    # TTFT (red) — below left axis markers, alternating 20/34 pt
    for i, (x, y) in enumerate(zip(idx, ttft)):
        pts = 20 if i % 2 == 0 else 34
        ax.annotate(f"{y:.0f}", (x, y),
                    textcoords="offset points", xytext=(0, -pts),
                    ha="center", fontsize=7, color="#e15759",
                    arrowprops=None)
    # ITL (green) — annotated on right axis with value + decode token count
    # Format: "Xms × N" so reader can estimate decode_time = ITL × N
    for i, (x, y, n) in enumerate(zip(idx, itl, osl)):
        pts = 14 if i % 2 == 0 else 26
        decode_steps = max(0, int(round(n)) - 1)  # OSL - 1 ≈ number of decode steps
        ax2.annotate(f"{y:.1f}ms\n×{decode_steps}tok", (x, y),
                     textcoords="offset points", xytext=(8, pts),
                     ha="left", fontsize=6, color="#59a14f",
                     arrowprops=None)

    # ── E2E breakdown stacked bar ─────────────────────────────────────────────
    # vLLM uses TWO different clocks:
    #   arrival_time       — wall-clock (time.time()) set in the frontend
    #   queued/scheduled/first_token/last_token — monotonic (time.monotonic())
    #                        set inside the engine core process
    #
    # Because of this, queue + prefill + decode ≠ E2E.  The gap is real time
    # spent BEFORE the engine core received the request:
    #   - HTTP parsing & Anthropic→OpenAI conversion
    #   - Chat-template application and tokenization (Dynamo vllm processor)
    #   - IPC serialisation + transport from frontend to engine core
    #
    # We call this "Pre-engine".  By construction:
    #   Pre-engine + Queue + Prefill + Decode = E2E   (exactly)
    #
    # Sources:
    #   vllm:request_queue_time_seconds   — scheduled_ts − queued_ts  (monotonic)
    #   vllm:request_prefill_time_seconds — first_token_ts − scheduled_ts  (mono)
    #   vllm:request_decode_time_seconds  — last_token_ts − first_token_ts (mono)
    #   vllm:e2e_request_latency_seconds  — wall-clock arrival → last token
    pre_engine_ms = [
        max(0.0, e - q - p - d)
        for e, q, p, d in zip(e2e, queue_ms, prefill_ms, decode_ms)
    ]

    # Stack bottom→top: decode, prefill, queue, pre-engine
    # So the visual reads top→bottom: pre-engine, queue, prefill, decode
    # matching the legend order the user expects.
    ax = axes[0][3]
    bottom_pf = decode_ms
    bottom_q  = [d + p  for d, p  in zip(decode_ms,  prefill_ms)]
    bottom_pe = [d + p + q for d, p, q in zip(decode_ms, prefill_ms, queue_ms)]

    b_dc = ax.bar(idx, decode_ms,     color="#59a14f", label="Decode")
    b_pf = ax.bar(idx, prefill_ms,   bottom=bottom_pf, color="#f28e2b", label="Prefill")
    b_q  = ax.bar(idx, queue_ms,     bottom=bottom_q,  color="#4e79a7", label="Queue")
    b_pe = ax.bar(idx, pre_engine_ms, bottom=bottom_pe, color="#adb3ba", label="Pre-engine (tokenize+IPC)")

    ax.set_title("E2E Breakdown per Turn")
    ax.set_xlabel("Request #")
    ax.set_ylabel("ms")
    ax.set_xticks(idx)
    # Legend ordered top→bottom matching visual: pre-engine, queue, prefill, decode
    ax.legend(handles=[b_pe, b_q, b_pf, b_dc], fontsize=7)

    # Labels inside each segment, top→bottom: pre-engine, queue, prefill, decode.
    # Only shown when segment is ≥7% of total to avoid overlap on tiny segments.
    for xi, (pe, q, p, d) in enumerate(zip(pre_engine_ms, queue_ms, prefill_ms, decode_ms)):
        x   = idx[xi]
        # y-midpoints for each segment (bottom-to-top stacking: d, p, q, pe)
        mid_d  = d / 2
        mid_p  = d + p / 2
        mid_q  = d + p + q / 2
        mid_pe = d + p + q + pe / 2
        total  = d + p + q + pe

        def _seg_label(mid, val, dark=False):
            # Only draw inside label if segment is ≥7% of total (avoids overlap)
            if val > 0 and total > 0 and val / total >= 0.07:
                ax.text(x, mid, f"{val:.0f}",
                        ha="center", va="center", fontsize=6,
                        color="black" if dark else "white", fontweight="bold")

        _seg_label(mid_pe, pe, dark=True)   # pre-engine — gray bg, black text
        _seg_label(mid_q,  q)               # queue
        _seg_label(mid_p,  p)               # prefill
        _seg_label(mid_d,  d)               # decode

        # Color-coded component values + E2E stacked above each bar.
        # Top-down reading order: E2E → pre-engine → queue → prefill → decode
        # Alternate left/right x-offset for even/odd bars to prevent overlap.
        if total > 0:
            x_off = -8 if xi % 2 == 0 else 8
            line_h = 13  # pt between each label — enough white space
            y_base = 4   # pt gap between bar top and first label (decode)

            # Stack bottom-up so top-down reads: E2E → pre-engine → queue → prefill → decode
            above = [
                (d,  "#59a14f"),   # decode      (green)   — li=0, closest to bar
                (p,  "#f28e2b"),   # prefill     (orange)  — li=1
                (q,  "#4e79a7"),   # queue       (blue)    — li=2
                (pe, "#adb3ba"),   # pre-engine  (gray)    — li=3
            ]
            for li, (val, color) in enumerate(above):
                ax.annotate(f"{val:.0f}", (x, total),
                            textcoords="offset points",
                            xytext=(x_off, y_base + li * line_h),
                            ha="center", fontsize=6, color=color)
            # E2E at the very top — extra gap above pre-engine, bold black
            e2e_y = y_base + len(above) * line_h + 6
            ax.annotate(f"E2E: {total:.0f}ms", (x, total),
                        textcoords="offset points",
                        xytext=(x_off, e2e_y),
                        ha="center", fontsize=6.5, color="black",
                        fontweight="bold")

    # ── Cache hit rates — GPU vs CPU/MTier per-turn and cumulative ────────────
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
    ax.set_ylim(0, 140)   # extra headroom above 100% for annotation stacks
    ax.set_title("Prefix Cache Hit Rate — GPU vs CPU/MTier")
    ax.set_xlabel("Request #")
    ax.set_ylabel("%")
    ax.set_xticks(idx)
    ax.axhline(100, color="gray", linewidth=0.5, linestyle="--", alpha=0.4)
    ax.legend(fontsize=8)

    # Value annotations for all 4 lines.
    # GPU lines label to the left, CPU lines to the right.
    # Cumulative lines label above their marker; per-turn lines label below.
    # Alternating heights (14 pt / 28 pt) prevent neighbour overlap.
    for i, x in enumerate(idx):
        even = (i % 2 == 0)
        lo, hi = (14, 28)

        # GPU cumulative — above-left
        yg = gpu_cumul[i]
        ax.annotate(f"{yg:.1f}%", (x, yg),
                    textcoords="offset points",
                    xytext=(-14, hi if even else lo),
                    ha="center", fontsize=6, color="#4e79a7")
        # GPU per-turn — below-left
        yg_pt = gpu_hit[i]
        ax.annotate(f"{yg_pt:.1f}%", (x, yg_pt),
                    textcoords="offset points",
                    xytext=(-14, -(lo if even else hi)),
                    ha="center", fontsize=6, color="#4e79a7", alpha=0.8)
        # CPU cumulative — above-right
        yc = cpu_cumul[i]
        ax.annotate(f"{yc:.1f}%", (x, yc),
                    textcoords="offset points",
                    xytext=(14, lo if even else hi),
                    ha="center", fontsize=6, color="#f28e2b")
        # CPU per-turn — below-right
        yc_pt = cpu_hit[i]
        ax.annotate(f"{yc_pt:.1f}%", (x, yc_pt),
                    textcoords="offset points",
                    xytext=(14, -(hi if even else lo)),
                    ha="center", fontsize=6, color="#f28e2b", alpha=0.8)

    bar(axes[1][1], g2c_mb, "GPU→CPU Offload (MB)", "#59a14f", "MB")
    bar(axes[1][2], c2g_mb, "CPU→GPU Recall (MB)",  "#76b7b2", "MB")

    # axes[1][3] unused — hide it
    axes[1][3].set_visible(False)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"\nChart saved → {path}")


# ── main ───────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Per-request metrics watcher for vLLM (scrapes /metrics directly)"
    )
    ap.add_argument("--vllm",     default="http://localhost:8000",
                    help="vLLM server base URL (default: http://localhost:8000)")
    ap.add_argument("--interval", type=float, default=0.1,
                    help="Poll interval in seconds (default: 0.1)")
    _static = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    _default_chart = os.path.join(_static, f"session_metrics_vllm_{_ts}.png")
    ap.add_argument("--chart",    default=_default_chart,
                    help="Output chart path (default: static/session_metrics_vllm_<timestamp>.png)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.chart), exist_ok=True)
    vllm_url = args.vllm.rstrip("/")
    turns: list[Turn] = []

    # ── column header ──────────────────────────────────────────────────────────
    hdr = (f"{'#':>4}  {'time':>8}  {'ISL':>7}  {'OSL':>7}"
           f"  {'TTFT':>8}  {'ITL':>8}  {'E2E':>8}"
           f"  {'GPU$':>7}  {'CPU$':>7}"
           f"  {'gCum':>7}  {'cCum':>7}"
           f"  {'G→C':>8}  {'C→G':>8}")
    sep = "─" * len(hdr)
    print(sep)
    print(hdr)
    print(sep)

    # ── baseline state ─────────────────────────────────────────────────────────
    prev_e2e_count    = None
    # ISL / OSL histograms
    prev_isl_sum      = None
    prev_isl_count    = None
    prev_osl_sum      = None
    prev_osl_count    = None
    # Latency histograms
    prev_ttft_sum     = None
    prev_ttft_count   = None
    prev_itl_sum      = None
    prev_itl_count    = None
    prev_e2e_sum      = None
    # E2E phase breakdown (queue / prefill / decode)
    prev_queue_sum    = None
    prev_queue_count  = None
    prev_prefill_sum  = None
    prev_prefill_count= None
    prev_decode_sum   = None
    prev_decode_count = None
    # GPU-local prefix cache
    prev_hits         = None
    prev_hit_queries  = None
    # CPU/MTier external prefix cache
    prev_ext_hits     = None
    prev_ext_queries  = None
    # KV offload bytes (native vLLM OffloadingConnector metrics)
    prev_g2c_bytes    = None
    prev_c2g_bytes    = None
    turn_idx          = 0
    # Buffer completed turns; flush only when server goes idle
    pending_turns: list[Turn] = []
    idle_polls     = 0          # consecutive polls with num_requests_running == 0
    IDLE_THRESHOLD = 2          # flush after this many idle polls (~0.2 s)

    def on_exit(*_):
        if pending_turns:
            for t in pending_turns:
                print(
                    f"{t.index:>4}  {t.ts:>8}"
                    f"  {fmt_f(t.isl, 0):>7}"
                    f"  {fmt_f(t.osl, 0):>7}"
                    f"  {fmt_f(t.ttft_ms, 1, 'ms'):>8}"
                    f"  {fmt_f(t.itl_ms,  2, 'ms'):>8}"
                    f"  {fmt_f(t.e2e_ms,  1, 'ms'):>8}"
                    f"  {fmt_f(None if t.gpu_hit  is None else t.gpu_hit  * 100, 1, '%'):>7}"
                    f"  {fmt_f(None if t.cpu_hit  is None else t.cpu_hit  * 100, 1, '%'):>7}"
                    f"  {fmt_f(None if t.gpu_cumul is None else t.gpu_cumul * 100, 1, '%'):>7}"
                    f"  {fmt_f(None if t.cpu_cumul is None else t.cpu_cumul * 100, 1, '%'):>7}"
                    f"  {fmt_bytes(t.g2c_bytes):>8}"
                    f"  {fmt_bytes(t.c2g_bytes):>8}"
                )
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

    print(f"Scraping {vllm_url}/metrics for completed requests… (Ctrl+C to stop & save chart)\n")

    while True:
        s = scrape(vllm_url)

        # Use e2e_request_latency_seconds_count as the request completion counter.
        # It increments once per completed request — same information as Dynamo's
        # dynamo_frontend_requests_total, but available natively from vLLM.
        e2e_count = sum_metric(s, "vllm:e2e_request_latency_seconds_count")

        if e2e_count is not None:
            if prev_e2e_count is None:
                # ── seed all baselines ─────────────────────────────────────────
                prev_e2e_count    = e2e_count
                prev_isl_sum      = sum_metric(s, "vllm:request_prompt_tokens_sum")
                prev_isl_count    = sum_metric(s, "vllm:request_prompt_tokens_count")
                prev_osl_sum      = sum_metric(s, "vllm:request_generation_tokens_sum")
                prev_osl_count    = sum_metric(s, "vllm:request_generation_tokens_count")
                prev_ttft_sum     = sum_metric(s, "vllm:time_to_first_token_seconds_sum")
                prev_ttft_count   = sum_metric(s, "vllm:time_to_first_token_seconds_count")
                prev_itl_sum      = sum_metric(s, "vllm:inter_token_latency_seconds_sum")
                prev_itl_count    = sum_metric(s, "vllm:inter_token_latency_seconds_count")
                prev_e2e_sum      = sum_metric(s, "vllm:e2e_request_latency_seconds_sum")
                prev_queue_sum    = sum_metric(s, "vllm:request_queue_time_seconds_sum")
                prev_queue_count  = sum_metric(s, "vllm:request_queue_time_seconds_count")
                prev_prefill_sum  = sum_metric(s, "vllm:request_prefill_time_seconds_sum")
                prev_prefill_count= sum_metric(s, "vllm:request_prefill_time_seconds_count")
                prev_decode_sum   = sum_metric(s, "vllm:request_decode_time_seconds_sum")
                prev_decode_count = sum_metric(s, "vllm:request_decode_time_seconds_count")
                prev_hits         = sum_metric(s, "vllm:prefix_cache_hits_total")
                prev_hit_queries  = sum_metric(s, "vllm:prefix_cache_queries_total")
                prev_ext_hits     = sum_metric(s, "vllm:external_prefix_cache_hits_total")
                prev_ext_queries  = sum_metric(s, "vllm:external_prefix_cache_queries_total")
                prev_g2c_bytes    = get_labeled(s, "vllm:kv_offload_total_bytes_total",
                                                transfer_type="GPU_to_CPU")
                prev_c2g_bytes    = get_labeled(s, "vllm:kv_offload_total_bytes_total",
                                                transfer_type="CPU_to_GPU")
                if e2e_count > 0:
                    print(f"[seeded at request #{int(e2e_count)} — earlier requests not shown]")

            elif e2e_count > prev_e2e_count:
                new_requests = int(e2e_count - prev_e2e_count)

                # ── fetch current values ───────────────────────────────────────
                isl_sum       = sum_metric(s, "vllm:request_prompt_tokens_sum")
                isl_count     = sum_metric(s, "vllm:request_prompt_tokens_count")
                osl_sum       = sum_metric(s, "vllm:request_generation_tokens_sum")
                osl_count     = sum_metric(s, "vllm:request_generation_tokens_count")
                ttft_sum      = sum_metric(s, "vllm:time_to_first_token_seconds_sum")
                ttft_count    = sum_metric(s, "vllm:time_to_first_token_seconds_count")
                itl_sum       = sum_metric(s, "vllm:inter_token_latency_seconds_sum")
                itl_count     = sum_metric(s, "vllm:inter_token_latency_seconds_count")
                e2e_sum       = sum_metric(s, "vllm:e2e_request_latency_seconds_sum")
                queue_sum     = sum_metric(s, "vllm:request_queue_time_seconds_sum")
                queue_count   = sum_metric(s, "vllm:request_queue_time_seconds_count")
                prefill_sum   = sum_metric(s, "vllm:request_prefill_time_seconds_sum")
                prefill_count = sum_metric(s, "vllm:request_prefill_time_seconds_count")
                decode_sum    = sum_metric(s, "vllm:request_decode_time_seconds_sum")
                decode_count  = sum_metric(s, "vllm:request_decode_time_seconds_count")
                hits          = sum_metric(s, "vllm:prefix_cache_hits_total")
                hit_q         = sum_metric(s, "vllm:prefix_cache_queries_total")
                ext_hits      = sum_metric(s, "vllm:external_prefix_cache_hits_total")
                ext_q         = sum_metric(s, "vllm:external_prefix_cache_queries_total")
                g2c_bytes     = get_labeled(s, "vllm:kv_offload_total_bytes_total",
                                            transfer_type="GPU_to_CPU")
                c2g_bytes     = get_labeled(s, "vllm:kv_offload_total_bytes_total",
                                            transfer_type="CPU_to_GPU")

                # ── per-turn ISL / OSL ─────────────────────────────────────────
                isl = _delta_avg(isl_sum, isl_count, prev_isl_sum, prev_isl_count)
                osl = _delta_avg(osl_sum, osl_count, prev_osl_sum, prev_osl_count)

                # ── latency: exact per-turn via delta(sum)/delta(count) ────────
                ttft_ms   = _delta_ms(ttft_sum,    ttft_count,    prev_ttft_sum,    prev_ttft_count)
                itl_ms    = _delta_ms(itl_sum,     itl_count,     prev_itl_sum,     prev_itl_count)
                e2e_ms    = _delta_ms(e2e_sum,     e2e_count,     prev_e2e_sum,     prev_e2e_count)
                queue_ms_ = _delta_ms(queue_sum,   queue_count,   prev_queue_sum,   prev_queue_count)
                prefill_ms_= _delta_ms(prefill_sum, prefill_count, prev_prefill_sum, prev_prefill_count)
                decode_ms_ = _delta_ms(decode_sum,  decode_count,  prev_decode_sum,  prev_decode_count)

                # ── GPU (HBM) prefix cache hit rate this turn ──────────────────
                gpu_hit = None
                if None not in (hits, hit_q, prev_hits, prev_hit_queries):
                    d_q = hit_q - prev_hit_queries
                    if d_q > 0:
                        gpu_hit = (hits - prev_hits) / d_q
                gpu_cumul = (hits / hit_q
                             if (hits is not None and hit_q and hit_q > 0) else None)

                # ── CPU/MTier external prefix cache hit rate this turn ─────────
                cpu_hit = None
                if None not in (ext_hits, ext_q, prev_ext_hits, prev_ext_queries):
                    d_ext_q = ext_q - prev_ext_queries
                    if d_ext_q > 0:
                        cpu_hit = (ext_hits - prev_ext_hits) / d_ext_q
                cpu_cumul = (ext_hits / ext_q
                             if (ext_hits is not None and ext_q and ext_q > 0) else None)

                # ── KV transfer bytes this turn ────────────────────────────────
                # vllm:kv_offload_total_bytes_total is a native OffloadingConnector counter
                d_g2c = (max(0.0, g2c_bytes - prev_g2c_bytes)
                         if None not in (g2c_bytes, prev_g2c_bytes) else None)
                d_c2g = (max(0.0, c2g_bytes - prev_c2g_bytes)
                         if None not in (c2g_bytes, prev_c2g_bytes) else None)

                # ── buffer completed turns; print after server goes idle ───────
                for _ in range(new_requests):
                    turn_idx += 1
                    ts = datetime.now().strftime("%H:%M:%S")
                    t  = Turn(
                        turn_idx, ts, isl, osl,
                        ttft_ms, itl_ms, e2e_ms,
                        gpu_hit, cpu_hit, gpu_cumul, cpu_cumul,
                        d_g2c, d_c2g,
                        queue_ms_, prefill_ms_, decode_ms_,
                    )
                    turns.append(t)
                    pending_turns.append(t)

                # ── advance all baselines ──────────────────────────────────────
                prev_e2e_count    = e2e_count
                prev_isl_sum      = isl_sum      if isl_sum      is not None else prev_isl_sum
                prev_isl_count    = isl_count    if isl_count    is not None else prev_isl_count
                prev_osl_sum      = osl_sum      if osl_sum      is not None else prev_osl_sum
                prev_osl_count    = osl_count    if osl_count    is not None else prev_osl_count
                prev_ttft_sum     = ttft_sum     if ttft_sum     is not None else prev_ttft_sum
                prev_ttft_count   = ttft_count   if ttft_count   is not None else prev_ttft_count
                prev_itl_sum      = itl_sum      if itl_sum      is not None else prev_itl_sum
                prev_itl_count    = itl_count    if itl_count    is not None else prev_itl_count
                prev_e2e_sum      = e2e_sum      if e2e_sum      is not None else prev_e2e_sum
                prev_queue_sum    = queue_sum    if queue_sum    is not None else prev_queue_sum
                prev_queue_count  = queue_count  if queue_count  is not None else prev_queue_count
                prev_prefill_sum  = prefill_sum  if prefill_sum  is not None else prev_prefill_sum
                prev_prefill_count= prefill_count if prefill_count is not None else prev_prefill_count
                prev_decode_sum   = decode_sum   if decode_sum   is not None else prev_decode_sum
                prev_decode_count = decode_count if decode_count is not None else prev_decode_count
                prev_hits         = hits         if hits         is not None else prev_hits
                prev_hit_queries  = hit_q        if hit_q        is not None else prev_hit_queries
                prev_ext_hits     = ext_hits     if ext_hits     is not None else prev_ext_hits
                prev_ext_queries  = ext_q        if ext_q        is not None else prev_ext_queries
                prev_g2c_bytes    = g2c_bytes    if g2c_bytes    is not None else prev_g2c_bytes
                prev_c2g_bytes    = c2g_bytes    if c2g_bytes    is not None else prev_c2g_bytes

        # ── flush pending turns when server is idle ────────────────────────
        running = sum_metric(s, "vllm:num_requests_running")
        if running is not None and running == 0 and pending_turns:
            idle_polls += 1
            if idle_polls >= IDLE_THRESHOLD:
                for t in pending_turns:
                    print(
                        f"{t.index:>4}  {t.ts:>8}"
                        f"  {fmt_f(t.isl, 0):>7}"
                        f"  {fmt_f(t.osl, 0):>7}"
                        f"  {fmt_f(t.ttft_ms, 1, 'ms'):>8}"
                        f"  {fmt_f(t.itl_ms,  2, 'ms'):>8}"
                        f"  {fmt_f(t.e2e_ms,  1, 'ms'):>8}"
                        f"  {fmt_f(None if t.gpu_hit  is None else t.gpu_hit  * 100, 1, '%'):>7}"
                        f"  {fmt_f(None if t.cpu_hit  is None else t.cpu_hit  * 100, 1, '%'):>7}"
                        f"  {fmt_f(None if t.gpu_cumul is None else t.gpu_cumul * 100, 1, '%'):>7}"
                        f"  {fmt_f(None if t.cpu_cumul is None else t.cpu_cumul * 100, 1, '%'):>7}"
                        f"  {fmt_bytes(t.g2c_bytes):>8}"
                        f"  {fmt_bytes(t.c2g_bytes):>8}"
                    )
                pending_turns.clear()
                idle_polls = 0
        else:
            idle_polls = 0

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
