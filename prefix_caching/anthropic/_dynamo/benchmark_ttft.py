#!/usr/bin/env python3
"""
KV cache tier TTFT benchmark.

Drives three phases to isolate CPU↔GPU offload latency:

  Phase 1 — FILL
    Send N requests with long, unique prefixes.
    Each prefix occupies GPU KV blocks.
    Target: fill GPU to ~90% capacity.

  Phase 2 — EVICT
    Send M new requests with different prefixes to push
    Phase 1 blocks out of GPU via LRU into the CPU tier.
    Target: GPU still ~90% full, Phase 1 blocks now in CPU.

  Phase 3 — RECALL
    Re-send Phase 1 prefixes.
    Scheduler finds them in CPU → triggers CPU→GPU load → TTFT increases.
    Compare with Phase 1 TTFT (GPU-hit baseline) to isolate load cost.

Usage:
  python3 benchmark_ttft.py                        # defaults
  python3 benchmark_ttft.py --fill 40 --evict 50   # tune phases
  python3 benchmark_ttft.py --util 0.50             # matches .env setting
  python3 benchmark_ttft.py --no-evict              # GPU-hit baseline only

Reports:
  TTFT (ms) per phase — mean / p50 / p95
  GPU→CPU / CPU→GPU bytes from worker log
  GPU fill % at each phase boundary
"""

import argparse
import json
import re
import os
import sys
import time
import urllib.request
import urllib.error
import statistics
from dataclasses import dataclass, field
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────

DEFAULT_BASE_URL  = "http://127.0.0.1:8000"
DEFAULT_MODEL     = "Qwen/Qwen2.5-Coder-32B-Instruct"
DEFAULT_WORKER_LOG = "/tmp/dynamo_worker.log"

# Tokens per phase-1/evict request (prefix length).
# Keep large enough to fill multiple KV blocks (16 tok/block).
# ~1500 tokens ≈ 94 blocks per request.
TOKENS_PER_PREFIX = 1500

# Short follow-up to force the model to produce at least 1 token
# (required to measure TTFT).
FOLLOWUP = "Summarise the above in one sentence."

# ── helpers ───────────────────────────────────────────────────────────

def make_long_prefix(seed: int, target_tokens: int = TOKENS_PER_PREFIX) -> str:
    """Generate a deterministic, unique long user message."""
    # Each seed produces a structurally different document so hash prefixes differ.
    words_per_sentence = 18
    sentence = (
        f"This is document {seed}, section {seed % 7 + 1}. "
        + " ".join([
            f"word{seed * 1000 + i}"
            for i in range(words_per_sentence)
        ])
        + "."
    )
    # Repeat to reach target length (~1.3 tokens/word rough estimate)
    repeats = max(1, target_tokens // (len(sentence.split()) * 4 // 3))
    return (sentence + " ") * repeats


def ttft_request(base_url: str, model: str, messages: list[dict],
                 max_tokens: int = 16, timeout: int = 120) -> float | None:
    """
    Send a streaming request and return time-to-first-token in ms.
    Returns None on error.
    """
    payload = json.dumps({
        "model":      model,
        "messages":   messages,
        "max_tokens": max_tokens,
        "stream":     True,
    }).encode()

    req = urllib.request.Request(
        f"{base_url}/v1/messages",
        data=payload,
        headers={
            "Content-Type":        "application/json",
            "x-api-key":           "dummy",
            "anthropic-version":   "2023-06-01",
        },
        method="POST",
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode().strip()
                if not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str in ("", "[DONE]"):
                    continue
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                etype = event.get("type", "")
                # First content block delta = first token
                if etype == "content_block_delta":
                    return (time.perf_counter() - t0) * 1000
                # Also accept message_delta with stop reason (short replies)
                if etype == "message_delta":
                    return (time.perf_counter() - t0) * 1000
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"  [request error: {e}]", file=sys.stderr)
        return None
    return None


def fetch_metric(base_url: str, name: str) -> float:
    """Read a single Dynamo Prometheus gauge."""
    try:
        with urllib.request.urlopen(f"{base_url}/metrics", timeout=3) as r:
            text = r.read().decode()
    except Exception:
        return 0.0
    m = re.search(r'^' + re.escape(name) + r'(?:\{[^}]*\})?\s+([\d.e+\-]+)',
                  text, re.MULTILINE)
    return float(m.group(1)) if m else 0.0


def gpu_fill_pct(base_url: str) -> float:
    """Return GPU KV cache fill % (0-100)."""
    blocks_used  = fetch_metric(base_url, "dynamo_frontend_input_sequence_tokens_sum")
    total_blocks = fetch_metric(base_url, "dynamo_frontend_model_total_kv_blocks")
    kv_usage     = fetch_metric(base_url, "vllm:kv_cache_usage_perc")
    if kv_usage > 0:
        return kv_usage * 100
    return 0.0


def read_cpu_offload_log(log_path: str, offset: int) -> tuple[dict, int]:
    """Read new KV Transfer metric lines from worker log since offset."""
    delta: dict[str, float] = {}
    try:
        with open(log_path, 'r', errors='replace') as f:
            f.seek(0, 2)
            eof = f.tell()
            if offset > eof:
                offset = 0
            f.seek(offset)
            text = f.read()
            new_offset = f.tell()
    except FileNotFoundError:
        return delta, offset

    for m in re.finditer(r'KV Transfer metrics:\s+(.+?)(?:\n|$)', text):
        for kv in m.group(1).split(','):
            kv = kv.strip()
            if '=' in kv:
                k, v = kv.split('=', 1)
                try:
                    delta[k.strip()] = delta.get(k.strip(), 0.0) + float(v.strip())
                except ValueError:
                    pass
    return delta, new_offset


def fmt_ms(values: list[float]) -> str:
    if not values:
        return "no data"
    p50 = statistics.median(values)
    p95 = sorted(values)[int(len(values) * 0.95)]
    return f"mean={statistics.mean(values):.0f}ms  p50={p50:.0f}ms  p95={p95:.0f}ms  n={len(values)}"


def fmt_bytes(b: float) -> str:
    if b >= 1e9:  return f"{b/1e9:.2f} GB"
    if b >= 1e6:  return f"{b/1e6:.1f} MB"
    if b >= 1e3:  return f"{b/1e3:.1f} KB"
    return f"{b:.0f} B"


# ── benchmark ─────────────────────────────────────────────────────────

@dataclass
class PhaseResult:
    name:     str
    ttfts:    list[float] = field(default_factory=list)
    g2c_bytes: float = 0.0
    c2g_bytes: float = 0.0
    gpu_fill_before: float = 0.0
    gpu_fill_after:  float = 0.0


def run_phase(
    name:        str,
    seeds:       list[int],
    base_url:    str,
    model:       str,
    log_path:    str,
    log_offset:  int,
    verbose:     bool = True,
) -> tuple[PhaseResult, int]:
    result = PhaseResult(name=name)
    result.gpu_fill_before = gpu_fill_pct(base_url)

    delta, new_offset = read_cpu_offload_log(log_path, log_offset)
    # reset — we only care about deltas within this phase
    _, new_offset = read_cpu_offload_log(log_path, new_offset)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  Phase: {name}  ({len(seeds)} requests, GPU fill: {result.gpu_fill_before:.1f}%)")
        print(f"{'─'*60}")

    for i, seed in enumerate(seeds):
        prefix = make_long_prefix(seed)
        messages = [
            {"role": "user",      "content": prefix},
            {"role": "assistant", "content": "Understood."},
            {"role": "user",      "content": FOLLOWUP},
        ]
        t = ttft_request(base_url, model, messages)
        if t is not None:
            result.ttfts.append(t)
            if verbose:
                tag = f"seed={seed:4d}"
                print(f"  [{i+1:3d}/{len(seeds)}] {tag}  TTFT: {t:6.0f} ms")
        else:
            if verbose:
                print(f"  [{i+1:3d}/{len(seeds)}] seed={seed:4d}  TTFT: ERROR")

    phase_delta, new_offset = read_cpu_offload_log(log_path, new_offset)
    result.g2c_bytes = phase_delta.get('GPU_to_CPU_total_bytes', 0.0)
    result.c2g_bytes = phase_delta.get('CPU_to_GPU_total_bytes', 0.0)
    result.gpu_fill_after = gpu_fill_pct(base_url)
    return result, new_offset


def print_summary(phases: list[PhaseResult]):
    print(f"\n{'═'*60}")
    print("  BENCHMARK SUMMARY")
    print(f"{'═'*60}")
    for r in phases:
        print(f"\n  {r.name}")
        print(f"    TTFT     : {fmt_ms(r.ttfts)}")
        print(f"    GPU fill : {r.gpu_fill_before:.1f}% → {r.gpu_fill_after:.1f}%")
        if r.g2c_bytes > 0 or r.c2g_bytes > 0:
            print(f"    GPU→CPU  : {fmt_bytes(r.g2c_bytes)}")
            print(f"    CPU→GPU  : {fmt_bytes(r.c2g_bytes)}")
        else:
            print(f"    CPU offload: none in this phase")

    # Delta: recall vs fill (isolates CPU load cost)
    fill_phase   = next((r for r in phases if "FILL"   in r.name.upper()), None)
    recall_phase = next((r for r in phases if "RECALL" in r.name.upper()), None)
    if fill_phase and recall_phase and fill_phase.ttfts and recall_phase.ttfts:
        delta = statistics.mean(recall_phase.ttfts) - statistics.mean(fill_phase.ttfts)
        print(f"\n  CPU load overhead (RECALL - FILL mean TTFT): {delta:+.0f} ms")
    print(f"\n{'═'*60}\n")


# ── main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url",   default=DEFAULT_BASE_URL)
    parser.add_argument("--model",      default=DEFAULT_MODEL)
    parser.add_argument("--log",        default=DEFAULT_WORKER_LOG)
    parser.add_argument("--fill",       type=int, default=35,
                        help="# requests in Fill phase (default 35 ≈ fills 47K tok GPU)")
    parser.add_argument("--evict",      type=int, default=40,
                        help="# requests in Evict phase (default 40 pushes Fill blocks to CPU)")
    parser.add_argument("--recall",     type=int, default=20,
                        help="# Fill requests to re-request in Recall phase (default 20)")
    parser.add_argument("--no-evict",   action="store_true",
                        help="Skip eviction — benchmarks GPU-hit TTFT only")
    parser.add_argument("--quiet",      action="store_true",
                        help="Print only summary, not per-request lines")
    args = parser.parse_args()

    # Seed ranges: Fill uses 0..fill-1, Evict uses 10000..10000+evict-1
    fill_seeds   = list(range(args.fill))
    evict_seeds  = list(range(10000, 10000 + args.evict))
    recall_seeds = fill_seeds[:args.recall]  # re-request first N fill seeds

    print(f"KV Cache TTFT Benchmark")
    print(f"  endpoint : {args.base_url}")
    print(f"  model    : {args.model}")
    print(f"  log      : {args.log}")
    print(f"  phases   : fill={args.fill}  evict={0 if args.no_evict else args.evict}  recall={args.recall}")
    print(f"  prefix   : ~{TOKENS_PER_PREFIX} tokens/request")

    # Check server is up
    try:
        urllib.request.urlopen(f"{args.base_url}/health", timeout=5)
    except Exception:
        print(f"\nERROR: server not reachable at {args.base_url}", file=sys.stderr)
        sys.exit(1)

    log_offset = 0
    # Advance log offset to current end (don't count pre-existing transfers)
    try:
        with open(args.log, 'r') as f:
            f.seek(0, 2)
            log_offset = f.tell()
    except FileNotFoundError:
        pass

    phases: list[PhaseResult] = []

    # ── Phase 1: Fill ─────────────────────────────────────────────────
    fill_result, log_offset = run_phase(
        "FILL (first-time prefill, blocks enter GPU)",
        fill_seeds, args.base_url, args.model, args.log, log_offset,
        verbose=not args.quiet,
    )
    phases.append(fill_result)

    # ── Phase 2: Evict ────────────────────────────────────────────────
    if not args.no_evict:
        evict_result, log_offset = run_phase(
            "EVICT (new prefixes push Fill blocks to CPU)",
            evict_seeds, args.base_url, args.model, args.log, log_offset,
            verbose=not args.quiet,
        )
        phases.append(evict_result)

    # ── Phase 3: Recall ───────────────────────────────────────────────
    recall_result, log_offset = run_phase(
        "RECALL (re-request Fill prefixes — should load from CPU)",
        recall_seeds, args.base_url, args.model, args.log, log_offset,
        verbose=not args.quiet,
    )
    phases.append(recall_result)

    print_summary(phases)


if __name__ == "__main__":
    main()
