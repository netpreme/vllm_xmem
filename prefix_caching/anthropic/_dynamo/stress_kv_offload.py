#!/usr/bin/env python3
"""
KV cache CPU offload stress test.

Drives three phases to confirm CPU offloading is working:

  Phase 1 — FILL
    Send N requests with long unique prefixes → fills GPU KV cache.

  Phase 2 — EVICT
    Send M new requests with different prefixes → pushes Phase 1
    blocks out of GPU into CPU tier (GPU→CPU bytes should appear).

  Phase 3 — RECALL
    Re-send Phase 1 prefixes → scheduler loads them back from CPU
    (CPU→GPU bytes should appear, TTFT higher than Phase 1).

Usage:
    python3 stress_kv_offload.py                     # defaults
    python3 stress_kv_offload.py --fill 20 --evict 25
    python3 stress_kv_offload.py --url http://localhost:8000
    python3 stress_kv_offload.py --no-evict          # GPU-hit baseline only

Reads GPU↔CPU transfer bytes from /tmp/dynamo_worker.log.
"""

import argparse
import json
import re
import statistics
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

# ── defaults ──────────────────────────────────────────────────────

DEFAULT_URL        = "http://127.0.0.1:8000"
DEFAULT_MODEL      = "Qwen/Qwen2.5-Coder-32B-Instruct"
DEFAULT_WORKER_LOG = "/tmp/dynamo_worker.log"
TOKENS_PER_PREFIX  = 1500   # ~94 KV blocks per request (16 tok/block)
FOLLOWUP           = "Summarise the above in one sentence."

# ── helpers ───────────────────────────────────────────────────────

def make_prefix(seed: int, target_tokens: int = TOKENS_PER_PREFIX) -> str:
    sentence = (
        f"Document {seed}, section {seed % 7 + 1}. "
        + " ".join([f"word{seed * 1000 + i}" for i in range(18)])
        + "."
    )
    repeats = max(1, target_tokens // (len(sentence.split()) * 4 // 3))
    return (sentence + " ") * repeats


def ttft_request(url: str, model: str, messages: list[dict],
                 timeout: int = 120) -> float | None:
    payload = json.dumps({
        "model": model, "messages": messages,
        "max_tokens": 16, "stream": True,
    }).encode()
    req = urllib.request.Request(
        f"{url}/v1/messages", data=payload,
        headers={"Content-Type": "application/json",
                 "x-api-key": "dummy",
                 "anthropic-version": "2023-06-01"},
        method="POST",
    )
    t0 = time.perf_counter()
    ttft = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:"):
                    continue
                s = line[5:].strip()
                if s in ("", "[DONE]"):
                    continue
                try:
                    ev = json.loads(s)
                except json.JSONDecodeError:
                    continue
                etype = ev.get("type")
                if ttft is None and etype in ("content_block_delta", "message_delta"):
                    ttft = (time.perf_counter() - t0) * 1000
                # Drain the full response so Dynamo doesn't log stream errors
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"  [request error: {e}]", file=sys.stderr)
    return ttft


def read_log_delta(log_path: str, offset: int) -> tuple[dict, int]:
    delta: dict[str, float] = {}
    try:
        with open(log_path, "r", errors="replace") as f:
            f.seek(0, 2)
            eof = f.tell()
            if offset > eof:
                offset = 0
            f.seek(offset)
            text = f.read()
            new_offset = f.tell()
    except FileNotFoundError:
        return delta, offset
    for m in re.finditer(r"KV Transfer metrics:\s+(.+?)(?:\n|$)", text):
        for kv in m.group(1).split(","):
            kv = kv.strip()
            if "=" in kv:
                k, v = kv.split("=", 1)
                try:
                    delta[k.strip()] = delta.get(k.strip(), 0.0) + float(v.strip())
                except ValueError:
                    pass
    return delta, new_offset


def fmt(values: list[float]) -> str:
    if not values:
        return "no data"
    sv = sorted(values)
    n  = len(sv)
    return (f"mean={statistics.mean(sv):.0f}ms  "
            f"p50={sv[n//2]:.0f}ms  "
            f"p95={sv[int(n*0.95)]:.0f}ms  "
            f"n={n}")


def fmt_bytes(b: float) -> str:
    if b >= 1e9: return f"{b/1e9:.2f} GB"
    if b >= 1e6: return f"{b/1e6:.1f} MB"
    if b >= 1e3: return f"{b/1e3:.1f} KB"
    return f"{b:.0f} B"


# ── phase runner ──────────────────────────────────────────────────

def run_phase(name: str, seeds: list[int], url: str, model: str,
              log_path: str, log_offset: int) -> tuple[list[float], dict, int]:
    print(f"\n{'─'*55}")
    print(f"  {name}  ({len(seeds)} requests)")
    print(f"{'─'*55}")

    # Advance log offset so we only count transfers from this phase
    _, log_offset = read_log_delta(log_path, log_offset)

    ttfts = []
    for i, seed in enumerate(seeds):
        prefix = make_prefix(seed)
        msgs = [
            {"role": "user",      "content": prefix},
            {"role": "assistant", "content": "Understood."},
            {"role": "user",      "content": FOLLOWUP},
        ]
        t = ttft_request(url, model, msgs)
        if t is not None:
            ttfts.append(t)
            print(f"  [{i+1:3d}/{len(seeds)}] seed={seed:5d}  TTFT {t:6.0f} ms")
        else:
            print(f"  [{i+1:3d}/{len(seeds)}] seed={seed:5d}  ERROR")

    cpu_delta, log_offset = read_log_delta(log_path, log_offset)
    g2c = cpu_delta.get("GPU_to_CPU_total_bytes", 0.0)
    c2g = cpu_delta.get("CPU_to_GPU_total_bytes", 0.0)

    print(f"\n  TTFT     : {fmt(ttfts)}")
    if g2c > 0 or c2g > 0:
        print(f"  GPU→CPU  : {fmt_bytes(g2c)}")
        print(f"  CPU→GPU  : {fmt_bytes(c2g)}")
    else:
        print(f"  CPU offload: none detected this phase")

    return ttfts, cpu_delta, log_offset


# ── main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url",      default=DEFAULT_URL)
    parser.add_argument("--model",    default=DEFAULT_MODEL)
    parser.add_argument("--log",      default=DEFAULT_WORKER_LOG)
    parser.add_argument("--fill",     type=int, default=20,
                        help="requests in Fill phase (default 20)")
    parser.add_argument("--evict",    type=int, default=25,
                        help="requests in Evict phase (default 25)")
    parser.add_argument("--recall",   type=int, default=15,
                        help="Fill requests to replay in Recall (default 15)")
    parser.add_argument("--no-evict", action="store_true",
                        help="Skip eviction — GPU-hit baseline only")
    args = parser.parse_args()

    print(f"\nKV Offload Stress Test")
    print(f"  endpoint : {args.url}")
    print(f"  model    : {args.model}")
    print(f"  log      : {args.log}")
    print(f"  fill={args.fill}  evict={0 if args.no_evict else args.evict}  recall={args.recall}")

    try:
        urllib.request.urlopen(f"{args.url}/health", timeout=5)
    except Exception:
        print(f"\nERROR: server not reachable at {args.url}", file=sys.stderr)
        sys.exit(1)

    # Advance log to current end so pre-existing transfers aren't counted
    log_offset = 0
    try:
        with open(args.log, "r") as f:
            f.seek(0, 2)
            log_offset = f.tell()
    except FileNotFoundError:
        pass

    fill_seeds   = list(range(args.fill))
    evict_seeds  = list(range(10000, 10000 + args.evict))
    recall_seeds = fill_seeds[:args.recall]

    results = {}

    fill_ttfts, _, log_offset = run_phase(
        "PHASE 1 — FILL (cold prefill, blocks enter GPU)",
        fill_seeds, args.url, args.model, args.log, log_offset,
    )
    results["fill"] = fill_ttfts

    if not args.no_evict:
        evict_ttfts, _, log_offset = run_phase(
            "PHASE 2 — EVICT (new prefixes push Fill blocks to CPU)",
            evict_seeds, args.url, args.model, args.log, log_offset,
        )
        results["evict"] = evict_ttfts

    recall_ttfts, _, log_offset = run_phase(
        "PHASE 3 — RECALL (re-request Fill prefixes — loads from CPU)",
        recall_seeds, args.url, args.model, args.log, log_offset,
    )
    results["recall"] = recall_ttfts

    # ── summary ───────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  SUMMARY")
    print(f"{'═'*55}")
    for name, ttfts in results.items():
        print(f"  {name:8s} : {fmt(ttfts)}")

    if results.get("fill") and results.get("recall"):
        overhead = statistics.mean(results["recall"]) - statistics.mean(results["fill"])
        sign = "+" if overhead >= 0 else ""
        print(f"\n  CPU load overhead (recall - fill mean): {sign}{overhead:.0f} ms")
        if overhead > 50:
            print(f"  ✓ CPU offloading confirmed — RECALL is slower due to CPU→GPU load")
        else:
            print(f"  ✗ No overhead detected")
            print(f"    Check worker log for 'External prefix cache hit rate'")
            print(f"    If 0.0%: prefix hash mismatch (Rust tokenizer issue)")
            print(f"    GPU→CPU write-through still works even without cache hits")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()
