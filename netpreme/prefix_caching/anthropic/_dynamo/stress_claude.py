#!/usr/bin/env python3
"""
KV cache CPU offload stress test using Claude Code as the client.

Claude Code's large system prompt (~22K tokens) is consistent across
runs, so prefix hashes are stable and the cache works correctly.

  Phase 1 — FILL:   run N multi-turn sessions → fills GPU + writes all
                     blocks to CPU/chip tier via write-through
  Phase 2 — EVICT:  run M different sessions → pushes Phase 1 blocks
                     out of GPU → CPU offload manager
  Phase 3 — RECALL: re-run Phase 1 sessions from scratch → loads blocks
                     back from CPU/chip tier into GPU via DMA

HOW EVICTION WORKS
  The GPU block pool (2387 blocks × 16 tokens = ~38K token capacity) must
  be SATURATED before fill blocks can be evicted to the offload tier.

  - Claude Code's system prompt uses ~1431 shared blocks (identical every
    session, so only allocated once and LRU-refreshed by each session).
  - Each session-turn contributes ~21 UNIQUE blocks (user msg + response).
  - Free GPU blocks after shared prefix: 2387 - 1431 ≈ 956 blocks.

  GPU saturation requirement:
    (fill + evict) × turns × 21 > 956
    → fill=10, evict=10, turns=1:  420 < 956  ✗  NO GPU eviction
    → fill=10, evict=10, turns=3: 1260 > 956  ✓  ~304 fill blocks evicted
    → fill=10, evict=10, turns=5: 2100 > 956  ✓  ~1144 fill blocks evicted

  Without GPU eviction the RECALL phase finds everything in the GPU prefix
  cache, so CPU→GPU = 0.  Use --turns 3 or more (the default).

  Chip (CPU DRAM or xmem) saturation for bandwidth benchmarking:
    Single-turn (T=1):  ~21 unique blocks/session → need ~879 sessions to fill 73 GB chip
    Multi-turn  (T=5):  ~105 unique blocks/session → need ~176 sessions to fill chip
    Multi-turn  (T=10): ~210 unique blocks/session → need ~88 sessions to fill chip

Prerequisites:
    ./launch_dynamo.sh              # Dynamo + vLLM must be running
    python3 ttft_proxy.py &         # proxy on :8001 for TTFT measurement

Usage:
    python3 stress_claude.py
    python3 stress_claude.py --fill 20 --evict 15 --recall 15 --turns 5
    python3 stress_claude.py --no-proxy   # point directly at :8000
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ── config ────────────────────────────────────────────────────────

PROXY_URL      = "http://localhost:8001"
DYNAMO_URL     = "http://localhost:8000"
MODEL          = "Qwen/Qwen2.5-Coder-32B-Instruct"
LOG_FILE       = Path("ttft_log.jsonl")
WORKER_LOG     = "/tmp/dynamo_worker.log"
SESSION_FILE   = Path("fill_sessions.json")  # stores session IDs for recall
KV_OFFLOAD_GB  = 72   # --kv-offloading-size passed to launch_dynamo.sh

# Turn 1: initial coding question
# Turns 2+: follow-up prompts that build on the previous response,
#            growing the conversation context (more unique blocks per session)
FILL_TASKS = [
    "Print a Python function that returns the nth fibonacci number using recursion.",
    "Print a Python function that does binary search on a sorted list.",
    "Print a Python function that reverses a linked list in place.",
    "Print a Python function that checks if a string is a palindrome.",
    "Print a Python function that finds all prime numbers up to n using the sieve of Eratosthenes.",
    "Print a Python function that implements bubble sort.",
    "Print a Python function that computes the greatest common divisor using Euclid's algorithm.",
    "Print a Python function that flattens a nested list of arbitrary depth.",
    "Print a Python function that implements a simple LRU cache using an OrderedDict.",
    "Print a Python function that counts word frequencies in a string.",
    "Print a Python function that validates a valid IPv4 address string.",
    "Print a Python function that converts a decimal number to binary without using bin().",
    "Print a Python function that finds the longest common subsequence of two strings.",
    "Print a Python function that implements depth-first search on a graph.",
    "Print a Python function that merges two sorted arrays into one sorted array.",
]

# Follow-up turns that extend the conversation (each adds ~200-400 unique tokens)
FOLLOW_UP_TURNS = [
    "Now add input validation and raise ValueError for invalid inputs.",
    "Add a docstring with Args, Returns, and a usage example.",
    "Write 3 pytest unit tests covering edge cases.",
    "Rewrite it iteratively instead of recursively, keeping the same interface.",
    "Add type hints and make it work with Python 3.8+.",
    "Optimize it for performance and explain what you changed.",
    "Add error handling for all possible exception cases.",
    "Print a second version using a different algorithm and compare the two.",
]

EVICT_TASKS = [
    "Print a Python function that implements quicksort.",
    "Print a Python function that checks if a binary tree is balanced.",
    "Print a Python function that serializes and deserializes a binary tree.",
    "Print a Python function that finds the kth largest element in an array.",
    "Print a Python function that implements a min-heap from scratch.",
    "Print a Python function that solves the coin change problem with dynamic programming.",
    "Print a Python function that finds all permutations of a list.",
    "Print a Python function that implements run-length encoding.",
    "Print a Python function that rotates a matrix 90 degrees clockwise.",
    "Print a Python function that finds the median of two sorted arrays.",
    "Print a Python function that implements the Knuth-Morris-Pratt string search.",
    "Print a Python function that computes the edit distance between two strings.",
    "Print a Python function that implements a trie data structure.",
    "Print a Python function that finds the longest palindromic substring.",
    "Print a Python function that implements topological sort.",
    "Print a Python function that solves the N-queens problem.",
    "Print a Python function that computes the number of ways to climb N stairs.",
]

# ── helpers ───────────────────────────────────────────────────────

def set_proxy_context(task_id: str, proxy_url: str):
    try:
        payload = json.dumps({"task_id": task_id, "turn": 0}).encode()
        req = urllib.request.Request(
            f"{proxy_url}/x-benchmark-context",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=3)
    except Exception:
        pass


def _make_env(base_url: str, model: str) -> dict:
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"]             = base_url
    env["ANTHROPIC_API_KEY"]              = "dummy"
    env["ANTHROPIC_AUTH_TOKEN"]           = "dummy"
    env["ANTHROPIC_DEFAULT_OPUS_MODEL"]   = model
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"]  = model
    env["CLAUDE_CODE_ATTRIBUTION_HEADER"] = "0"
    return env


def run_claude_turn1(prompt: str, base_url: str, model: str,
                     timeout: int = 120) -> tuple[float, str | None]:
    """Run the first turn of a new session. Returns (elapsed, session_id)."""
    env = _make_env(base_url, model)
    cmd = [
        "claude", "-p", prompt,
        "--model", model,
        "--output-format", "json",   # need JSON to capture session_id
        "--dangerously-skip-permissions",
    ]
    t0 = time.perf_counter()
    result = subprocess.run(cmd, env=env, capture_output=True,
                            timeout=timeout, text=True)
    elapsed = time.perf_counter() - t0

    session_id = None
    try:
        data = json.loads(result.stdout)
        session_id = data.get("session_id")
    except (json.JSONDecodeError, AttributeError):
        pass

    return elapsed, session_id


def run_claude_resume(prompt: str, session_id: str, base_url: str, model: str,
                      timeout: int = 120) -> float:
    """Continue an existing session (adds to conversation history)."""
    env = _make_env(base_url, model)
    cmd = [
        "claude", "-p", prompt,
        "--resume", session_id,
        "--model", model,
        "--output-format", "text",
        "--dangerously-skip-permissions",
    ]
    t0 = time.perf_counter()
    subprocess.run(cmd, env=env, capture_output=True,
                   timeout=timeout, text=True)
    return time.perf_counter() - t0


def run_session(task_idx: int, prompt: str, num_turns: int,
                base_url: str, model: str,
                use_proxy: bool, phase_name: str) -> tuple[float, str | None]:
    """
    Run a full multi-turn session.
    Turn 1: fresh session with the coding prompt.
    Turns 2+: follow-up prompts using --resume to build conversation history.
    Returns (total_elapsed, session_id).
    """
    task_id = f"{phase_name}_{task_idx:02d}"
    if use_proxy:
        set_proxy_context(task_id, PROXY_URL)

    # Turn 1 — new session
    elapsed, session_id = run_claude_turn1(prompt, base_url, model)
    total_elapsed = elapsed

    # Turns 2..N — resume same session
    if session_id and num_turns > 1:
        for turn_idx in range(1, num_turns):
            follow_up = FOLLOW_UP_TURNS[(turn_idx - 1) % len(FOLLOW_UP_TURNS)]
            turn_elapsed = run_claude_resume(
                follow_up, session_id, base_url, model)
            total_elapsed += turn_elapsed

    return total_elapsed, session_id


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


def read_ttft_for_task(task_id: str) -> list[float]:
    ttfts = []
    if not LOG_FILE.exists():
        return ttfts
    with open(LOG_FILE) as f:
        for line in f:
            try:
                r = json.loads(line)
                if r.get("task_id") == task_id and r.get("ttft_ms") is not None:
                    ttfts.append(r["ttft_ms"])
            except json.JSONDecodeError:
                pass
    return ttfts


def fmt_ms(values: list[float]) -> str:
    if not values:
        return "no proxy data"
    sv = sorted(values)
    n  = len(sv)
    return (f"mean={statistics.mean(sv):.0f}ms  "
            f"p50={sv[n//2]:.0f}ms  n={n}")


def fmt_bytes(b: float) -> str:
    if b >= 1e9: return f"{b/1e9:.2f} GB"
    if b >= 1e6: return f"{b/1e6:.1f} MB"
    if b >= 1e3: return f"{b/1e3:.1f} KB"
    return f"{b:.0f} B"


# ── phase runners ─────────────────────────────────────────────────

def _dma_bw(bytes_: float, dma_seconds: float) -> str:
    """Format DMA bandwidth from worker-log transfer time (not wall clock)."""
    if bytes_ <= 0 or dma_seconds <= 0:
        return ""
    gb_s = bytes_ / dma_seconds / 1e9
    return f"  → {gb_s:.1f} GB/s DMA"


def _bw_str(bytes_: float, seconds: float) -> str:
    """Format bandwidth given total bytes and total wall seconds."""
    if bytes_ <= 0 or seconds <= 0:
        return ""
    gb_s = bytes_ / seconds / 1e9
    return f"  ({gb_s:.1f} GB/s wall)"


def run_fill_phase(tasks: list[str], num_turns: int,
                   base_url: str, model: str,
                   log_offset: int, use_proxy: bool) -> tuple[list[float], dict, int, list[str | None]]:
    """
    Fill phase: run N multi-turn sessions and save session IDs for recall.
    """
    print(f"\n{'─'*55}")
    print(f"  PHASE 1 — FILL  ({len(tasks)} tasks × {num_turns} turn{'s' if num_turns>1 else ''})")
    print(f"  Each session builds ~{num_turns * 21} unique KV blocks")
    print(f"{'─'*55}")

    _, log_offset = read_log_delta(WORKER_LOG, log_offset)

    elapsed_list = []
    ttft_list    = []
    session_ids  = []
    phase_start  = time.perf_counter()

    for i, prompt in enumerate(tasks):
        total_elapsed, session_id = run_session(
            i, prompt, num_turns, base_url, model, use_proxy, "1")
        elapsed_list.append(total_elapsed)
        session_ids.append(session_id)

        ttfts = read_ttft_for_task(f"1_{i:02d}") if use_proxy else []
        ttft_list.extend(ttfts)

        ttft_str = f"  TTFT {ttfts[0]:.0f}ms" if ttfts else ""
        sid_str  = f"  sid={session_id[:8]}" if session_id else "  no-sid"
        print(f"  [{i+1:2d}/{len(tasks)}] {total_elapsed:5.1f}s{ttft_str}{sid_str}  "
              f"{prompt[:45]}...")

    phase_wall = time.perf_counter() - phase_start
    cpu_delta, log_offset = read_log_delta(WORKER_LOG, log_offset)
    g2c      = cpu_delta.get("GPU_to_CPU_total_bytes", 0.0)
    c2g      = cpu_delta.get("CPU_to_GPU_total_bytes", 0.0)
    g2c_time = cpu_delta.get("GPU_to_CPU_total_time",  0.0)
    c2g_time = cpu_delta.get("CPU_to_GPU_total_time",  0.0)

    print(f"\n  Elapsed  : mean={statistics.mean(elapsed_list):.1f}s  total={phase_wall:.1f}s")
    if ttft_list:
        print(f"  TTFT     : {fmt_ms(ttft_list)}")
    print(f"  GPU→CPU  : {fmt_bytes(g2c)}{_dma_bw(g2c, g2c_time)}")
    print(f"  CPU→GPU  : {fmt_bytes(c2g)}{_dma_bw(c2g, c2g_time)}")

    # Save session IDs so recall phase can reuse them
    SESSION_FILE.write_text(json.dumps(session_ids))
    print(f"  Sessions : saved {len([s for s in session_ids if s])} session IDs → {SESSION_FILE}")

    return elapsed_list, cpu_delta, log_offset, session_ids


def run_evict_phase(tasks: list[str], num_turns: int,
                    base_url: str, model: str,
                    log_offset: int, use_proxy: bool) -> tuple[list[float], dict, int]:
    print(f"\n{'─'*55}")
    print(f"  PHASE 2 — EVICT  ({len(tasks)} tasks × {num_turns} turn{'s' if num_turns>1 else ''})")
    print(f"  Pushes FILL blocks out of GPU into CPU/chip tier")
    print(f"{'─'*55}")

    _, log_offset = read_log_delta(WORKER_LOG, log_offset)

    elapsed_list = []
    ttft_list    = []
    phase_start  = time.perf_counter()

    for i, prompt in enumerate(tasks):
        total_elapsed, _ = run_session(
            i, prompt, num_turns, base_url, model, use_proxy, "2")
        elapsed_list.append(total_elapsed)

        ttfts = read_ttft_for_task(f"2_{i:02d}") if use_proxy else []
        ttft_list.extend(ttfts)

        ttft_str = f"  TTFT {ttfts[0]:.0f}ms" if ttfts else ""
        print(f"  [{i+1:2d}/{len(tasks)}] {total_elapsed:5.1f}s{ttft_str}  "
              f"{prompt[:55]}...")

    phase_wall = time.perf_counter() - phase_start
    cpu_delta, log_offset = read_log_delta(WORKER_LOG, log_offset)
    g2c      = cpu_delta.get("GPU_to_CPU_total_bytes", 0.0)
    c2g      = cpu_delta.get("CPU_to_GPU_total_bytes", 0.0)
    g2c_time = cpu_delta.get("GPU_to_CPU_total_time",  0.0)
    c2g_time = cpu_delta.get("CPU_to_GPU_total_time",  0.0)

    print(f"\n  Elapsed  : mean={statistics.mean(elapsed_list):.1f}s  total={phase_wall:.1f}s")
    if ttft_list:
        print(f"  TTFT     : {fmt_ms(ttft_list)}")
    print(f"  GPU→CPU  : {fmt_bytes(g2c)}{_dma_bw(g2c, g2c_time)}")
    print(f"  CPU→GPU  : {fmt_bytes(c2g)}{_dma_bw(c2g, c2g_time)}")

    return elapsed_list, cpu_delta, log_offset


def run_recall_phase(tasks: list[str], num_turns: int,
                     base_url: str, model: str,
                     log_offset: int, use_proxy: bool,
                     session_ids: list[str | None]) -> tuple[list[float], dict, int]:
    """
    Recall phase: re-run the FILL sessions from scratch (fresh sessions,
    same prompts). vLLM finds prefix blocks in the CPU/chip tier and loads
    them back to GPU via DMA. Higher bandwidth tier → faster recall.
    """
    print(f"\n{'─'*55}")
    print(f"  PHASE 3 — RECALL  ({len(tasks)} tasks × {num_turns} turn{'s' if num_turns>1 else ''})")
    print(f"  Re-runs FILL sessions — blocks load from CPU/chip tier")
    print(f"  Higher bandwidth → lower elapsed time & higher GB/s")
    print(f"{'─'*55}")

    _, log_offset = read_log_delta(WORKER_LOG, log_offset)

    elapsed_list = []
    ttft_list    = []
    phase_start  = time.perf_counter()

    for i, prompt in enumerate(tasks):
        # Fresh session — same prompt as fill, no --resume
        # vLLM will look up the prefix hash and find blocks on the tier
        total_elapsed, _ = run_session(
            i, prompt, num_turns, base_url, model, use_proxy, "3")
        elapsed_list.append(total_elapsed)

        ttfts = read_ttft_for_task(f"3_{i:02d}") if use_proxy else []
        ttft_list.extend(ttfts)

        ttft_str = f"  TTFT {ttfts[0]:.0f}ms" if ttfts else ""
        print(f"  [{i+1:2d}/{len(tasks)}] {total_elapsed:5.1f}s{ttft_str}  "
              f"{prompt[:55]}...")

    phase_wall = time.perf_counter() - phase_start
    cpu_delta, log_offset = read_log_delta(WORKER_LOG, log_offset)
    g2c      = cpu_delta.get("GPU_to_CPU_total_bytes", 0.0)
    c2g      = cpu_delta.get("CPU_to_GPU_total_bytes", 0.0)
    g2c_time = cpu_delta.get("GPU_to_CPU_total_time",  0.0)
    c2g_time = cpu_delta.get("CPU_to_GPU_total_time",  0.0)

    # DMA bandwidth: bytes / actual DMA time (from log) — excludes inference time
    # This is the number to compare across CPU DRAM vs xmem runs
    dma_bw_gb_s = c2g / c2g_time / 1e9 if c2g > 0 and c2g_time > 0 else 0.0

    print(f"\n  Elapsed  : mean={statistics.mean(elapsed_list):.1f}s  total={phase_wall:.1f}s")
    if ttft_list:
        print(f"  TTFT     : {fmt_ms(ttft_list)}")
    print(f"  GPU→CPU  : {fmt_bytes(g2c)}{_dma_bw(g2c, g2c_time)}")
    print(f"  CPU→GPU  : {fmt_bytes(c2g)}{_dma_bw(c2g, c2g_time)}")
    if dma_bw_gb_s > 0:
        print(f"  *** Recall DMA bandwidth : {dma_bw_gb_s:.1f} GB/s ***")
        print(f"      CPU DRAM theoretical ~55 GB/s | xmem theoretical ~1400 GB/s")

    # Store bandwidth in delta dict for summary
    cpu_delta["_recall_bw_gb_s"] = dma_bw_gb_s
    cpu_delta["_recall_wall_s"]  = phase_wall

    return elapsed_list, cpu_delta, log_offset


# ── main ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fill",     type=int, default=10,
                        help="Number of fill sessions (default 10)")
    parser.add_argument("--evict",    type=int, default=10,
                        help="Number of evict sessions (default 10)")
    parser.add_argument("--recall",   type=int, default=8,
                        help="Number of fill sessions to replay in recall (default 8)")
    parser.add_argument("--turns",    type=int, default=3,
                        help="Turns per session (default 3). More turns = more unique blocks per session")
    parser.add_argument("--no-proxy", action="store_true",
                        help="Skip TTFT proxy, go direct to Dynamo")
    parser.add_argument("--no-evict", action="store_true")
    parser.add_argument("--label",    default="",
                        help="Label for this run (e.g. 'cpu' or 'xmem') — shown in summary")
    parser.add_argument("--save",     default="",
                        help="Save JSON summary to this file for offline comparison")
    args = parser.parse_args()

    base_url  = DYNAMO_URL if args.no_proxy else PROXY_URL
    use_proxy = not args.no_proxy

    # Saturation check:
    #   GPU must be saturated first (fill blocks evicted to CPU) before
    #   the RECALL phase can demonstrate CPU→GPU bandwidth.
    #
    #   GPU block budget (2387 blocks total, ~1431 consumed by the shared
    #   system-prompt prefix that every session touches).
    #   Each session-turn contributes ~21 unique blocks (user msg + response).
    GPU_BLOCKS          = 2387   # from dynamo_frontend_model_total_kv_blocks
    GPU_SHARED_BLOCKS   = 1431   # Claude Code system-prompt prefix (~22.9K tok)
    UNIQUE_BLOCKS_TURN  = 21     # unique blocks per session-turn
    gpu_free_blocks     = GPU_BLOCKS - GPU_SHARED_BLOCKS  # 956
    evict_count         = 0 if args.no_evict else args.evict
    total_sessions      = args.fill + evict_count
    unique_total        = total_sessions * args.turns * UNIQUE_BLOCKS_TURN
    gpu_saturated       = unique_total > gpu_free_blocks

    chip_blocks        = int(KV_OFFLOAD_GB * 1024 / 4)  # 72 GB / 4 MB per block
    blocks_per_session = args.turns * UNIQUE_BLOCKS_TURN
    total_fill_blocks  = args.fill * blocks_per_session
    chip_saturation_pct = total_fill_blocks / chip_blocks * 100

    label_str = f"  [{args.label}]" if args.label else ""
    print(f"\nClaude Code KV Offload Stress Test{label_str}")
    print(f"  base_url : {base_url}")
    print(f"  model    : {MODEL}")
    print(f"  fill={args.fill}  evict={evict_count}  recall={args.recall}  turns={args.turns}")
    print(f"  proxy    : {'yes — TTFT measured' if use_proxy else 'no'}")
    print(f"  unique blocks (fill+evict): ~{unique_total}  GPU free budget: {gpu_free_blocks}")
    if not gpu_saturated:
        min_turns = (gpu_free_blocks // (total_sessions * UNIQUE_BLOCKS_TURN)) + 1
        min_sessions = (gpu_free_blocks // (args.turns * UNIQUE_BLOCKS_TURN)) + 1
        print(f"  WARNING: GPU not saturated — fill blocks stay in GPU, RECALL will show CPU→GPU = 0!")
        print(f"           Fix: use --turns {min_turns}  OR  --fill {min_sessions//2} --evict {min_sessions//2}")
    else:
        evicted = unique_total - gpu_free_blocks
        print(f"  GPU saturated: ~{evicted} fill blocks evicted to offload tier during evict phase")
    print(f"  chip fill (fill only): ~{chip_saturation_pct:.1f}% of {KV_OFFLOAD_GB} GB tier")

    try:
        urllib.request.urlopen(f"{DYNAMO_URL}/health", timeout=5)
    except Exception:
        print(f"\nERROR: Dynamo not reachable at {DYNAMO_URL}", file=sys.stderr)
        sys.exit(1)

    if use_proxy:
        try:
            urllib.request.urlopen(f"{PROXY_URL}/health", timeout=3)
        except Exception:
            print(f"WARNING: proxy not reachable at {PROXY_URL} — running without TTFT",
                  file=sys.stderr)
            use_proxy = False
            base_url  = DYNAMO_URL

    log_offset = 0
    try:
        with open(WORKER_LOG, "r") as f:
            f.seek(0, 2)
            log_offset = f.tell()
    except FileNotFoundError:
        pass

    fill_tasks   = FILL_TASKS[:args.fill]
    evict_tasks  = EVICT_TASKS[:args.evict]
    recall_tasks = fill_tasks[:args.recall]

    results = {}

    # Phase 1 — Fill
    fill_elapsed, _, log_offset, session_ids = run_fill_phase(
        fill_tasks, args.turns, base_url, MODEL, log_offset, use_proxy)
    results["fill"] = fill_elapsed

    # Phase 2 — Evict
    if not args.no_evict:
        evict_elapsed, _, log_offset = run_evict_phase(
            evict_tasks, args.turns, base_url, MODEL, log_offset, use_proxy)
        results["evict"] = evict_elapsed

    # Phase 3 — Recall
    recall_session_ids = session_ids[:args.recall]
    recall_elapsed, recall_delta, log_offset = run_recall_phase(
        recall_tasks, args.turns, base_url, MODEL, log_offset,
        use_proxy, recall_session_ids)
    results["recall"] = recall_elapsed

    # ── summary ───────────────────────────────────────────────────
    label_hdr = f"  [{args.label}]" if args.label else ""
    print(f"\n{'═'*55}")
    print(f"  SUMMARY{label_hdr}")
    print(f"{'═'*55}")
    for name, elapsed in results.items():
        print(f"  {name:8s} : mean elapsed {statistics.mean(elapsed):.1f}s")

    recall_bw = recall_delta.get("_recall_bw_gb_s", 0.0)
    if recall_bw > 0:
        print(f"\n  Recall DMA BW  : {recall_bw:.1f} GB/s  (bytes / actual DMA time from log)")
        print(f"  Theoretical    : CPU DRAM ~55 GB/s | xmem ~1400 GB/s")

    if results.get("fill") and results.get("recall"):
        overhead = statistics.mean(results["recall"]) - statistics.mean(results["fill"])
        sign = "+" if overhead >= 0 else ""
        print(f"\n  Elapsed overhead (recall - fill): {sign}{overhead:.1f}s")
        if overhead < -0.5:
            print(f"  ✓ RECALL is faster — prefix cache hits from tier working")
        elif overhead > 0.5:
            print(f"  ✗ RECALL is slower — unexpected, check logs")
        else:
            print(f"  ~ No significant difference")

    summary = {
        "label":      args.label,
        "fill":       args.fill,
        "evict":      args.evict,
        "recall":     args.recall,
        "turns":      args.turns,
        "fill_mean_s":   statistics.mean(results["fill"])   if results.get("fill")   else None,
        "evict_mean_s":  statistics.mean(results["evict"])  if results.get("evict")  else None,
        "recall_mean_s": statistics.mean(results["recall"]) if results.get("recall") else None,
        "recall_bw_gb_s": recall_bw,
    }
    if args.save:
        Path(args.save).write_text(json.dumps(summary, indent=2))
        print(f"\n  Saved summary → {args.save}")

    if use_proxy and LOG_FILE.exists():
        print(f"\n  Full TTFT data : {LOG_FILE}")
        print(f"  Sessions saved : {SESSION_FILE}")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()
