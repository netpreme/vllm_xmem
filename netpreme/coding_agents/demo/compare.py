#!/usr/bin/env python3
"""
Demo: compare hybrid-cpu (port 8001) vs hybrid-mtier (port 8002).

What this shows
───────────────
Mode A — cold→warm GPU prefix cache speedup (default, single-turn):
  Round 1 (cold):   full GPU prefill              → TTFT ~3,500ms
  Round 2 (warm):   KV already in GPU HBM         → TTFT  ~430ms   (~8× faster)
  Both CPU and MTier setups show the same speedup here — it's pure GPU caching.

Mode B — fill-recall: external KV bandwidth comparison:
  Phase 1 (fill):   flood GPU cache with competing sessions, evicting the
                    original session's KV blocks to external storage
                    (CPU DRAM on hybrid-cpu, MTier chip on hybrid-mtier)
  Phase 2 (recall): request the original session again
                    → TTFT depends on external KV recall bandwidth
    hybrid-cpu:   loads KV from CPU DRAM (DeviceToHost + HostToDevice over PCIe)
    hybrid-mtier: loads KV from MTier chip (DeviceToDevice in CUDA VA space)
  Expected: MTier recall is faster if DeviceToDevice > PCIe bandwidth

Mode C — high-concurrency capacity demo (--spill):
  At c=12+ with --spill, each session's KV overflows GPU HBM to external.
  MTier capacity (70 GB+) vs CPU DRAM (72 GB) — at very high concurrency,
  MTier's multi-bank config (4 × 70 GB = 280 GB) prevents evictions that
  CPU DRAM would hit, keeping external cache hit rate high.

Demo servers must be running: bash demo/start_servers.sh
  hybrid-cpu   port 8001  GPU 0
  hybrid-mtier port 8002  GPU 0 (MTier MUST use GPU 0 — cuMemCreate topology)

Usage:
    python3 demo/compare.py                          # cold→warm, c=4, 30K tokens
    python3 demo/compare.py -m fill-recall -r 2      # fill GPU cache, then recall from external
    python3 demo/compare.py -m single-turn --spill   # force external KV at each concurrency
    python3 demo/compare.py -c 12 -r 5               # 12 users, 5 rounds
    python3 demo/compare.py -s 0x<SESSION_ID>        # rerun warm (skip cold start)
"""
import argparse
import asyncio
import random
import time
import sys
from dataclasses import dataclass
from typing import Optional

import httpx

MODEL     = "qwen/qwen3-coder-30b-a3b-instruct-fp8"
CPU_URL   = "http://localhost:8001"
MTIER_URL = "http://localhost:8002"

# ── Colors ───────────────────────────────────────────────────────────────────
BLUE   = "\033[94m"
ORANGE = "\033[38;5;208m"
GREEN  = "\033[92m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"
YELLOW = "\033[93m"

# ── Prompt building ──────────────────────────────────────────────────────────
_CODE_BLOCK = """\
def process_batch(items, config):
    \"\"\"Process a batch of items using the provided configuration.\"\"\"
    results = []
    for item in items:
        if config.get("validate", True):
            if not validate_item(item):
                continue
        transformed = transform_item(item, config["transform_fn"])
        results.append(transformed)
    return results

def validate_item(item):
    return item is not None and isinstance(item, dict) and "id" in item

def transform_item(item, fn):
    return fn(item)

"""  # ~200 tokens

_FIXED_REPLY = "The `process_batch` function returns a list of transformed items."


def build_single_turn_prompt(target_tokens: int, req_id: int, session_id: int) -> list[dict]:
    repeats = max(1, target_tokens // 200)
    context = _CODE_BLOCK * repeats
    return [
        {
            "role": "system",
            "content": (
                f"[s:{session_id:08x}][r:{req_id:04d}] "
                "You are a senior Python engineer. "
                "Review the code below and answer the question concisely in one sentence."
            ),
        },
        {
            "role": "user",
            "content": f"Code:\n```python\n{context}\n```\n\n/no_think /no_think What does process_batch return?",
        },
    ]


def build_multi_turn_history(
    session_id: int,
    req_id: int,
    turn: int,
    tokens_per_turn: int = 1000,
) -> list[dict]:
    repeats = max(1, tokens_per_turn // 200)
    code = _CODE_BLOCK * repeats
    messages: list[dict] = [
        {
            "role": "system",
            "content": (
                f"[s:{session_id:08x}][r:{req_id:04d}] "
                "You are a senior Python engineer. Answer each question in one sentence."
            ),
        }
    ]
    for t in range(turn - 1):
        messages.append({
            "role": "user",
            "content": f"[turn {t+1}]\nCode:\n```python\n{code}\n```\n/no_think What does process_batch return?",
        })
        messages.append({"role": "assistant", "content": _FIXED_REPLY})
    messages.append({
        "role": "user",
        "content": f"[turn {turn}]\nCode:\n```python\n{code}\n```\n/no_think What does process_batch return?",
    })
    return messages


# ── Streaming request ────────────────────────────────────────────────────────
@dataclass
class Result:
    label:   str
    req_id:  int
    ttft_ms: Optional[float] = None
    e2e_ms:  Optional[float] = None
    tokens:  int = 0
    text:    str = ""
    error:   Optional[str] = None


async def send_request(
    client: httpx.AsyncClient,
    base_url: str,
    label: str,
    req_id: int,
    messages: list[dict],
) -> Result:
    result  = Result(label=label, req_id=req_id)
    t_start = time.perf_counter()
    t_first: Optional[float] = None
    try:
        async with client.stream(
            "POST",
            f"{base_url}/v1/chat/completions",
            json={
                "model":       MODEL,
                "messages":    messages,
                "stream":      True,
                "max_tokens":  500,
                "temperature": 0.0,
                "seed":        42,
            },
            timeout=120,
        ) as resp:
            resp.raise_for_status()
            async for raw_line in resp.aiter_lines():
                if not raw_line.startswith("data:"):
                    continue
                data = raw_line[5:].strip()
                if data == "[DONE]":
                    break
                import json
                try:
                    chunk = json.loads(data)
                except Exception:
                    continue
                delta = chunk["choices"][0]["delta"].get("content", "")
                if delta:
                    if t_first is None:
                        t_first = time.perf_counter()
                    result.text   += delta
                    result.tokens += 1
        t_end = time.perf_counter()
        result.ttft_ms = (t_first - t_start) * 1000 if t_first else None
        result.e2e_ms  = (t_end   - t_start) * 1000
    except Exception as e:
        result.error = str(e)
    return result


# ── Print helpers ────────────────────────────────────────────────────────────
def _bar(val_ms: float, ref_ms: float, width: int = 16) -> str:
    filled = round(val_ms / ref_ms * width) if ref_ms > 0 else width
    filled = max(1, min(filled, width))
    return "█" * filled + "░" * (width - filled)


def _avg(results: list[Result], attr: str) -> Optional[float]:
    vals = [getattr(r, attr) for r in results if getattr(r, attr) is not None]
    return sum(vals) / len(vals) if vals else None


def classify_hit(ttft_ms: float, label: str) -> str:
    if ttft_ms < 200:
        return "mtier-hit" if "mtier" in label else "cpu-hit"
    return "prefill"


def print_round_header(round_num: int, concurrency: int, context_tokens: int, mode: str = "single-turn") -> None:
    if mode == "multi-turn":
        note = f"turn {round_num}  context~{context_tokens // 1000}K tokens"
        print(f"\n{BOLD}━━━ Round {round_num}  ({concurrency} concurrent)  [{note}]{RESET}")
    else:
        print(f"\n{BOLD}━━━ Round {round_num}  ({concurrency} concurrent){RESET}")
    print(f"  {'Req':>3}  {'Setup':<14}  {'TTFT':>8}  {'E2E':>9}  {'Toks':>5}  {'Bar':16}  Hit?")
    print(f"  {'───':>3}  {'──────────────':<14}  {'────────':>8}  {'─────────':>9}  {'────':>5}  {'────────────────'}  ────────")


def print_result(r: Result, max_ttft: float) -> None:
    color = BLUE if "cpu" in r.label else ORANGE
    if r.error:
        print(f"  {r.req_id:>3}  {color}{r.label:<14}{RESET}  {'ERR':>8}  {'ERR':>9}  {'?':>5}  {'':16}  {r.error[:30]}")
        return

    ttft_str = f"{r.ttft_ms:>7.0f}ms" if r.ttft_ms else "       ?"
    e2e_str  = f"{r.e2e_ms:>8.0f}ms" if r.e2e_ms  else "        ?"
    bar      = _bar(r.ttft_ms or 0, max_ttft)

    hit = classify_hit(r.ttft_ms, r.label) if r.ttft_ms else "?"
    hit_color = {"prefill": DIM, "cpu-hit": BLUE, "mtier-hit": ORANGE}.get(hit, RESET)

    print(f"  {r.req_id:>3}  {color}{r.label:<14}{RESET}  "
          f"{ttft_str}  {e2e_str}  {r.tokens:>5}  "
          f"{color}{bar}{RESET}  {hit_color}{hit}{RESET}")


def print_round_summary(cpu_results: list[Result], mtier_results: list[Result],
                        label: str = "") -> None:
    cpu_ttft = _avg(cpu_results,   "ttft_ms")
    cpu_e2e  = _avg(cpu_results,   "e2e_ms")
    mt_ttft  = _avg(mtier_results, "ttft_ms")
    mt_e2e   = _avg(mtier_results, "e2e_ms")

    if not (cpu_ttft and mt_ttft):
        return

    ref = max(cpu_ttft, mt_ttft)
    header = f"  Average{f'  [{label}]' if label else ''}"
    print()
    print(f"  {DIM}{header:<34}{RESET}  {'TTFT':>8}   {'E2E':>9}")
    print(f"  {BLUE}{'hybrid-cpu':<14}{RESET}             {cpu_ttft:>7.0f}ms  {cpu_e2e:>8.0f}ms  {BLUE}{_bar(cpu_ttft, ref)}{RESET}")
    print(f"  {ORANGE}{'hybrid-mtier':<14}{RESET}             {mt_ttft:>7.0f}ms  {mt_e2e:>8.0f}ms  {ORANGE}{_bar(mt_ttft, ref)}{RESET}")
    print()

    speedup = cpu_ttft / mt_ttft
    sp_color = GREEN if speedup >= 1.5 else (YELLOW if speedup >= 1.1 else RESET)
    direction = "MTier faster" if speedup > 1.05 else ("CPU faster" if speedup < 0.95 else "similar")
    print(f"  {BOLD}MTier vs CPU TTFT:  {sp_color}{speedup:.2f}×  {direction}{RESET}")


# ── Single-turn round ────────────────────────────────────────────────────────
async def run_single_turn_round(
    context_tokens: int,
    concurrency: int,
    round_num: int,
    session_id: int,
) -> tuple[list[Result], list[Result]]:
    print_round_header(round_num, concurrency, context_tokens, mode="single-turn")

    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(concurrency):
            messages = build_single_turn_prompt(context_tokens, req_id=i, session_id=session_id)
            tasks.append(send_request(client, CPU_URL,   "hybrid-cpu",   i, messages))
            tasks.append(send_request(client, MTIER_URL, "hybrid-mtier", i, messages))
        all_results = await asyncio.gather(*tasks)

    cpu_results   = [r for r in all_results if "cpu"   == r.label.split("-")[1]]
    mtier_results = [r for r in all_results if "mtier" in r.label]

    all_ttft = [r.ttft_ms for r in all_results if r.ttft_ms]
    max_ttft = max(all_ttft) if all_ttft else 1000

    for i in range(concurrency):
        print_result(cpu_results[i],   max_ttft)
        print_result(mtier_results[i], max_ttft)
        if i < concurrency - 1:
            print()

    print_round_summary(cpu_results, mtier_results)
    return cpu_results, mtier_results


# ── Multi-turn round ─────────────────────────────────────────────────────────
async def run_multi_turn_round(
    turn: int,
    concurrency: int,
    session_id: int,
    tokens_per_turn: int,
) -> tuple[list[Result], list[Result]]:
    context_tokens = turn * tokens_per_turn
    print_round_header(turn, concurrency, context_tokens, mode="multi-turn")

    async with httpx.AsyncClient() as client:
        tasks = []
        for i in range(concurrency):
            messages = build_multi_turn_history(session_id, i, turn, tokens_per_turn)
            tasks.append(send_request(client, CPU_URL,   "hybrid-cpu",   i, messages))
            tasks.append(send_request(client, MTIER_URL, "hybrid-mtier", i, messages))
        all_results = await asyncio.gather(*tasks)

    cpu_results   = [r for r in all_results if "cpu"   == r.label.split("-")[1]]
    mtier_results = [r for r in all_results if "mtier" in r.label]

    all_ttft = [r.ttft_ms for r in all_results if r.ttft_ms]
    max_ttft = max(all_ttft) if all_ttft else 1000

    for i in range(concurrency):
        print_result(cpu_results[i],   max_ttft)
        print_result(mtier_results[i], max_ttft)
        if i < concurrency - 1:
            print()

    print_round_summary(cpu_results, mtier_results)
    return cpu_results, mtier_results


# ── Main ─────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser(description="MTier vs CPU KV cache demo")
    parser.add_argument("--concurrency",    "-c", type=int, default=4)
    parser.add_argument("--rounds",         "-r", type=int, default=3)
    parser.add_argument("--tokens",         "-t", type=int, default=30000)
    parser.add_argument("--tokens-per-turn",      type=int, default=1000)
    parser.add_argument("--mode",           "-m", choices=["single-turn", "multi-turn", "fill-recall"],
                        default="single-turn")
    parser.add_argument("--fill",                type=int, default=0,
                        help="fill-recall mode: number of unique-session fill rounds to run "
                             "before the recall rounds (auto-computed if 0)")
    parser.add_argument("--spill",               action="store_true",
                        help="Use tokens large enough to overflow GPU cache at given concurrency. "
                             "GPU capacity ~688K tokens; sets tokens = 688K/concurrency + 10%%. "
                             "Forces external cache loading on round 2+ → shows MTier vs CPU TTFT.")
    parser.add_argument("--session",        "-s", type=lambda x: int(x, 0), default=None,
                        help="Reuse session (hex OK, e.g. 0x17bf078e); omit for fresh cold start")
    parser.add_argument("--wait",           "-w", type=float, default=None,
                        help="Seconds to wait between rounds for cache writes (default: auto)")
    args = parser.parse_args()

    # --spill: auto-size tokens to overflow GPU cache at this concurrency
    if args.spill:
        GPU_TOKEN_CAPACITY = 687_888  # 80GB × 0.85 - 29GB model, 49152 bytes/token
        # Use 110% of per-session capacity to guarantee external cache pressure
        spill_tokens = int(GPU_TOKEN_CAPACITY / args.concurrency * 1.10)
        args.tokens = max(args.tokens, spill_tokens)
        print(f"  {YELLOW}--spill: using {args.tokens // 1000}K tokens to overflow GPU cache "
              f"({args.concurrency} × {args.tokens // 1000}K = "
              f"{args.concurrency * args.tokens // 1000}K > {GPU_TOKEN_CAPACITY // 1000}K GPU capacity){RESET}")

    session_id = args.session if args.session is not None else random.getrandbits(32)
    # Auto wait: give more time at higher concurrency for async cache writes to flush
    wait_secs = args.wait if args.wait is not None else max(2.0, args.concurrency * 0.4)

    # fill-recall: compute fill rounds needed to overflow GPU KV cache
    fill_rounds = args.fill
    if args.mode == "fill-recall" and fill_rounds == 0:
        # GPU capacity ≈ (80GB × 0.85 - 30GB model weights) / 49KB per token ≈ 750K tokens
        # Conservative: use ~40% of total GPU memory for KV
        gpu_token_capacity = int(0.40 * 80e9 / 49_152)
        tokens_per_fill_round = args.concurrency * args.tokens
        fill_rounds = max(3, (gpu_token_capacity // tokens_per_fill_round) + 3)

    print(f"\n{BOLD}MTier vs CPU — KV Cache Recall Demo{RESET}")
    print(f"  {BLUE}blue   = hybrid-cpu   (port 8001){RESET}")
    print(f"  {ORANGE}orange = hybrid-mtier (port 8002){RESET}")
    print(f"  mode={args.mode}  concurrency={args.concurrency}  rounds={args.rounds}  "
          f"session=0x{session_id:08x}")
    if args.mode == "fill-recall":
        print(f"  context~{args.tokens // 1000}K tokens  fill={fill_rounds} rounds  "
              f"then {args.rounds} recall round(s)")
        print(f"  {DIM}Fill saturates GPU cache, forcing recalls from external KV (CPU DRAM vs MTier){RESET}")
    elif args.mode == "single-turn":
        print(f"  context~{args.tokens // 1000}K tokens  wait={wait_secs:.0f}s between rounds")
    else:
        tpt = args.tokens_per_turn
        print(f"  {tpt} tokens/turn → ~{args.rounds * tpt // 1000}K tokens by round {args.rounds}")
    print(f"  {DIM}Grafana: http://localhost:3000{RESET}")
    if args.mode != "fill-recall":
        print(f"  {DIM}Rerun warm:  python3 compare.py -s 0x{session_id:08x} -m {args.mode}{RESET}")

    async with httpx.AsyncClient() as client:
        for url, label in [(CPU_URL, "CPU (8001)"), (MTIER_URL, "MTier (8002)")]:
            try:
                r = await client.get(f"{url}/health", timeout=3)
                r.raise_for_status()
                print(f"  ✓ {label} healthy")
            except Exception as e:
                print(f"  ✗ {label} not ready: {e}", file=sys.stderr)
                sys.exit(1)

    all_cpu: list[Result]   = []
    all_mtier: list[Result] = []

    if args.mode == "fill-recall":
        # Phase 1: fill GPU KV cache with unique sessions so original session is evicted to external
        print(f"\n{BOLD}━━━ Phase 1: Fill ({fill_rounds} rounds, unique sessions) ━━━{RESET}")
        print(f"  {DIM}Saturating GPU KV cache to evict original session to external cache…{RESET}")
        for fill_i in range(fill_rounds):
            fill_session = random.getrandbits(32)
            pct = (fill_i + 1) / fill_rounds * 100
            bar_w = 40
            filled = round(pct / 100 * bar_w)
            bar = "█" * filled + "░" * (bar_w - filled)
            print(f"  [{bar}] {fill_i+1:3d}/{fill_rounds}  session=0x{fill_session:08x}",
                  end="\r", flush=True)
            async with httpx.AsyncClient() as client:
                tasks = []
                for i in range(args.concurrency):
                    messages = build_single_turn_prompt(args.tokens, req_id=i, session_id=fill_session)
                    tasks.append(send_request(client, CPU_URL,   "hybrid-cpu",   i, messages))
                    tasks.append(send_request(client, MTIER_URL, "hybrid-mtier", i, messages))
                await asyncio.gather(*tasks)
            await asyncio.sleep(max(1.0, args.concurrency * 0.2))
        print(f"\n  {DIM}Fill complete. Original session 0x{session_id:08x} should now be in external cache.{RESET}")
        await asyncio.sleep(3)

        # Phase 2: recall rounds against original session (now in external cache)
        print(f"\n{BOLD}━━━ Phase 2: External KV Recall (session 0x{session_id:08x}) ━━━{RESET}")
        print(f"  {DIM}Session evicted from GPU HBM → KV now in external cache{RESET}")
        print(f"  {BLUE}hybrid-cpu:   KV in CPU DRAM  (PCIe DeviceToHost/HostToDevice){RESET}")
        print(f"  {ORANGE}hybrid-mtier: KV in MTier chip (CUDA DeviceToDevice){RESET}\n")
        recall_cpu: list[Result]   = []
        recall_mtier: list[Result] = []
        for round_num in range(1, args.rounds + 1):
            cpu_r, mt_r = await run_single_turn_round(
                args.tokens, args.concurrency, round_num, session_id)
            all_cpu.extend(cpu_r)
            all_mtier.extend(mt_r)
            recall_cpu.extend(cpu_r)
            recall_mtier.extend(mt_r)
            if round_num < args.rounds:
                print(f"  {DIM}Waiting {wait_secs:.0f}s…{RESET}")
                await asyncio.sleep(wait_secs)
        print(f"\n{BOLD}━━━ Fill-Recall Summary ━━━{RESET}")
        print_round_summary(recall_cpu, recall_mtier, label="external KV recall")

    elif args.mode == "single-turn":
        for round_num in range(1, args.rounds + 1):
            cpu_r, mt_r = await run_single_turn_round(
                args.tokens, args.concurrency, round_num, session_id)
            all_cpu.extend(cpu_r)
            all_mtier.extend(mt_r)
            if round_num < args.rounds:
                print(f"  {DIM}Waiting {wait_secs:.0f}s…{RESET}")
                await asyncio.sleep(wait_secs)
    else:
        for turn in range(1, args.rounds + 1):
            cpu_r, mt_r = await run_multi_turn_round(
                turn, args.concurrency, session_id, args.tokens_per_turn)
            all_cpu.extend(cpu_r)
            all_mtier.extend(mt_r)
            if turn < args.rounds:
                print(f"  {DIM}Waiting {wait_secs:.0f}s…{RESET}")
                await asyncio.sleep(wait_secs)

    if args.rounds > 1 and args.mode != "fill-recall":
        print(f"\n{BOLD}━━━ Overall ({args.rounds} rounds × {args.concurrency} requests) ━━━{RESET}")
        print_round_summary(all_cpu, all_mtier)

    ok_cpu   = [r for r in all_cpu   if not r.error and r.text]
    ok_mtier = [r for r in all_mtier if not r.error and r.text]
    if ok_cpu and ok_mtier:
        print(f"\n{DIM}Generated output (req 0, last round):{RESET}")
        print(f"  {BLUE}CPU  :{RESET} {ok_cpu[-1].text[:120]}")
        print(f"  {ORANGE}MTier:{RESET} {ok_mtier[-1].text[:120]}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
