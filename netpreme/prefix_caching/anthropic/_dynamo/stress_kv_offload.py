#!/usr/bin/env python3
"""
KV cache CPU offload stress test — Claude Code simulation mode.

Drives three phases to confirm CPU offloading is working:

  Phase 1 — FILL
    Run N multi-turn sessions, each with a long shared system prompt +
    growing conversation history.  Every turn's prompt is the full prior
    context + a new user message, just like Claude Code sends.
    This fills GPU KV cache.  A shared ~500-token system prompt prefix
    is reused across all sessions (GPU prefix-cache hit after the first).

  Phase 2 — EVICT
    Run M new sessions with different per-session content to push Phase 1
    blocks out of GPU into CPU tier (GPU→CPU bytes should appear).

  Phase 3 — RECALL
    Re-send a new turn in Phase 1 sessions so the scheduler reloads
    their blocks from CPU (CPU→GPU bytes should appear).

  [Legacy mode: --mode independent]
    Original behaviour — independent single-turn requests per seed.

Usage:
    python3 stress_kv_offload.py                            # defaults (claude-code mode)
    python3 stress_kv_offload.py --mode independent         # old behaviour
    python3 stress_kv_offload.py --fill 8 --evict 10 --recall 4
    python3 stress_kv_offload.py --url http://localhost:8000
    python3 stress_kv_offload.py --no-evict                 # GPU-hit baseline only

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
DEFAULT_WORKER_LOG = "/tmp/dynamo_worker_8000.log"

# Claude Code sends a ~8-10K token system prompt.  We approximate with
# a ~600-token constant block that every session shares.
SYSTEM_PROMPT_TOKENS = 600
# Per-session unique context per turn (simulates user messages + code snippets)
TOKENS_PER_TURN  = 400
NUM_TURNS        = 4       # turns per fill session
FOLLOWUP         = "Summarise the above in one sentence."

# Legacy mode (original test)
TOKENS_PER_PREFIX  = 1500

# ── helpers ───────────────────────────────────────────────────────

def _words(seed: int, n: int) -> str:
    return " ".join(f"w{seed}_{i}" for i in range(n))


def make_system_prompt(tokens: int = SYSTEM_PROMPT_TOKENS) -> str:
    """Constant shared prefix — like the Claude Code system prompt."""
    base = (
        "You are a highly capable software-engineering assistant embedded in "
        "Claude Code.  You have access to the user's filesystem, can read and "
        "edit files, run commands, and search codebases.  "
    )
    # Pad to target token count by repeating a filler sentence
    filler = "Follow the user's instructions carefully and respond concisely. "
    approx_words_per_token = 0.75
    target_words = int(tokens * approx_words_per_token)
    while len(base.split()) < target_words:
        base += filler
    return base.strip()


def make_turn_content(session_id: int, turn: int, tokens: int = TOKENS_PER_TURN) -> str:
    """Unique per-session, per-turn content — like user code/context."""
    prefix = (
        f"Session {session_id}, turn {turn}. "
        f"Here is the file content for this task: "
    )
    # Generate stable pseudo-unique words for this session+turn
    seed = session_id * 100 + turn
    filler = " ".join([f"tok{seed}_{i}" for i in range(tokens)]) + "."
    return prefix + filler


def make_prefix(seed: int, target_tokens: int = TOKENS_PER_PREFIX) -> str:
    """Legacy: independent single-turn prefix for mode=independent."""
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
        "max_tokens": 32, "stream": True,
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
    response_text = ""
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
                # Collect response text for multi-turn simulation
                if etype == "content_block_delta":
                    delta = ev.get("delta", {})
                    if delta.get("type") == "text_delta":
                        response_text += delta.get("text", "")
    except (urllib.error.URLError, TimeoutError) as e:
        print(f"  [request error: {e}]", file=sys.stderr)
    return ttft, response_text


def _last_lru_size_from_log(log_path: str) -> int:
    """Read the most recent lru_size value from CPU_LOOKUP log lines."""
    try:
        with open(log_path, "r", errors="replace") as f:
            text = f.read()
        matches = re.findall(r"lru_size=(\d+)", text)
        if matches:
            return int(matches[-1])
    except FileNotFoundError:
        pass
    return 0


def read_log_delta(log_path: str, offset: int,
                   wait: bool = False,
                   timeout: float = 60.0) -> tuple[dict, int]:
    """Read KV transfer metrics from log since offset.

    If wait=True, blocks until a metric line appears or the worker looks dead
    (timeout seconds with no new log activity). timeout is a safety net only —
    under normal operation the function returns as soon as vLLM writes its next
    10-second metric line.
    """
    deadline = time.time() + timeout
    last_eof = -1
    while True:
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
        if delta:
            return delta, new_offset
        if not wait or time.time() >= deadline:
            return delta, new_offset
        # reset deadline if log is still growing (worker is alive)
        if eof != last_eof:
            deadline = time.time() + timeout
            last_eof = eof
        time.sleep(1.0)


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


# ── Claude Code simulation helpers ────────────────────────────────

def build_claude_code_session(session_id: int,
                               system_prompt: str,
                               num_turns: int,
                               history: list[dict] | None = None
                               ) -> tuple[list[dict], list[dict]]:
    """
    Build the full message list for a new turn in a Claude Code-like session.

    Returns:
        (messages, updated_history)
        messages   = what to send to /v1/messages
        updated_history = full conversation so far (for subsequent turns)
    """
    if history is None:
        history = []

    turn = len(history) // 2  # each turn = user + assistant
    user_content = make_turn_content(session_id, turn)

    messages = history + [{"role": "user", "content": user_content}]
    return messages, messages  # history updated after we get the response


# ── phase runners ──────────────────────────────────────────────────

def run_phase_claude_code(
    name: str,
    session_ids: list[int],
    system_prompt: str,
    num_turns: int,
    url: str,
    model: str,
    log_path: str,
    log_offset: int,
    existing_histories: dict | None = None,
) -> tuple[list[float], dict, int, dict]:
    """
    Run a phase where each session_id gets num_turns of multi-turn conversation.

    Returns:
        (ttfts, cpu_delta, log_offset, session_histories)
        session_histories: dict[session_id -> message history]
    """
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"  ({len(session_ids)} sessions × {num_turns} turns each = "
          f"{len(session_ids)*num_turns} total requests)")
    print(f"  System prompt: ~{SYSTEM_PROMPT_TOKENS} tokens (shared prefix)")
    print(f"{'─'*60}")

    # Advance log offset to only capture this phase
    _, log_offset = read_log_delta(log_path, log_offset)

    histories: dict[int, list[dict]] = (
        {sid: list(h) for sid, h in existing_histories.items()}
        if existing_histories else {}
    )

    ttfts = []
    total = len(session_ids) * num_turns
    idx = 0
    for sid in session_ids:
        history = histories.get(sid, [])
        for turn in range(num_turns):
            idx += 1
            messages, _ = build_claude_code_session(sid, system_prompt, num_turns, history)

            # system prompt goes in the "system" field (Anthropic API)
            payload = json.dumps({
                "model": model,
                "system": system_prompt,
                "messages": messages,
                "max_tokens": 32,
                "stream": True,
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
            response_text = ""
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:
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
                        if etype == "content_block_delta":
                            delta_ev = ev.get("delta", {})
                            if delta_ev.get("type") == "text_delta":
                                response_text += delta_ev.get("text", "")
            except (urllib.error.URLError, TimeoutError) as e:
                print(f"  [request error: {e}]", file=sys.stderr)

            if ttft is not None:
                ttfts.append(ttft)
                print(f"  [{idx:3d}/{total}] session={sid:3d} turn={turn}  TTFT {ttft:6.0f} ms"
                      f"  context={len(messages)} msgs")
            else:
                print(f"  [{idx:3d}/{total}] session={sid:3d} turn={turn}  ERROR")

            # Update history for next turn (include assistant response)
            history = messages + [{"role": "assistant", "content": response_text or "OK."}]
        histories[sid] = history

    # Wait up to 15s for the 10-second metric cadence to fire
    cpu_delta, log_offset = read_log_delta(log_path, log_offset, wait=True)
    g2c = cpu_delta.get("GPU_to_CPU_total_bytes", 0.0)
    c2g = cpu_delta.get("CPU_to_GPU_total_bytes", 0.0)

    print(f"\n  TTFT     : {fmt(ttfts)}")
    if g2c > 0 or c2g > 0:
        print(f"  GPU→CPU  : {fmt_bytes(g2c)}")
        print(f"  CPU→GPU  : {fmt_bytes(c2g)}")
    else:
        print(f"  CPU offload: none detected this phase")

    return ttfts, cpu_delta, log_offset, histories


def run_phase_claude_code_recall(
    name: str,
    session_ids: list[int],
    system_prompt: str,
    histories: dict,
    url: str,
    model: str,
    log_path: str,
    log_offset: int,
) -> tuple[list[float], dict, int]:
    """
    Send one new turn to existing sessions to trigger CPU→GPU recall.
    """
    print(f"\n{'─'*60}")
    print(f"  {name}")
    print(f"  ({len(session_ids)} sessions, 1 new turn each = {len(session_ids)} requests)")
    print(f"{'─'*60}")

    _, log_offset = read_log_delta(log_path, log_offset)

    ttfts = []
    for i, sid in enumerate(session_ids):
        history = histories.get(sid, [])
        turn = len(history) // 2
        user_content = make_turn_content(sid, turn)
        messages = history + [{"role": "user", "content": user_content}]

        payload = json.dumps({
            "model": model,
            "system": system_prompt,
            "messages": messages,
            "max_tokens": 32,
            "stream": True,
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
            with urllib.request.urlopen(req, timeout=120) as resp:
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
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  [request error: {e}]", file=sys.stderr)

        if ttft is not None:
            ttfts.append(ttft)
            print(f"  [{i+1:3d}/{len(session_ids)}] session={sid:3d}  TTFT {ttft:6.0f} ms"
                  f"  context_turns={turn}")
        else:
            print(f"  [{i+1:3d}/{len(session_ids)}] session={sid:3d}  ERROR")

    # Wait up to 15s for the metric cadence
    cpu_delta, log_offset = read_log_delta(log_path, log_offset, wait=True)
    g2c = cpu_delta.get("GPU_to_CPU_total_bytes", 0.0)
    c2g = cpu_delta.get("CPU_to_GPU_total_bytes", 0.0)

    print(f"\n  TTFT     : {fmt(ttfts)}")
    if g2c > 0 or c2g > 0:
        print(f"  GPU→CPU  : {fmt_bytes(g2c)}")
        print(f"  CPU→GPU  : {fmt_bytes(c2g)}")
    else:
        print(f"  CPU offload: none detected this phase")

    return ttfts, cpu_delta, log_offset


# ── legacy independent-mode phase runner ──────────────────────────

def run_phase_independent(name: str, seeds: list[int], url: str, model: str,
              log_path: str, log_offset: int) -> tuple[list[float], dict, int]:
    print(f"\n{'─'*55}")
    print(f"  {name}  ({len(seeds)} requests)")
    print(f"{'─'*55}")

    _, log_offset = read_log_delta(log_path, log_offset)

    ttfts = []
    for i, seed in enumerate(seeds):
        prefix = make_prefix(seed)
        msgs = [
            {"role": "user",      "content": prefix},
            {"role": "assistant", "content": "Understood."},
            {"role": "user",      "content": FOLLOWUP},
        ]
        t, _ = ttft_request(url, model, msgs)
        if t is not None:
            ttfts.append(t)
            print(f"  [{i+1:3d}/{len(seeds)}] seed={seed:5d}  TTFT {t:6.0f} ms")
        else:
            print(f"  [{i+1:3d}/{len(seeds)}] seed={seed:5d}  ERROR")

    cpu_delta, log_offset = read_log_delta(log_path, log_offset, wait=True)
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
    parser.add_argument("--mode",     choices=["claude-code", "independent"],
                        default="claude-code",
                        help="claude-code: multi-turn sessions (default); "
                             "independent: single-turn per seed (legacy)")
    parser.add_argument("--fill",     type=int, default=12,
                        help="sessions in Fill phase (claude-code) or "
                             "requests in Fill phase (independent) [default 12/20]")
    parser.add_argument("--turns",    type=int, default=NUM_TURNS,
                        help=f"turns per session in claude-code mode [default {NUM_TURNS}]")
    parser.add_argument("--evict",    type=int, default=15,
                        help="sessions/requests in Evict phase [default 15/25]")
    parser.add_argument("--recall",   type=int, default=6,
                        help="Fill sessions/requests to replay in Recall [default 6/15]")
    parser.add_argument("--no-evict", action="store_true",
                        help="Skip eviction — GPU-hit baseline only")
    args = parser.parse_args()

    print(f"\nKV Offload Stress Test  [{args.mode} mode]")
    print(f"  endpoint : {args.url}")
    print(f"  model    : {args.model}")
    print(f"  log      : {args.log}")
    if args.mode == "claude-code":
        print(f"  fill={args.fill} sessions × {args.turns} turns  "
              f"evict={0 if args.no_evict else args.evict} sessions  "
              f"recall={args.recall} sessions")
        print(f"  system_prompt: ~{SYSTEM_PROMPT_TOKENS} tokens (shared)")
        print(f"  per_turn_tokens: ~{TOKENS_PER_TURN} tokens")
    else:
        print(f"  fill={args.fill}  evict={0 if args.no_evict else args.evict}  "
              f"recall={args.recall}")

    try:
        urllib.request.urlopen(f"{args.url}/health", timeout=5)
    except Exception:
        print(f"\nERROR: server not reachable at {args.url}", file=sys.stderr)
        sys.exit(1)

    log_offset = 0
    try:
        with open(args.log, "r") as f:
            f.seek(0, 2)
            log_offset = f.tell()
    except FileNotFoundError:
        pass

    results = {}

    if args.mode == "claude-code":
        system_prompt = make_system_prompt()
        fill_sessions  = list(range(args.fill))
        evict_sessions = list(range(100, 100 + args.evict))
        recall_sessions = fill_sessions[:args.recall]

        fill_ttfts, fill_delta, log_offset, fill_histories = run_phase_claude_code(
            "PHASE 1 — FILL (multi-turn sessions, blocks enter GPU + CPU)",
            fill_sessions, system_prompt, args.turns,
            args.url, args.model, args.log, log_offset,
        )
        results["fill"] = fill_ttfts
        # Snapshot LRU size after fill for overflow diagnostics
        fill_lru = _last_lru_size_from_log(args.log)
        results["fill_lru_size"] = fill_lru
        if fill_lru:
            blocks_per_session = fill_lru // max(1, args.fill)
            max_safe = (17408 - fill_lru) // max(1, blocks_per_session)
            print(f"  [LRU after fill: {fill_lru} / 17408 blocks  "
                  f"free={17408-fill_lru}  "
                  f"max_safe_evict≈{max_safe} sessions]")
        else:
            print(f"  [LRU diagnostic unavailable: no lru_size= entries in log]")

        if not args.no_evict:
            evict_ttfts, _, log_offset, _ = run_phase_claude_code(
                "PHASE 2 — EVICT (new sessions push Fill blocks to CPU)",
                evict_sessions, system_prompt, args.turns,
                args.url, args.model, args.log, log_offset,
            )
            results["evict"] = evict_ttfts

        recall_ttfts, recall_delta, log_offset = run_phase_claude_code_recall(
            "PHASE 3 — RECALL (resume Fill sessions — loads blocks from CPU)",
            recall_sessions, system_prompt, fill_histories,
            args.url, args.model, args.log, log_offset,
        )
        results["recall"] = recall_ttfts
        results["recall_c2g_bytes"] = recall_delta.get("CPU_to_GPU_total_bytes", 0.0)

    else:
        # Legacy independent mode
        fill_seeds   = list(range(args.fill))
        evict_seeds  = list(range(10000, 10000 + args.evict))
        recall_seeds = fill_seeds[:args.recall]

        fill_ttfts, _, log_offset = run_phase_independent(
            "PHASE 1 — FILL (cold prefill, blocks enter GPU)",
            fill_seeds, args.url, args.model, args.log, log_offset,
        )
        results["fill"] = fill_ttfts

        if not args.no_evict:
            evict_ttfts, _, log_offset = run_phase_independent(
                "PHASE 2 — EVICT (new prefixes push Fill blocks to CPU)",
                evict_seeds, args.url, args.model, args.log, log_offset,
            )
            results["evict"] = evict_ttfts

        recall_ttfts, recall_delta, log_offset = run_phase_independent(
            "PHASE 3 — RECALL (re-request Fill prefixes — loads from CPU)",
            recall_seeds, args.url, args.model, args.log, log_offset,
        )
        results["recall"] = recall_ttfts
        results["recall_c2g_bytes"] = recall_delta.get("CPU_to_GPU_total_bytes", 0.0)

    # ── summary ───────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  SUMMARY")
    print(f"{'═'*60}")
    for name, ttfts in results.items():
        if isinstance(ttfts, list):
            print(f"  {name:8s} : {fmt(ttfts)}")

    if results.get("fill") and results.get("recall"):
        overhead = statistics.mean(results["recall"]) - statistics.mean(results["fill"])
        sign = "+" if overhead >= 0 else ""
        recall_c2g = results.get("recall_c2g_bytes", 0.0)
        print(f"\n  TTFT overhead (recall - fill mean): {sign}{overhead:.0f} ms")
        print(f"  CPU→GPU bytes in recall: {fmt_bytes(recall_c2g)}")
        if recall_c2g > 0:
            print(f"  ✓ CPU offloading confirmed — blocks loaded from CPU tier")
        elif overhead > 200:
            print(f"  ✗ Recall is slow but CPU→GPU = 0 B")
            print(f"    Likely cause: evict phase overflowed the CPU LRU, evicting fill blocks")
            print(f"    Overhead is recompute penalty, NOT CPU load")
            print(f"    Fix: reduce --evict so fill blocks stay in LRU")
            # Estimate max safe evict sessions
            fill_lru = results.get("fill_lru_size", 0)
            if fill_lru:
                free = 17408 - fill_lru
                blocks_per_session = fill_lru // max(1, args.fill if hasattr(args, 'fill') else 15)
                max_evict = free // max(1, blocks_per_session)
                print(f"    LRU after fill: {fill_lru} / 17408 blocks")
                print(f"    Free slots: {free}  →  max safe evict ≈ {max_evict} sessions")
        else:
            print(f"  ✗ No clear overhead detected")
            print(f"    Check worker log for 'External prefix cache hit rate'")
            print(f"    If 0.0%: prefix hash mismatch (tokenizer/config issue)")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
