"""
Drive codex on a single SWE-bench Verified problem against a local vLLM
server, parsing `codex exec --json` events into per-turn records.

Per-turn fields recorded (one JSONL row per codex turn):
  isl                 : input_tokens (this turn)
  osl                 : output_tokens (this turn)
  isl_new             : input_tokens - cached_input_tokens
  isl_cached          : cached_input_tokens
  cache_hit_rate      : isl_cached / isl
  reasoning_output_tokens
  category            : text_only | tool_only | mixed | empty
  num_tool_calls      : count of command_execution / file_change / mcp_tool_call items
  num_text_blocks     : count of agent_message items

Codex stream-json events of interest (one event per line):
  thread.started          { thread_id }
  turn.started            {}
  item.started / .completed  { item: {item_type: ..., ...} }
  turn.completed          { usage: {input_tokens, cached_input_tokens,
                                    output_tokens, reasoning_output_tokens} }

Agent: codex   |   Backend: vLLM (OpenAI Responses API on port 8000)

The codex client must be configured via ~/.codex/config.toml to point at vLLM.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


PROMPT_TEMPLATE = """You are working on a real software-engineering bug from \
SWE-bench Verified. Solve it by editing files in this repository.

Repository: {repo}
Base commit: {base_commit}

# Problem statement
{problem_statement}

# Instructions
- The repo is checked out in your working directory at the base commit.
- Read relevant files, understand the bug, then make code edits to fix it.
- You may run shell commands to explore and validate.
- When you believe the fix is complete, summarize what you changed and stop.
"""


# Item types that count as tool actions vs textual reasoning.
TOOL_ITEM_TYPES = {
    "command_execution",
    "file_change",
    "mcp_tool_call",
    "web_search",
}
TEXT_ITEM_TYPES = {
    "agent_message",
    "reasoning",
}


def _categorize(item_types: list[str]) -> str:
    has_text = any(t in TEXT_ITEM_TYPES for t in item_types)
    has_tool = any(t in TOOL_ITEM_TYPES for t in item_types)
    if has_text and has_tool:
        return "mixed"
    if has_tool:
        return "tool_only"
    if has_text:
        return "text_only"
    return "empty"


def consume_stream(
    proc: subprocess.Popen[str],
    instance_id: str,
    usage_out: Path,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "instance_id": instance_id,
        "turns": 0,
        "categories": {"text_only": 0, "tool_only": 0, "mixed": 0, "empty": 0},
        "isl_total": 0,
        "osl_total": 0,
        "thread_id": None,
        "result": None,
    }
    assert proc.stdout is not None

    # Items observed since last turn.started.
    pending_item_types: list[str] = []
    pending_tool_count = 0
    pending_text_count = 0
    turn_index = 0

    with usage_out.open("a") as out:
        for raw in proc.stdout:
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            et = ev.get("type")

            if et == "thread.started":
                summary["thread_id"] = ev.get("thread_id")
                continue
            if et == "turn.started":
                pending_item_types = []
                pending_tool_count = 0
                pending_text_count = 0
                continue
            if et in ("item.started", "item.completed"):
                # Use item_type from completed events to avoid double counting.
                if et == "item.completed":
                    item = ev.get("item") or {}
                    it = item.get("item_type") or item.get("type")
                    if it:
                        pending_item_types.append(it)
                        if it in TOOL_ITEM_TYPES:
                            pending_tool_count += 1
                        elif it in TEXT_ITEM_TYPES:
                            pending_text_count += 1
                continue
            if et == "turn.completed":
                usage = ev.get("usage") or {}
                inp = int(usage.get("input_tokens") or 0)
                cached = int(usage.get("cached_input_tokens") or 0)
                out_tok = int(usage.get("output_tokens") or 0)
                reason_tok = int(usage.get("reasoning_output_tokens") or 0)
                isl = inp
                isl_new = max(0, inp - cached)
                cat = _categorize(pending_item_types)
                rate = (cached / isl) if isl > 0 else 0.0

                row = {
                    "ts": time.time(),
                    "instance_id": instance_id,
                    "agent": "codex",
                    "turn": turn_index,
                    "category": cat,
                    "isl": isl,
                    "osl": out_tok,
                    "isl_new": isl_new,
                    "isl_cached": cached,
                    "cache_hit_rate": round(rate, 4),
                    "input_tokens": inp,
                    "cached_input_tokens": cached,
                    "output_tokens": out_tok,
                    "reasoning_output_tokens": reason_tok,
                    "num_tool_calls": pending_tool_count,
                    "num_text_blocks": pending_text_count,
                    "item_types": pending_item_types,
                }
                out.write(json.dumps(row) + "\n")
                out.flush()

                summary["turns"] += 1
                summary["categories"][cat] += 1
                summary["isl_total"] += isl
                summary["osl_total"] += out_tok
                turn_index += 1
                # Reset for next turn
                pending_item_types = []
                pending_tool_count = 0
                pending_text_count = 0
                continue
            if et == "turn.failed" or et == "error":
                summary["result"] = {"failed_event": ev}
                continue
            # Other events (item.* during streaming, etc.) ignored.

    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--base-commit", required=True)
    ap.add_argument("--problem-statement", required=True)
    ap.add_argument("--model", required=True, help="model name as known by vllm")
    ap.add_argument("--base-url", required=True,
                    help="OpenAI-compatible base URL (vLLM /v1)")
    ap.add_argument("--workdir-root", required=True, type=Path)
    ap.add_argument("--usage-out", required=True, type=Path)
    ap.add_argument("--max-turns", type=int, default=15,
                    help="codex sets via -c approval_policy / hard cap")
    ap.add_argument("--timeout-secs", type=int, default=900)
    args = ap.parse_args()

    args.workdir_root.mkdir(parents=True, exist_ok=True)
    args.usage_out.parent.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=f"{args.instance_id}.", dir=args.workdir_root))
    repo_dir = workdir / "repo"

    clone_url = f"https://github.com/{args.repo}.git"
    try:
        subprocess.run(
            ["git", "clone", "--quiet", clone_url, str(repo_dir)],
            check=True, timeout=600,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "checkout", "--quiet", args.base_commit],
            check=True, timeout=120,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(json.dumps({
            "instance_id": args.instance_id,
            "error": f"clone_failed: {e!r}",
        }))
        shutil.rmtree(workdir, ignore_errors=True)
        return 1

    prompt = PROMPT_TEMPLATE.format(
        repo=args.repo,
        base_commit=args.base_commit,
        problem_statement=args.problem_statement,
    )

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY", "dummy-local")
    env["OPENAI_BASE_URL"] = args.base_url

    cmd = [
        "codex", "exec",
        "--json",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C", str(repo_dir),
        "-m", args.model,
        # Enforce vLLM provider in case ~/.codex/config.toml is missing.
        "-c", f'model_provider="vllm"',
        "-c", f'model_providers.vllm.name="vLLM (local)"',
        "-c", f'model_providers.vllm.base_url="{args.base_url}"',
        "-c", f'model_providers.vllm.wire_api="responses"',
        "-c", f'model_providers.vllm.env_key="OPENAI_API_KEY"',
        "-c", f'model_providers.vllm.stream_idle_timeout_ms=600000',
        "-c", f'model_providers.vllm.request_max_retries=0',
        "-c", f'model_providers.vllm.stream_max_retries=0',
        prompt,
    ]

    started = time.time()
    proc = subprocess.Popen(
        cmd, env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    timed_out = False
    try:
        summary = consume_stream(proc, args.instance_id, args.usage_out)
        try:
            proc.wait(timeout=max(1, args.timeout_secs - int(time.time() - started)))
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    finally:
        if proc.stderr is not None:
            err_tail = proc.stderr.read()[-2000:]
            if err_tail.strip():
                print(f"[codex stderr {args.instance_id}] {err_tail}", file=sys.stderr)
        shutil.rmtree(workdir, ignore_errors=True)

    print(json.dumps({
        "instance_id": args.instance_id,
        "agent": "codex",
        "elapsed_s": round(time.time() - started, 2),
        "timed_out": timed_out,
        "exit_code": proc.returncode,
        **{k: summary.get(k) for k in (
            "turns", "categories", "isl_total", "osl_total", "thread_id", "result",
        )},
    }))
    return 0 if not timed_out else 124


if __name__ == "__main__":
    raise SystemExit(main())
