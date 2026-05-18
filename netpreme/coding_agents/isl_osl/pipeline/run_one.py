"""
Run claude -p on a single SWE-bench Verified problem.

Per-call ISL/OSL tracking is done by the sibling proxy.py process — claude
points at the proxy via ANTHROPIC_BASE_URL and the proxy tees one row per
/v1/messages call to usage.jsonl. This module's only job is to launch claude,
clone the repo, stream its stdout for diagnostics, and emit a per-problem
summary on stdout.

The instance_id is forwarded to the proxy via the X-Instance-Id header so
each row carries problem provenance. Categorization (text_only / tool_only /
mixed / empty) is computed inside the proxy on the actual API response.
"""
from __future__ import annotations

import argparse
import csv
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


# CSV schema matches what pipeline/proxy.py writes, so analysis scripts work
# unchanged regardless of which backend produced the run.
CSV_COLUMNS = [
    "ts", "instance_id", "elapsed_ms", "ttft_ms", "decode_ms", "itl_ms",
    "isl", "osl", "isl_new", "isl_cached", "cache_hit_rate",
    "stop_reason", "category", "num_tool_calls",
]


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


def categorize(content_blocks: list[dict[str, Any]]) -> str:
    has_text = any(
        b.get("type") == "text" and (b.get("text") or "").strip()
        for b in content_blocks
    )
    has_tool = any(b.get("type") == "tool_use" for b in content_blocks)
    if has_text and has_tool:
        return "mixed"
    if has_tool:
        return "tool_only"
    if has_text:
        return "text_only"
    return "empty"


def _usage_row(instance_id: str, ts: float, elapsed_ms: int,
               usage: dict[str, Any], content: list[dict[str, Any]],
               stop_reason: str | None) -> dict[str, Any]:
    """Build a per-turn CSV row matching the proxy's schema.

    No ttft/decode/itl timings — those require raw SSE-level instrumentation
    which we don't have via claude --output-format=stream-json. The columns
    are left blank so analysis scripts that need them gracefully skip.
    """
    inp = int(usage.get("input_tokens") or 0)
    cache_cr = int(usage.get("cache_creation_input_tokens") or 0)
    cache_rd = int(usage.get("cache_read_input_tokens") or 0)
    out_tok = int(usage.get("output_tokens") or 0)
    isl = inp + cache_cr + cache_rd
    isl_cached = cache_rd
    isl_new = isl - isl_cached
    cache_hit = (isl_cached / isl) if isl > 0 else 0.0
    return {
        "ts": ts,
        "instance_id": instance_id,
        "elapsed_ms": elapsed_ms,
        "ttft_ms": "",
        "decode_ms": "",
        "itl_ms": "",
        "isl": isl,
        "osl": out_tok,
        "isl_new": isl_new,
        "isl_cached": isl_cached,
        "cache_hit_rate": round(cache_hit, 4),
        "stop_reason": stop_reason or "",
        "category": categorize(content),
        "num_tool_calls": sum(1 for b in content if b.get("type") == "tool_use"),
    }


def consume_stream(proc: subprocess.Popen[str], instance_id: str,
                   csv_path: Path | None = None,
                   summary_path: Path | None = None) -> dict[str, Any]:
    """Consume claude --output-format stream-json.

    claude can split a single assistant response into multiple events (one
    per content block, partial usage per chunk). We dedupe by message id:
    buffer one logical turn, flush to CSV when a new message id arrives or
    when a user/result event signals end-of-turn.

    If summary_path is given, write the final result-event totals (ttft_ms,
    duration_ms, total_cost_usd, etc.) there so downstream analysis can
    compute throughput per problem.
    """
    summary: dict[str, Any] = {
        "instance_id": instance_id,
        "assistant_events": 0,
        "assistant_turns": 0,
        "categories_seen": {"text_only": 0, "tool_only": 0, "mixed": 0, "empty": 0},
        "session_id": None,
        "result": None,
    }

    csv_writer = None
    csv_fh = None
    if csv_path is not None:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        csv_fh = csv_path.open("w", newline="")
        csv_writer = csv.DictWriter(csv_fh, fieldnames=CSV_COLUMNS)
        csv_writer.writeheader()

    # Buffer for the current logical assistant turn.
    buf: dict[str, Any] | None = None

    def flush_buf() -> None:
        nonlocal buf
        if buf is None or csv_writer is None:
            buf = None
            return
        elapsed_ms = int((buf["last_ts"] - buf["first_ts"]) * 1000)
        # Time from previous turn's end to this turn's first event = "wait" /
        # tool-execution time; we record the API-call duration in elapsed_ms.
        csv_writer.writerow(_usage_row(
            instance_id=instance_id,
            ts=buf["first_ts"],
            elapsed_ms=max(elapsed_ms, 0),
            usage=buf["usage"],
            content=buf["content"],
            stop_reason=buf["stop_reason"],
        ))
        csv_fh.flush()
        summary["assistant_turns"] += 1
        summary["categories_seen"][categorize(buf["content"])] += 1
        buf = None

    assert proc.stdout is not None
    try:
        for raw in proc.stdout:
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            ev_type = ev.get("type")
            now = time.time()

            if ev_type == "system" and ev.get("subtype") == "init":
                summary["session_id"] = ev.get("session_id")
            elif ev_type == "assistant":
                msg = ev.get("message", {}) or {}
                msg_id = msg.get("id")
                content_chunk = msg.get("content", []) or []
                usage_chunk = msg.get("usage") or {}
                summary["assistant_events"] += 1
                if buf is None or buf["message_id"] != msg_id:
                    flush_buf()
                    buf = {
                        "message_id": msg_id,
                        "first_ts": now,
                        "last_ts": now,
                        "content": list(content_chunk),
                        "usage": dict(usage_chunk),
                        "stop_reason": msg.get("stop_reason"),
                    }
                else:
                    buf["last_ts"] = now
                    buf["content"].extend(content_chunk)
                    # Take latest usage values (claude reports running totals).
                    if usage_chunk:
                        buf["usage"].update(usage_chunk)
                    if msg.get("stop_reason"):
                        buf["stop_reason"] = msg["stop_reason"]
            elif ev_type == "user":
                flush_buf()
            elif ev_type == "result":
                flush_buf()
                summary["result"] = {
                    "subtype": ev.get("subtype"),
                    "num_turns": ev.get("num_turns"),
                    "duration_ms": ev.get("duration_ms"),
                    "duration_api_ms": ev.get("duration_api_ms"),
                    "ttft_ms": ev.get("ttft_ms"),
                    "is_error": ev.get("is_error"),
                    "total_cost_usd": ev.get("total_cost_usd"),
                    "usage": ev.get("usage"),
                    "modelUsage": ev.get("modelUsage"),
                }
    finally:
        flush_buf()
        if csv_fh is not None:
            csv_fh.close()

    if summary_path is not None and summary.get("result"):
        r = summary["result"]
        u = r.get("usage") or {}
        total_in = int(u.get("input_tokens") or 0) \
                 + int(u.get("cache_creation_input_tokens") or 0) \
                 + int(u.get("cache_read_input_tokens") or 0)
        total_out = int(u.get("output_tokens") or 0)
        dur_api = r.get("duration_api_ms")
        throughput = (total_out / (dur_api / 1000)) if dur_api else None
        out = {
            "instance_id": instance_id,
            "session_id": summary["session_id"],
            "assistant_turns": summary["assistant_turns"],
            "result_subtype": r.get("subtype"),
            "is_error": r.get("is_error"),
            "num_turns": r.get("num_turns"),
            "duration_ms": r.get("duration_ms"),
            "duration_api_ms": dur_api,
            "ttft_ms": r.get("ttft_ms"),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "cache_read_input_tokens": u.get("cache_read_input_tokens"),
            "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
            "total_cost_usd": r.get("total_cost_usd"),
            "throughput_out_tok_per_s": throughput,
            "modelUsage": r.get("modelUsage"),
        }
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(out, indent=2) + "\n")

    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--repo", required=True, help="e.g. astropy/astropy")
    ap.add_argument("--base-commit", required=True)
    ap.add_argument("--problem-statement", required=True)
    ap.add_argument("--model", required=True,
                    help="model id passed to claude (e.g. claude-opus-4-7)")
    ap.add_argument("--base-url", default="",
                    help="proxy URL — empty = use claude's default Anthropic")
    ap.add_argument("--no-proxy", action="store_true",
                    help="use claude's stored Anthropic credentials directly; "
                         "write per-turn CSV in this script (no proxy needed)")
    ap.add_argument("--per-problem-csv-dir", type=Path, default=None,
                    help="required with --no-proxy: one CSV per problem here")
    ap.add_argument("--workdir-root", required=True, type=Path)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--timeout-secs", type=int, default=600)
    args = ap.parse_args()

    if args.no_proxy and args.per_problem_csv_dir is None:
        print("--no-proxy requires --per-problem-csv-dir", file=sys.stderr)
        return 2

    args.workdir_root.mkdir(parents=True, exist_ok=True)

    # Per-problem scratch dir; cloned repo lives inside.
    workdir = Path(tempfile.mkdtemp(prefix=f"{args.instance_id}.", dir=args.workdir_root))
    repo_dir = workdir / "repo"

    # Shallow clone then fetch the exact base_commit.
    clone_url = f"https://github.com/{args.repo}.git"
    try:
        subprocess.run(
            ["git", "clone", "--quiet", clone_url, str(repo_dir)],
            check=True,
            timeout=600,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "checkout", "--quiet", args.base_commit],
            check=True,
            timeout=120,
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
    print(
        f"[run_one] {args.instance_id} prompt_chars={len(prompt)} "
        f"problem_statement_chars={len(args.problem_statement)}",
        file=sys.stderr, flush=True,
    )

    env = os.environ.copy()
    if not args.no_proxy:
        # Proxy mode: point claude at our proxy. Auth handling depends on the
        # ultimate upstream:
        #   vllm     — vLLM ignores auth, so inject a dummy API key.
        #   anthropic — keep the OAuth bearer the parent shell already put in
        #               ANTHROPIC_AUTH_TOKEN; do not set ANTHROPIC_API_KEY
        #               (that would make claude send x-api-key instead of
        #               Authorization: Bearer, and Anthropic would reject).
        env["ANTHROPIC_BASE_URL"] = args.base_url
        if not env.get("ANTHROPIC_AUTH_TOKEN"):
            env["ANTHROPIC_API_KEY"] = env.get("ANTHROPIC_API_KEY", "dummy-local")
        env["ANTHROPIC_CUSTOM_HEADERS"] = (
            env.get("ANTHROPIC_CUSTOM_HEADERS", "").rstrip()
            + ("\n" if env.get("ANTHROPIC_CUSTOM_HEADERS") else "")
            + f"X-Instance-Id: {args.instance_id}"
        )
    else:
        # OAuth mode: let claude use its stored credentials (set via `claude
        # login`). Don't override BASE_URL or API_KEY — those would replace
        # the cached auth state and force unauthenticated requests.
        env.pop("ANTHROPIC_BASE_URL", None)
        env.pop("ANTHROPIC_API_KEY", None)
    env["ANTHROPIC_MODEL"] = args.model
    # Claude Code internally also picks small/fast models; pin them all.
    env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = args.model
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = args.model
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = args.model
    # Required to allow --dangerously-skip-permissions when running as root.
    env["IS_SANDBOX"] = "1"

    cmd = [
        "claude",
        "-p", prompt,
        "--output-format", "stream-json",
        "--verbose",
        "--max-turns", str(args.max_turns),
        "--dangerously-skip-permissions",
        "--model", args.model,
        # Vanilla claude (no --bare): loads the full ~20K system prompt,
        # auto-memory, hooks, plugin sync, and CLAUDE.md. Per-problem workdirs
        # are fresh (timestamped), so there is no prior session to resume.
    ]

    started = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_dir),
        env=env,
        stdin=subprocess.DEVNULL,  # claude -p appends stdin to its prompt; we
                                   # inherit run.sh's stdin (the 8 MB
                                   # problems.jsonl), so explicitly close it.
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    timed_out = False
    try:
        csv_path = (args.per_problem_csv_dir / f"{args.instance_id}.csv"
                    if args.no_proxy and args.per_problem_csv_dir else None)
        summary_path = (args.per_problem_csv_dir / f"{args.instance_id}.summary.json"
                        if args.no_proxy and args.per_problem_csv_dir else None)
        summary = consume_stream(proc, args.instance_id,
                                 csv_path=csv_path, summary_path=summary_path)
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
                print(f"[run_one stderr {args.instance_id}] {err_tail}", file=sys.stderr)
        shutil.rmtree(workdir, ignore_errors=True)

    print(json.dumps({
        "instance_id": args.instance_id,
        "elapsed_s": round(time.time() - started, 2),
        "timed_out": timed_out,
        "exit_code": proc.returncode,
        **{k: summary.get(k) for k in (
            "assistant_events", "categories_seen", "session_id", "result",
        )},
    }))
    return 0 if not timed_out else 124


if __name__ == "__main__":
    raise SystemExit(main())
