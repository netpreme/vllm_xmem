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


def consume_stream(proc: subprocess.Popen[str], instance_id: str) -> dict[str, Any]:
    """Consume claude --output-format stream-json for diagnostics only.

    Per-call ISL/OSL is logged by the proxy; here we just track session id,
    high-level categories from claude's split assistant events (best-effort
    indicator that tool calls are happening), and the final result.
    """
    summary: dict[str, Any] = {
        "instance_id": instance_id,
        "assistant_events": 0,
        "categories_seen": {"text_only": 0, "tool_only": 0, "mixed": 0, "empty": 0},
        "session_id": None,
        "result": None,
    }
    assert proc.stdout is not None
    for raw in proc.stdout:
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            continue

        ev_type = ev.get("type")
        if ev_type == "system" and ev.get("subtype") == "init":
            summary["session_id"] = ev.get("session_id")
        elif ev_type == "assistant":
            msg = ev.get("message", {}) or {}
            content = msg.get("content", []) or []
            cat = categorize(content)
            summary["assistant_events"] += 1
            summary["categories_seen"][cat] += 1
        elif ev_type == "result":
            summary["result"] = {
                "subtype": ev.get("subtype"),
                "num_turns": ev.get("num_turns"),
                "duration_ms": ev.get("duration_ms"),
                "is_error": ev.get("is_error"),
                "total_cost_usd": ev.get("total_cost_usd"),
            }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--repo", required=True, help="e.g. astropy/astropy")
    ap.add_argument("--base-commit", required=True)
    ap.add_argument("--problem-statement", required=True)
    ap.add_argument("--model", required=True, help="served-model-name on vllm")
    ap.add_argument("--base-url", required=True,
                    help="proxy URL (http://host:port) — claude points here")
    ap.add_argument("--workdir-root", required=True, type=Path)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--timeout-secs", type=int, default=600)
    args = ap.parse_args()

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
    env["ANTHROPIC_BASE_URL"] = args.base_url
    env["ANTHROPIC_API_KEY"] = env.get("ANTHROPIC_API_KEY", "dummy-local")
    env["ANTHROPIC_MODEL"] = args.model
    # Claude Code internally also picks small/fast models; pin them all.
    env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = args.model
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = args.model
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = args.model
    # Required to allow --dangerously-skip-permissions when running as root.
    env["IS_SANDBOX"] = "1"

    # Tag every API call this claude run makes with our instance_id so the
    # proxy can attribute usage rows. ANTHROPIC_CUSTOM_HEADERS is honored by
    # claude as a "k:v\nk:v" string and forwarded on every /v1/messages call.
    env["ANTHROPIC_CUSTOM_HEADERS"] = (
        env.get("ANTHROPIC_CUSTOM_HEADERS", "").rstrip()
        + ("\n" if env.get("ANTHROPIC_CUSTOM_HEADERS") else "")
        + f"X-Instance-Id: {args.instance_id}"
    )

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
        summary = consume_stream(proc, args.instance_id)
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
