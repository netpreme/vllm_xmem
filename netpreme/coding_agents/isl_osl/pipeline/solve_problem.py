"""Solve one SWE-bench problem by running `claude -p` against a vLLM server.

This script is a thin wrapper around claude-cli:
    1. Clone the target repo at the specified base commit into a tempdir.
    2. Build the agent prompt from the problem statement.
    3. Launch claude-cli pointed at our local vLLM endpoint.
    4. Stream claude-cli's stream-json output, collecting the session id
       and the final `result` event for the per-problem summary.json.
    5. Tear down the tempdir.

Per-turn metrics (TTFT, ITL, KV-cache, etc.) are collected by the
companion `pipeline/metrics_watcher.py`, which polls vLLM's Prometheus
`/metrics` endpoint in the background. This script does not touch
metrics — it only drives the agent.
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


# ---------------------------------------------------------------------------
# Agent prompt.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Git checkout.
# ---------------------------------------------------------------------------

def clone_repo(repo: str, base_commit: str, workdir: Path) -> Path:
    """Clone `github.com/<repo>` into <workdir>/repo and checkout `base_commit`.
    Returns the path of the checked-out tree."""
    repo_dir = workdir / "repo"
    subprocess.run(
        ["git", "clone", "--quiet", f"https://github.com/{repo}.git", str(repo_dir)],
        check=True, timeout=600,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "checkout", "--quiet", base_commit],
        check=True, timeout=120,
    )
    return repo_dir


# ---------------------------------------------------------------------------
# Claude environment.
# ---------------------------------------------------------------------------

def build_claude_env(model: str, vllm_url: str) -> dict[str, str]:
    """Build the env dict for the claude-cli subprocess.

    We point claude-cli at our local vLLM via ANTHROPIC_BASE_URL, and
    pin every internal model slot (main / opus / sonnet / haiku) to the
    same locally-served model. vLLM ignores the API key but claude-cli
    refuses to start without one, so we drop in a placeholder.
    """
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = vllm_url
    env.setdefault("ANTHROPIC_API_KEY", "vllm-local")
    for slot in (
        "ANTHROPIC_MODEL",
        "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "ANTHROPIC_DEFAULT_HAIKU_MODEL",
    ):
        env[slot] = model
    env["IS_SANDBOX"] = "1"  # allows --dangerously-skip-permissions as root
    return env


# ---------------------------------------------------------------------------
# Stream-json consumer.
# ---------------------------------------------------------------------------

def consume_stream(proc: subprocess.Popen[str]) -> dict[str, Any]:
    """Drain claude-cli's `--output-format=stream-json` output.
    Returns just the bits we need for the per-problem summary file:
    `session_id` (from the first `system/init` event) and `result` (the
    aggregate event claude emits at the very end)."""
    out: dict[str, Any] = {"session_id": None, "result": None}
    assert proc.stdout is not None
    for raw in proc.stdout:
        raw = raw.strip()
        if not raw:
            continue
        try:
            ev = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "system" and ev.get("subtype") == "init":
            out["session_id"] = ev.get("session_id")
        elif ev.get("type") == "result":
            out["result"] = ev
    return out


# ---------------------------------------------------------------------------
# Per-problem summary file.
# ---------------------------------------------------------------------------

def write_summary(path: Path, instance_id: str, session_id: str | None,
                  result: dict[str, Any]) -> None:
    """Flatten claude-cli's `result` event into a per-problem summary."""
    usage      = result.get("usage") or {}
    input_tot  = sum(int(usage.get(k) or 0) for k in (
        "input_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    ))
    output_tot = int(usage.get("output_tokens") or 0)
    duration_api_ms = result.get("duration_api_ms")
    throughput = (output_tot / (duration_api_ms / 1000)
                  if duration_api_ms else None)

    payload = {
        "instance_id":                 instance_id,
        "session_id":                  session_id,
        "result_subtype":              result.get("subtype"),
        "is_error":                    result.get("is_error"),
        "num_turns":                   result.get("num_turns"),
        "duration_ms":                 result.get("duration_ms"),
        "duration_api_ms":             duration_api_ms,
        "ttft_ms":                     result.get("ttft_ms"),
        "total_input_tokens":          input_tot,
        "total_output_tokens":         output_tot,
        "cache_read_input_tokens":     usage.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "total_cost_usd":              result.get("total_cost_usd"),
        "throughput_out_tok_per_s":    throughput,
        "modelUsage":                  result.get("modelUsage"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id",       required=True)
    parser.add_argument("--repo",              required=True,
                        help="GitHub org/name, e.g. 'astropy/astropy'")
    parser.add_argument("--base-commit",       required=True)
    parser.add_argument("--problem-statement", required=True)
    parser.add_argument("--model",             required=True,
                        help="model id claude-cli should send to vLLM")
    parser.add_argument("--vllm-url",          required=True,
                        help="e.g. http://localhost:8000")
    parser.add_argument("--workdir-root",      required=True, type=Path)
    parser.add_argument("--per-problem-dir",   type=Path, default=None,
                        help="if set, write <instance_id>.summary.json here")
    parser.add_argument("--max-turns",         type=int, default=30)
    parser.add_argument("--timeout-secs",      type=int, default=600)
    args = parser.parse_args()

    # --- 1. Repo checkout ---------------------------------------------------
    args.workdir_root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=f"{args.instance_id}.",
                                    dir=args.workdir_root))
    try:
        repo_dir = clone_repo(args.repo, args.base_commit, workdir)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        print(json.dumps({"instance_id": args.instance_id,
                          "error": f"clone_failed: {exc!r}"}))
        shutil.rmtree(workdir, ignore_errors=True)
        return 1

    # --- 2. Build prompt + environment -------------------------------------
    prompt = PROMPT_TEMPLATE.format(
        repo=args.repo,
        base_commit=args.base_commit,
        problem_statement=args.problem_statement,
    )
    env = build_claude_env(args.model, args.vllm_url)
    cmd = [
        "claude", "-p", prompt,
        "--output-format", "stream-json", "--verbose",
        "--max-turns", str(args.max_turns),
        "--dangerously-skip-permissions",
        "--model", args.model,
    ]
    print(f"[solve] {args.instance_id} prompt_chars={len(prompt)}",
          file=sys.stderr, flush=True)

    # --- 3. Run claude-cli, consume its stream-json output -----------------
    started = time.time()
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_dir),
        env=env,
        # claude -p appends stdin to its prompt; isolate it from run.sh's
        # 8 MB problems.jsonl by closing it explicitly.
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    timed_out = False
    try:
        events = consume_stream(proc)
        try:
            proc.wait(timeout=max(1, args.timeout_secs
                                  - int(time.time() - started)))
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
    finally:
        # Tail stderr for diagnostics, then drop the tempdir.
        if proc.stderr is not None:
            stderr_tail = proc.stderr.read()[-2000:]
            if stderr_tail.strip():
                print(f"[solve stderr {args.instance_id}] {stderr_tail}",
                      file=sys.stderr)
        shutil.rmtree(workdir, ignore_errors=True)

    # --- 4. Per-problem summary --------------------------------------------
    if args.per_problem_dir and events["result"]:
        write_summary(
            args.per_problem_dir / f"{args.instance_id}.summary.json",
            args.instance_id,
            events["session_id"],
            events["result"],
        )

    print(json.dumps({
        "instance_id": args.instance_id,
        "elapsed_s":   round(time.time() - started, 2),
        "timed_out":   timed_out,
        "exit_code":   proc.returncode,
        "session_id":  events["session_id"],
    }))
    return 0 if not timed_out else 124


if __name__ == "__main__":
    raise SystemExit(main())
