"""Run `claude -p` on one SWE-bench problem, routed through pipeline/proxy.py.

The proxy owns all per-turn metrics (CSV + body dumps); this script only
launches claude, clones the repo, streams its stdout for diagnostics, and
writes a per-problem summary.json from claude's final `result` event.
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


def consume_stream(proc: subprocess.Popen[str]) -> dict[str, Any]:
    """Parse claude's stream-json events, return session_id + final result."""
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


def write_summary(path: Path, instance_id: str, session_id: str | None,
                  result: dict[str, Any]) -> None:
    u = result.get("usage") or {}
    in_tot = sum(int(u.get(k) or 0) for k in
                 ("input_tokens", "cache_creation_input_tokens",
                  "cache_read_input_tokens"))
    out_tot = int(u.get("output_tokens") or 0)
    dur_api = result.get("duration_api_ms")
    payload = {
        "instance_id": instance_id,
        "session_id":  session_id,
        "result_subtype":     result.get("subtype"),
        "is_error":           result.get("is_error"),
        "num_turns":          result.get("num_turns"),
        "duration_ms":        result.get("duration_ms"),
        "duration_api_ms":    dur_api,
        "ttft_ms":            result.get("ttft_ms"),
        "total_input_tokens":  in_tot,
        "total_output_tokens": out_tot,
        "cache_read_input_tokens":     u.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": u.get("cache_creation_input_tokens"),
        "total_cost_usd":     result.get("total_cost_usd"),
        "throughput_out_tok_per_s": (out_tot / (dur_api / 1000)) if dur_api else None,
        "modelUsage":         result.get("modelUsage"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def clone_repo(repo: str, base_commit: str, workdir: Path) -> Path:
    repo_dir = workdir / "repo"
    subprocess.run(["git", "clone", "--quiet",
                    f"https://github.com/{repo}.git", str(repo_dir)],
                   check=True, timeout=600)
    subprocess.run(["git", "-C", str(repo_dir), "checkout", "--quiet", base_commit],
                   check=True, timeout=120)
    return repo_dir


def build_env(model: str, base_url: str, instance_id: str) -> dict[str, str]:
    """claude routes through our proxy. Preserve ANTHROPIC_AUTH_TOKEN if the
    parent shell set it (anthropic backend OAuth bearer); otherwise drop a
    dummy API key (vLLM ignores auth). Tag every request with the
    instance_id so the proxy can sort CSV rows by problem."""
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = base_url
    if not env.get("ANTHROPIC_AUTH_TOKEN"):
        env.setdefault("ANTHROPIC_API_KEY", "dummy-local")
    env["ANTHROPIC_CUSTOM_HEADERS"] = (
        env.get("ANTHROPIC_CUSTOM_HEADERS", "").rstrip()
        + ("\n" if env.get("ANTHROPIC_CUSTOM_HEADERS") else "")
        + f"X-Instance-Id: {instance_id}"
    )
    for k in ("ANTHROPIC_MODEL", "ANTHROPIC_DEFAULT_OPUS_MODEL",
              "ANTHROPIC_DEFAULT_SONNET_MODEL", "ANTHROPIC_DEFAULT_HAIKU_MODEL"):
        env[k] = model
    env["IS_SANDBOX"] = "1"  # lets --dangerously-skip-permissions work as root
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instance-id",       required=True)
    ap.add_argument("--repo",              required=True)
    ap.add_argument("--base-commit",       required=True)
    ap.add_argument("--problem-statement", required=True)
    ap.add_argument("--model",             required=True)
    ap.add_argument("--base-url",          required=True, help="proxy URL")
    ap.add_argument("--workdir-root",      required=True, type=Path)
    ap.add_argument("--per-problem-dir",   type=Path, default=None,
                    help="optional dir to drop <instance_id>.summary.json")
    ap.add_argument("--max-turns",         type=int, default=30)
    ap.add_argument("--timeout-secs",      type=int, default=600)
    args = ap.parse_args()

    args.workdir_root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix=f"{args.instance_id}.",
                                    dir=args.workdir_root))
    try:
        repo_dir = clone_repo(args.repo, args.base_commit, workdir)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(json.dumps({"instance_id": args.instance_id,
                          "error": f"clone_failed: {e!r}"}))
        shutil.rmtree(workdir, ignore_errors=True)
        return 1

    prompt = PROMPT_TEMPLATE.format(
        repo=args.repo, base_commit=args.base_commit,
        problem_statement=args.problem_statement,
    )
    env = build_env(args.model, args.base_url, args.instance_id)
    cmd = ["claude", "-p", prompt,
           "--output-format", "stream-json", "--verbose",
           "--max-turns", str(args.max_turns),
           "--dangerously-skip-permissions",
           "--model", args.model]

    print(f"[solve] {args.instance_id} prompt_chars={len(prompt)}",
          file=sys.stderr, flush=True)

    started = time.time()
    proc = subprocess.Popen(
        cmd, cwd=str(repo_dir), env=env,
        # Don't inherit run.sh's stdin (8 MB problems.jsonl) — claude -p
        # appends stdin to its prompt.
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )

    timed_out = False
    try:
        ev = consume_stream(proc)
        try:
            proc.wait(timeout=max(1, args.timeout_secs - int(time.time() - started)))
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.send_signal(signal.SIGINT)
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired: proc.kill()
    finally:
        if proc.stderr is not None:
            tail = proc.stderr.read()[-2000:]
            if tail.strip():
                print(f"[solve stderr {args.instance_id}] {tail}", file=sys.stderr)
        shutil.rmtree(workdir, ignore_errors=True)

    if args.per_problem_dir and ev["result"]:
        write_summary(args.per_problem_dir / f"{args.instance_id}.summary.json",
                      args.instance_id, ev["session_id"], ev["result"])

    print(json.dumps({
        "instance_id": args.instance_id,
        "elapsed_s":   round(time.time() - started, 2),
        "timed_out":   timed_out,
        "exit_code":   proc.returncode,
        "session_id":  ev["session_id"],
    }))
    return 0 if not timed_out else 124


if __name__ == "__main__":
    raise SystemExit(main())
