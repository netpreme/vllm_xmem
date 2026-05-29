"""claude-cli invocation.

Drives Claude Code as a subprocess. All per-turn telemetry comes from
the metrics_watcher (vLLM-side) and agent_labeler (HTTP-side); claude's
own stream-json output is discarded.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path


PROMPT = """You are working on a real software-engineering bug from SWE-bench Verified. \
Solve it by editing files in this repository.

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


def _env(model: str, base_url: str) -> dict[str, str]:
    """Env for claude-cli. Points it at our local vLLM (or proxy) and pins
    every model slot (main/opus/sonnet/haiku) to the same served model."""
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = base_url
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


def solve(problem: dict, repo_dir: Path, model: str, base_url: str) -> int:
    """Run claude-cli on `problem` inside `repo_dir`. Returns the exit code.

    stdout is consumed (and discarded) to avoid backpressure on the pipe;
    stderr is dropped. All per-turn telemetry is captured out-of-band by
    the watcher + proxy."""
    prompt = PROMPT.format(**problem)
    proc = subprocess.Popen(
        [
            "claude",
            "-p",
            prompt,
            "--output-format",
            "stream-json",
            "--verbose",
            "--dangerously-skip-permissions",
            "--model",
            model,
        ],
        cwd=str(repo_dir),
        env=_env(model, base_url),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    proc.wait()
    return proc.returncode
