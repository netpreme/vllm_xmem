"""claude-cli invocation.

Drives Claude Code as a subprocess. All per-turn telemetry comes from
the metrics_watcher (vLLM-side) and proxy (HTTP-side); claude's
own stream-json output is discarded.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import psutil
from loguru import logger

# Wall-clock ceiling for one claude session. The per-Bash-command caps
# (BASH_*_TIMEOUT_MS below) only bound individual tool calls; they can't catch
# a hang *outside* a Bash call — e.g. claude finishing its turn but never
# exiting, which wedges the whole benchmark on `proc.wait()`. This is the
# backstop for that. Observed problems finish in 1-3 min, so 30 min is
# generous headroom while still bounding any single problem.
DEFAULT_TIMEOUT_S = 1800

# Exit code returned when the session is killed for exceeding DEFAULT_TIMEOUT_S
# (matches coreutils `timeout`). Non-zero, so the problem is recorded as
# unsolved and a later --resume retries it.
TIMEOUT_EXIT_CODE = 124

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


def claude_version() -> str:
    """claude-cli version string, e.g. '2.1.156 (Claude Code)'; '' if unavailable."""
    try:
        return subprocess.run(
            ["claude", "--version"], capture_output=True, text=True, timeout=10
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return ""


def solve(
    problem: dict, repo_dir: Path, model: str, url: str, timeout_s: float = DEFAULT_TIMEOUT_S
) -> int:
    """Run claude-cli on `problem` inside `repo_dir`. Returns the exit code.

    stdout is consumed (and discarded) to avoid backpressure on the pipe;
    stderr is dropped. All per-turn telemetry is captured out-of-band by
    the watcher + proxy.

    The session is bounded by `timeout_s` of wall-clock. On timeout the whole
    process tree (claude + any Bash children) is killed and TIMEOUT_EXIT_CODE
    is returned, so a hung session can never block the benchmark."""
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
        env=create_claude_env(model=model, url=url),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    try:
        proc.wait(timeout=timeout_s)
        return proc.returncode
    except subprocess.TimeoutExpired:
        logger.warning(
            "{}: claude session exceeded {}s — killing process tree",
            problem["instance_id"],
            timeout_s,
        )
        _kill_tree(proc.pid)
        return TIMEOUT_EXIT_CODE
    except BaseException:
        # Ctrl-C / abort: still reap the tree so nothing is left orphaned.
        _kill_tree(proc.pid)
        raise


# Grace period between SIGTERM and SIGKILL when reaping the claude tree.
_TERM_GRACE_S = 10


def _kill_tree(pid: int) -> None:
    """Kill `pid` and every descendant: SIGTERM, then SIGKILL to survivors.

    claude-cli launches background Bash tasks (run_in_background) in their own
    session/process group via setsid, so os.killpg on claude's group misses
    them — and those background tasks (e.g. `python tests/runtests.py`) are the
    very things that hang. We walk the descendant tree by PID instead, which
    crosses those group/session boundaries. Descendants are collected *before*
    killing, since reaping the parent reparents survivors to init and breaks
    the tree linkage."""
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return
    procs = root.children(recursive=True)
    procs.append(root)
    for p in procs:
        try:
            p.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    _, alive = psutil.wait_procs(procs, timeout=_TERM_GRACE_S)
    for p in alive:
        try:
            p.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass


def create_claude_env(model: str, url: str) -> dict[str, str]:
    """Env for claude-cli. Points it at our local vLLM (or proxy) and pins
    every model slot (main/opus/sonnet/haiku) to the same served model."""
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = url
    env.setdefault("ANTHROPIC_API_KEY", "vllm-local")
    env["ANTHROPIC_MODEL"] = model
    env["ANTHROPIC_DEFAULT_OPUS_MODEL"] = model
    env["ANTHROPIC_DEFAULT_SONNET_MODEL"] = model
    env["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = model
    env["IS_SANDBOX"] = "1"  # allows --dangerously-skip-permissions as root
    # Cap every Bash tool invocation. Without this, claude can wedge on a
    # long-running server (e.g. `manage.py runserver`) and block the whole
    # benchmark indefinitely. Defaults: each command has 5 min; max ceiling
    # is 10 min so explicit `timeout=` calls in claude's prompt can't exceed it.
    env.setdefault("BASH_DEFAULT_TIMEOUT_MS", "300000")
    env.setdefault("BASH_MAX_TIMEOUT_MS", "600000")
    return env
