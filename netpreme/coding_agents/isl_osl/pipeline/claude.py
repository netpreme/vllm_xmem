"""claude-cli invocation.

Drives Claude Code as a subprocess. On the vLLM backend, per-turn telemetry
comes from the metrics_watcher (vLLM-side) and proxy (HTTP-side) and claude's
own stream-json output is discarded. On the Anthropic/OAuth backend (no proxy)
the only thing captured is a copy of claude-cli's session transcript; counts
and per-turn texts are derived from it later by the analysis pass.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
from pathlib import Path

from loguru import logger

from pipeline.utils.jsonl import instance_dir
from pipeline.utils.processes import terminate_process_tree

# Wall-clock ceiling for one claude session. The per-Bash-command caps
# (BASH_*_TIMEOUT_MS below) only bound individual tool calls, and only
# *foreground* ones — they do NOT cover `run_in_background` Bash (which
# setsids into its own group). So a backgrounded whole-suite test run
# (e.g. gpt-oss issuing `pytest -q sympy`) runs unbounded and wedges the
# benchmark on `proc.wait()`; this session backstop is the only thing that
# catches it. Set to 2h so genuinely slow problems get a real chance to
# finish before we give up (a wedged one still eventually gets reaped).
DEFAULT_TIMEOUT_S = 7200

# Exit code returned when the session is killed for exceeding DEFAULT_TIMEOUT_S
# (matches coreutils `timeout`). Non-zero, so the problem is recorded as
# unsolved and a later --resume retries it.
TIMEOUT_EXIT_CODE = 124

PROMPT = """You are working on a real software-engineering bug from SWE-bench. \
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
    problem: dict,
    repo_dir: Path,
    model: str,
    url: str,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    *,
    oauth: bool = False,
    telemetry_dir: Path | None = None,
) -> int:
    """Run claude-cli on `problem` inside `repo_dir`. Returns the exit code.

    On the vLLM backend, stdout is discarded and per-turn telemetry comes from
    the watcher + proxy. With `oauth=True` (Anthropic subscription, no proxy)
    there is no watcher, so when `telemetry_dir` is given we copy claude-cli's
    session transcript (the authoritative telemetry source); counts and per-turn
    texts are derived from it later by the analysis pass.

    The session is bounded by `timeout_s` of wall-clock. On timeout the whole
    process tree (claude + any Bash children) is killed and TIMEOUT_EXIT_CODE
    is returned, so a hung session can never block the benchmark."""
    prompt = PROMPT.format(**problem)
    capture = telemetry_dir is not None
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
        env=create_claude_env(model=model, url=url, oauth=oauth),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    reader = None
    session_sink: dict = {}
    if capture:
        # Drain stdout (avoid pipe backpressure) just to learn the session id;
        # the authoritative per-turn telemetry is parsed from the on-disk
        # transcript afterwards — stdout's usage.output_tokens is only a partial
        # mid-generation snapshot, which under-counts osl.
        reader = threading.Thread(
            target=get_session_id_from_logs,
            args=(proc.stdout, session_sink),
            daemon=True,
        )
        reader.start()
    try:
        proc.wait(timeout=timeout_s)
        return proc.returncode
    except subprocess.TimeoutExpired:
        logger.warning(
            "{}: claude session exceeded {}s — killing process tree",
            problem["instance_id"],
            timeout_s,
        )
        terminate_process_tree(proc.pid, _TERM_GRACE_S)
        return TIMEOUT_EXIT_CODE
    except BaseException:
        # Ctrl-C / abort: still reap the tree so nothing is left orphaned.
        terminate_process_tree(proc.pid, _TERM_GRACE_S)
        raise
    finally:
        if reader is not None:
            reader.join(timeout=10)
        if capture:
            _copy_transcript(
                session_sink.get("id"),
                Path(repo_dir),
                Path(telemetry_dir),
                problem["instance_id"],
            )


def get_session_id_from_logs(stdout, sink: dict) -> None:
    """Consume claude-cli's stream-json stdout (so the pipe never blocks) and
    grab the session id from the init event — used to find the on-disk
    transcript, which is the real telemetry source."""
    for line in stdout:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "system" and event.get("session_id"):
            sink["id"] = event["session_id"]


def _locate_transcript(session_id: str | None, repo_dir: Path) -> Path | None:
    """claude-cli writes ~/.claude/projects/<encoded-cwd>/<session_id>.jsonl.
    Prefer the session id (exact); fall back to the newest transcript under the
    cwd-encoded project dir."""
    projects_dir = Path.home() / ".claude" / "projects"
    if session_id:
        transcript_paths = list(projects_dir.glob(f"*/{session_id}.jsonl"))
        if transcript_paths:
            return transcript_paths[0]
    encoded_repo_dir = re.sub(r"[^A-Za-z0-9]", "-", str(repo_dir.resolve()))
    transcript_paths = sorted(
        (projects_dir / encoded_repo_dir).glob("*.jsonl"),
        key=lambda path: path.stat().st_mtime,
    )
    return transcript_paths[-1] if transcript_paths else None


def _copy_transcript(
    session_id: str | None,
    repo_dir: Path,
    telemetry_dir: Path,
    instance_id: str,
) -> None:
    """Copy claude-cli's on-disk session transcript to
    telemetry/<iid>/claude_transcript.jsonl — the authoritative, untouched
    record (final per-turn usage + full tool outputs + conversation text).
    That's the ONLY thing captured for the Anthropic/OAuth backend; the derived
    vllm_metrics.jsonl/vllm_traces.jsonl (counts + per-turn texts) are produced
    separately by the analysis pass, since they only make sense as a post-hoc
    transform of this transcript, not as live vLLM telemetry."""
    tpath = _locate_transcript(session_id, repo_dir)
    if tpath is None:
        logger.warning(
            "{}: no claude transcript found — telemetry skipped", instance_id
        )
        return
    idir = instance_dir(telemetry_dir, instance_id)
    idir.mkdir(parents=True, exist_ok=True)
    try:
        shutil.copy(tpath, idir / "claude_transcript.jsonl")
    except OSError as exc:
        logger.warning("{}: transcript copy failed: {!r}", instance_id, exc)


# Grace period between SIGTERM and SIGKILL when reaping the claude tree.
_TERM_GRACE_S = 10


def create_claude_env(model: str, url: str, *, oauth: bool = False) -> dict[str, str]:
    """Env for claude-cli.

    Default (vLLM backend): point it at our local vLLM/proxy via
    ANTHROPIC_BASE_URL and pin every model slot to the served model.

    oauth=True (Anthropic backend): talk DIRECTLY to Anthropic using claude-cli's
    own subscription login. We strip ANTHROPIC_BASE_URL/ANTHROPIC_API_KEY —
    otherwise claude-cli would auth with x-api-key against the override instead
    of the OAuth token — and let --model pick the model. Stripping the key
    unconditionally also guarantees a run can never spend metered API credits:
    the subscription is the only auth path here."""
    env = os.environ.copy()
    if oauth:
        env.pop("ANTHROPIC_BASE_URL", None)
        env.pop("ANTHROPIC_API_KEY", None)
    else:
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
