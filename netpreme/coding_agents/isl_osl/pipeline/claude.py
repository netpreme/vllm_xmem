"""claude-cli invocation.

Drives Claude Code as a subprocess. All per-turn telemetry comes from
the metrics_watcher (vLLM-side) and proxy (HTTP-side); claude's
own stream-json output is discarded.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import psutil
from loguru import logger

from pipeline.utils.jsonl import JsonlWriter, instance_dir

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
    raw: bool = False,
) -> int:
    """Run claude-cli on `problem` inside `repo_dir`. Returns the exit code.

    On the vLLM backend, stdout is discarded and per-turn telemetry comes from
    the watcher + proxy. With `oauth=True` (Anthropic subscription, no proxy)
    there is no watcher, so when `telemetry_dir` is given we parse claude's
    stream-json stdout for per-turn `usage` and write isl/osl/isl_new ourselves.

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
            target=_drain_for_session, args=(proc.stdout, session_sink), daemon=True
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
        _kill_tree(proc.pid)
        return TIMEOUT_EXIT_CODE
    except BaseException:
        # Ctrl-C / abort: still reap the tree so nothing is left orphaned.
        _kill_tree(proc.pid)
        raise
    finally:
        if reader is not None:
            reader.join(timeout=10)
        if capture:
            _capture_from_transcript(
                session_sink.get("id"),
                Path(repo_dir),
                Path(telemetry_dir),
                problem["instance_id"],
                raw,
            )


def _drain_for_session(stdout, sink: dict) -> None:
    """Consume claude-cli's stream-json stdout (so the pipe never blocks) and
    grab the session id from the init event — used to find the on-disk
    transcript, which is the real telemetry source."""
    for line in stdout:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "system" and ev.get("session_id"):
            sink["id"] = ev["session_id"]


def _locate_transcript(session_id: str | None, repo_dir: Path) -> Path | None:
    """claude-cli writes ~/.claude/projects/<encoded-cwd>/<session_id>.jsonl.
    Prefer the session id (exact); fall back to the newest transcript under the
    cwd-encoded project dir."""
    proj = Path.home() / ".claude" / "projects"
    if session_id:
        hits = list(proj.glob(f"*/{session_id}.jsonl"))
        if hits:
            return hits[0]
    enc = re.sub(r"[^A-Za-z0-9]", "-", str(repo_dir.resolve()))
    cands = sorted((proj / enc).glob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def _capture_from_transcript(
    session_id: str | None,
    repo_dir: Path,
    telemetry_dir: Path,
    instance_id: str,
    raw: bool,
) -> None:
    """Derive vllm.jsonl (+ raw.jsonl) from claude-cli's on-disk transcript,
    which carries the FINAL per-turn usage (correct osl) and full tool outputs.
    With raw, also copy the transcript itself next to the derived files."""
    tpath = _locate_transcript(session_id, repo_dir)
    if tpath is None:
        logger.warning("{}: no claude transcript found — telemetry skipped", instance_id)
        return
    with open(tpath) as fh:
        _capture_usage(fh, telemetry_dir, instance_id, raw)
    if raw:
        try:
            shutil.copy(tpath, instance_dir(telemetry_dir, instance_id) / "transcript.jsonl")
        except OSError as exc:
            logger.warning("{}: transcript copy failed: {!r}", instance_id, exc)


def _render_blocks(content) -> str:
    """Flatten an Anthropic message `content` (str or list of blocks) to text:
    visible text, reasoning, tool calls (name + args) and tool results — i.e.
    what actually went into / came out of the turn."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for b in content:
        if not isinstance(b, dict):
            continue
        bt = b.get("type")
        if bt == "text":
            parts.append(b.get("text") or "")
        elif bt == "thinking":
            parts.append(b.get("thinking") or "")
        elif bt == "tool_use":
            parts.append(f"[tool_use:{b.get('name')}] " + json.dumps(b.get("input") or {}))
        elif bt == "tool_result":
            parts.append("[tool_result] " + _render_blocks(b.get("content")))
    return "\n".join(p for p in parts if p)


def _capture_usage(
    stdout,
    telemetry_dir: Path,
    instance_id: str,
    raw: bool,
    session_sink: dict | None = None,
) -> None:
    """Parse claude-cli's stream-json stdout (the OAuth backend's telemetry
    source). Per `type=assistant` turn, write a vllm.jsonl row from its Anthropic
    `usage` — isl = full prompt, isl_new = the non-cache-read part. With `raw`,
    also derive per-turn osl_text (the assistant output) + isl_new_text (the
    tool_result/user content appended since the previous turn) into raw.jsonl.
    No token ids — Anthropic exposes none. The session id (from the init event)
    is reported via `session_sink` so solve() can also copy the full on-disk
    transcript, which carries the richer tool outputs."""
    idir = instance_dir(telemetry_dir, instance_id)
    idir.mkdir(parents=True, exist_ok=True)
    (idir / "vllm.jsonl").unlink(missing_ok=True)
    writer = JsonlWriter(telemetry_dir, "vllm.jsonl")
    raw_writer = None
    if raw:
        (idir / "raw.jsonl").unlink(missing_ok=True)
        raw_writer = JsonlWriter(telemetry_dir, "raw.jsonl")
    # One turn = one message id, but claude-cli streams it as several assistant
    # events (one per content block: thinking, tool_use, ...) that repeat the
    # message's usage. So accumulate blocks per id and flush one row when the id
    # changes / stream ends. pending = tool_result/user text feeding the NEXT
    # turn; captured into the turn's isl_new_text when that turn starts.
    cur_id = None
    cur_blocks: list = []
    cur_usage: dict = {}
    cur_isl_new = ""
    pending: list[str] = []

    def flush() -> None:
        if cur_id is None:
            return
        # Anthropic usage: input_tokens + cache_creation were processed this
        # turn (prefill); cache_read was served from cache. So, matching gpt-oss
        # (isl_new = isl - prefix_cache_hits): isl_new = isl - cache_read.
        cache_read = cur_usage.get("cache_read_input_tokens") or 0
        isl = (
            (cur_usage.get("input_tokens") or 0)
            + (cur_usage.get("cache_creation_input_tokens") or 0)
            + cache_read
        )
        writer.write(
            instance_id=instance_id,
            row={
                "ts": round(time.time(), 3),
                "isl": isl,
                "osl": cur_usage.get("output_tokens") or 0,
                "isl_new": isl - cache_read,
                "prefix_cache_hits": cache_read,
            },
        )
        if raw_writer is not None:
            raw_writer.write(
                instance_id=instance_id,
                row={
                    "ts": round(time.time(), 3),
                    "isl_new_text": cur_isl_new,
                    "osl_text": _render_blocks(cur_blocks),
                },
            )

    for line in stdout:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        etype = ev.get("type")
        msg = ev.get("message") or {}
        if etype == "system" and session_sink is not None and ev.get("session_id"):
            session_sink["id"] = ev["session_id"]  # locate the on-disk transcript
        elif etype == "user" and raw:
            pending.append(_render_blocks(msg.get("content")))
        elif etype == "assistant":
            mid = msg.get("id")
            content = msg.get("content") or []
            usage = msg.get("usage") or {}
            if mid != cur_id:  # new turn — finalize the previous one
                flush()
                cur_id, cur_blocks, cur_usage = mid, list(content), usage
                cur_isl_new = "\n".join(pending)
                pending = []
            else:  # same message, next content block
                cur_blocks += content
                if usage:
                    cur_usage = usage
    flush()  # last turn


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


def create_claude_env(model: str, url: str, *, oauth: bool = False) -> dict[str, str]:
    """Env for claude-cli.

    Default (vLLM backend): point it at our local vLLM/proxy via
    ANTHROPIC_BASE_URL and pin every model slot to the served model.

    oauth=True (Anthropic backend): talk DIRECTLY to Anthropic using claude-cli's
    own subscription login. We strip ANTHROPIC_BASE_URL/ANTHROPIC_API_KEY —
    otherwise claude-cli would auth with x-api-key against the override instead
    of the OAuth token — and let --model pick the model."""
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
