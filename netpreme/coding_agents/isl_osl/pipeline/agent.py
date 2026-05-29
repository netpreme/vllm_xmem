"""The coding agent — solve one SWE-bench task, and record the result.

``coding_agent`` is the whole agent: given a task, a sandbox dir, a served
model and a base URL, it clones the repo and drives claude-cli over it,
returning the exit code. ``Sandbox`` is the throwaway workspace + timer for
one problem, and ``write_meta`` records its metadata (the resume ledger).
None of these restart vLLM, attach the proxy, nor analyse anything — those
are the runner's (main.py) concern.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from loguru import logger
from pipeline import claude
from pipeline import git_repo as git
from pipeline.jsonl import instance_dir


class Sandbox:
    """Throwaway workspace + wall-clock timer for one problem.

    Creates a fresh temp dir under `root` on enter (``.dir``) and removes it
    on exit; records ``.started`` / ``.ended`` around the ``with`` body."""

    def __init__(self, root: Path, prefix: str) -> None:
        self.root = root
        self.prefix = prefix
        self.dir: Path | None = None
        self.started = 0.0
        self.ended = 0.0

    def __enter__(self) -> Sandbox:
        self.root.mkdir(parents=True, exist_ok=True)
        self.dir = Path(tempfile.mkdtemp(prefix=self.prefix, dir=self.root))
        self.started = time.time()
        return self

    def __exit__(self, *exc) -> bool:
        self.ended = time.time()
        if self.dir:
            shutil.rmtree(self.dir, ignore_errors=True)
        return False


def coding_agent(task: dict, sandbox_dir: Path, model: str, base_url: str) -> int:
    """Clone the task's repo into `sandbox_dir` and drive claude-cli over it.

    Returns the exit code; subprocess failures (clone or claude) map to a
    non-zero code."""
    try:
        repo = git.clone(task, sandbox_dir)
        return claude.solve(task, repo, model=model, url=base_url)
    except subprocess.SubprocessError as exc:
        logger.error("{}: clone/solve failed: {!r}", task["instance_id"], exc)
        return 1


def write_meta(
    save_dir: Path, task: dict, started_at: float, ended_at: float, exit_code: int
) -> None:
    """Record one problem's run metadata under telemetry/<iid>/meta.json."""
    iid = task["instance_id"]
    problem_dir = instance_dir(save_dir / "telemetry", iid)
    problem_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "instance_id": iid,
        "difficulty": task.get("difficulty"),
        "repo": task.get("repo"),
        "base_commit": task.get("base_commit"),
        "started_at": round(started_at, 3),
        "ended_at": round(ended_at, 3),
        "exit_code": exit_code,
    }
    (problem_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
