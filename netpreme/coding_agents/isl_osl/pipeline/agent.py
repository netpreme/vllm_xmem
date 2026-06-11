"""The coding agent — solve one SWE-bench task.

``coding_agent`` is the whole agent: given a task, a sandbox dir, a served
model and a base URL, it clones the repo and drives claude-cli over it,
returning the exit code. It does not restart vLLM, attach the proxy, manage
the sandbox, nor record metadata — those are the runner's (main.py) concern.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from loguru import logger
from pipeline import claude
from pipeline.utils import git_repo as git


def coding_agent(
    task: dict,
    sandbox_dir: Path,
    model: str,
    base_url: str,
    timeout_s: float = claude.DEFAULT_TIMEOUT_S,
    *,
    oauth: bool = False,
    telemetry_dir: Path | None = None,
) -> int:
    """Clone the task's repo into `sandbox_dir` and drive claude-cli over it.

    Returns the exit code; subprocess failures (clone or claude) map to a
    non-zero code. `timeout_s` bounds the claude session (see claude.solve).
    `oauth`/`telemetry_dir` select the Anthropic-subscription backend, where
    claude.solve copies the session transcript (no proxy); counts and per-turn
    texts are derived from it later by the analysis pass."""
    try:
        repo = git.clone(task, sandbox_dir)
        return claude.solve(
            task,
            repo,
            model=model,
            url=base_url,
            timeout_s=timeout_s,
            oauth=oauth,
            telemetry_dir=telemetry_dir,
        )
    except (subprocess.SubprocessError, OSError) as exc:
        logger.error("{}: clone/solve failed: {!r}", task["instance_id"], exc)
        return 1
