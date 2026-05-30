"""Clone a SWE-bench target repo at a specific commit."""

from __future__ import annotations

import subprocess
from pathlib import Path


def clone(problem: dict, workdir: Path) -> Path:
    """Clone github.com/<problem['repo']> into <workdir>/repo and
    checkout problem['base_commit']. Returns the repo path."""
    repo = workdir / "repo"
    subprocess.run(
        [
            "git",
            "clone",
            "--quiet",
            f"https://github.com/{problem['repo']}.git",
            str(repo),
        ],
        check=True,
        timeout=600,
    )
    subprocess.run(
        ["git", "-C", str(repo), "checkout", "--quiet", problem["base_commit"]],
        check=True,
        timeout=120,
    )
    return repo
