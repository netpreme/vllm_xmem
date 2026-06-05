"""Clone a SWE-bench target repo at a specific commit.

Prefers a local mirror under ``MIRROR_ROOT`` (created once per unique repo,
e.g. by a pre-mirroring script) — a local clone hardlinks objects and takes
seconds, where cloning big repos from GitHub can exceed the timeout. Falls
back to GitHub when no mirror exists.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

# One bare mirror per unique repo: <MIRROR_ROOT>/<owner>__<name>.git
MIRROR_ROOT = Path("/root/.cache/swe_repo_mirrors")


def clone(problem: dict, workdir: Path) -> Path:
    """Clone <problem['repo']> into <workdir>/repo and checkout
    problem['base_commit']. Returns the repo path."""
    repo = workdir / "repo"
    source = _clone_source(problem["repo"])
    subprocess.run(
        ["git", "clone", "--quiet", source, str(repo)],
        check=True,
        timeout=600,
    )
    subprocess.run(
        ["git", "-C", str(repo), "checkout", "--quiet", problem["base_commit"]],
        check=True,
        timeout=120,
    )
    return repo


def _clone_source(repo: str) -> str:
    """Local mirror path if one exists, else the GitHub URL."""
    mirror = MIRROR_ROOT / f"{repo.replace('/', '__')}.git"
    if (mirror / "HEAD").exists():
        return str(mirror)
    return f"https://github.com/{repo}.git"
