"""Clone a SWE-bench target repo at a specific commit.

Prefers a local mirror under ``MIRROR_ROOT`` (created once per unique repo,
e.g. by a pre-mirroring script) — a local clone hardlinks objects and takes
seconds, where cloning big repos from GitHub can exceed the timeout. Falls
back to GitHub when no mirror exists.

We fetch the ``base_commit`` BY SHA rather than ``clone`` + ``checkout``. A
plain clone only downloads history reachable from current branch/tag tips, but
some SWE-bench Pro base commits are orphaned from those tips (e.g.
ProtonMail/WebClients rewrites/force-pushes ``main``), so a clone succeeds yet
``checkout <sha>`` dies with git exit 128. GitHub still serves the object by
SHA, so a targeted fetch gets it. ``checkout`` + full-clone is kept as a
fallback for sources that won't serve an arbitrary SHA.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

# One bare mirror per unique repo: <MIRROR_ROOT>/<owner>__<name>.git.
# Under $HOME so it works whatever user runs this (not just root).
MIRROR_ROOT = Path.home() / ".cache/swe_repo_mirrors"


def clone(problem: dict, workdir: Path) -> Path:
    """Clone <problem['repo']> into <workdir>/repo and checkout
    problem['base_commit']. Returns the repo path."""
    repo_dir = workdir / "repo"
    source = _clone_source(problem["repo"])
    base_commit = problem["base_commit"]
    try:
        _fetch_commit(source, repo_dir, base_commit)
    except subprocess.CalledProcessError:
        # Source wouldn't serve the SHA directly (e.g. an incomplete mirror, or
        # a server without allowAnySHA1InWant). Fall back to the old path: full
        # clone, then checkout. Wipe the partial repo first so `clone` sees an
        # empty target.
        shutil.rmtree(repo_dir, ignore_errors=True)
        subprocess.run(
            ["git", "clone", "--quiet", source, str(repo_dir)],
            check=True,
            timeout=600,
        )
        subprocess.run(
            ["git", "-C", str(repo_dir), "checkout", "--quiet", base_commit],
            check=True,
            timeout=120,
        )
    return repo_dir


def _fetch_commit(source: str, repo_dir: Path, base_commit: str) -> None:
    """Init <repo>, then fetch exactly <base_commit> and check it out.
    Works for commits orphaned from every ref, as long as the source serves the
    object by SHA."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "--quiet", str(repo_dir)], check=True, timeout=60)
    subprocess.run(
        ["git", "-C", str(repo_dir), "remote", "add", "origin", source],
        check=True,
        timeout=60,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo_dir),
            "fetch",
            "--depth",
            "1",
            "--quiet",
            "origin",
            base_commit,
        ],
        check=True,
        timeout=600,
    )
    subprocess.run(
        ["git", "-C", str(repo_dir), "checkout", "--quiet", "FETCH_HEAD"],
        check=True,
        timeout=120,
    )


def _clone_source(repo: str) -> str:
    """Local mirror path if one exists, else the GitHub URL."""
    mirror = MIRROR_ROOT / f"{repo.replace('/', '__')}.git"
    try:
        if (mirror / "HEAD").exists():
            return str(mirror)
    except OSError:  # mirror root missing or not accessible — just use GitHub
        pass
    return f"https://github.com/{repo}.git"
