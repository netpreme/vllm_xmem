"""Per-task SWE-bench workspace setup — clone (or refresh) the target repo
at the instance's base_commit. Thread-safe when each caller passes a
distinct workspace_root (used by parallel setups)."""
import subprocess
from pathlib import Path


DEFAULT_WORKSPACE_ROOT = Path("/tmp/swe_workspaces")


def setup_workspace(instance: dict, workspace_root: Path | None = None) -> Path:
    """Clone or refresh a SWE-bench workspace. Returns the local path."""
    root = workspace_root if workspace_root is not None else DEFAULT_WORKSPACE_ROOT
    instance_id = instance["instance_id"]
    repo        = instance["repo"]
    base_commit = instance["base_commit"]
    workspace   = root / instance_id
    if workspace.exists():
        subprocess.run(["git", "checkout", "-f", base_commit],
                       cwd=workspace, check=True, capture_output=True)
        subprocess.run(["git", "clean", "-fd"],
                       cwd=workspace, check=True, capture_output=True)
    else:
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth=50",
                        f"https://github.com/{repo}.git", str(workspace)],
                       check=True, capture_output=True)
        subprocess.run(["git", "fetch", "--depth=50", "origin", base_commit],
                       cwd=workspace, check=True, capture_output=True)
        subprocess.run(["git", "checkout", base_commit],
                       cwd=workspace, check=True, capture_output=True)
    return workspace
