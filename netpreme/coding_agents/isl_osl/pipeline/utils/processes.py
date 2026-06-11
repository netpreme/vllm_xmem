"""Process-tree helpers shared by the pipeline runtime."""

from __future__ import annotations

from collections.abc import Iterable

import psutil


def process_family(root: psutil.Process) -> list[psutil.Process]:
    """Return ``root`` plus descendants, crossing process-group boundaries."""
    try:
        descendants = root.children(recursive=True)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        descendants = []
    return [*descendants, root]


def terminate_process_tree(pid: int, grace_seconds: float) -> list[int]:
    """Terminate one process tree; kill survivors after ``grace_seconds``."""
    try:
        root = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return []
    return terminate_processes(process_family(root), grace_seconds)


def terminate_processes(
    processes: Iterable[psutil.Process],
    grace_seconds: float,
) -> list[int]:
    """Terminate a deduplicated process list; kill survivors after grace."""
    targets = _unique_processes(processes)
    if not targets:
        return []

    for process in targets:
        try:
            process.terminate()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    _, alive = psutil.wait_procs(targets, timeout=grace_seconds)
    for process in alive:
        try:
            process.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return [process.pid for process in targets]


def _unique_processes(processes: Iterable[psutil.Process]) -> list[psutil.Process]:
    seen_pids: set[int] = set()
    unique_processes: list[psutil.Process] = []
    for process in processes:
        if process.pid in seen_pids:
            continue
        seen_pids.add(process.pid)
        unique_processes.append(process)
    return unique_processes
