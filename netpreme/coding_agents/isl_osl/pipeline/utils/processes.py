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


def processes_with_env(var: str, value: str) -> list[psutil.Process]:
    """Processes whose environment has ``var == value`` (e.g. a port tag)."""
    matched: list[psutil.Process] = []
    for process in psutil.process_iter(["pid"]):
        try:
            if process.environ().get(var) == value:
                matched.append(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return matched


def process_listening_on_port(port: int) -> psutil.Process | None:
    """The process holding a LISTEN socket on ``port``; None if none/unreadable."""
    try:
        connections = psutil.net_connections(kind="inet")
    except (psutil.AccessDenied, psutil.Error):
        return None
    for connection in connections:
        if connection.pid is None or connection.status != psutil.CONN_LISTEN:
            continue
        if connection.laddr and connection.laddr.port == port:
            try:
                return psutil.Process(connection.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return None
    return None


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
    targets = unique_processes(processes)
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


def unique_processes(processes: Iterable[psutil.Process]) -> list[psutil.Process]:
    seen_pids: set[int] = set()
    unique_processes: list[psutil.Process] = []
    for process in processes:
        if process.pid in seen_pids:
            continue
        seen_pids.add(process.pid)
        unique_processes.append(process)
    return unique_processes
