"""Start / stop a long-running sidecar subprocess (labeler, watcher)."""

from __future__ import annotations

import subprocess
from pathlib import Path


def start(name: str, argv: list[str], log_path: Path):
    """Launch `argv` in its own session, redirect stdout/stderr to log_path."""
    log = log_path.open("w")
    proc = subprocess.Popen(
        argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    print(f"[run] started {name} (pid {proc.pid})")
    return proc, log


def stop(name: str, sidecar) -> None:
    """SIGTERM then wait; SIGKILL if it doesn't exit in 10 s."""
    proc, log = sidecar
    if proc.poll() is None:
        print(f"[run] stopping {name} (pid {proc.pid})")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    log.close()
