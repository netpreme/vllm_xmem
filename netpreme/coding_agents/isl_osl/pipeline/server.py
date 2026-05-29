"""vLLM server lifecycle.

``initialize_server`` is a context manager that owns the server for the
duration of the ``with`` block: on enter it clears any stale vLLM, launches
a fresh one (``server.sh``) and blocks until it serves; on exit it kills it
again. Each problem gets its own clean, empty-cache server.

    with initialize_server(VLLM_URL) as model:      # start fresh vLLM
        with Proxy(save_dir, iid, vllm_url=VLLM_URL, ...) as proxy:
            agent.solve(task=task, save_dir=save_dir,
                        model=model, base_url=proxy.base_url)
    # vLLM killed here

All process handling is pure Python (psutil); the only shell file is
``server.sh``, which is just the vLLM launch command.
"""

from __future__ import annotations

import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

import psutil
from loguru import logger
from pipeline import http_utils

# coding_agents/server.sh — pipeline/ is two levels down from there.
SERVER_SH = Path(__file__).resolve().parents[2] / "server.sh"
LOG = Path("/tmp/vllm_server.log")

# vLLM forks an EngineCore worker that escapes a plain process-group kill,
# so (like the old `pkill -f`) we match the whole tree by cmdline substring.
_KILL_PATTERNS = (
    "vllm serve",
    "VLLM::EngineCore",
    "vllm.v1.engine",
    "multiprocessing.resource_tracker",
)
_GPU_RELEASE_TIMEOUT = 60.0
_READY_TIMEOUT = 600.0


def _kill_vllm() -> None:
    killed = []
    for p in psutil.process_iter(["cmdline"]):
        try:
            cmd = " ".join(p.info["cmdline"] or [])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if any(pat in cmd for pat in _KILL_PATTERNS):
            try:
                p.kill()
                killed.append(p.pid)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    if killed:
        logger.info("killed vllm pids {}", killed)


def _wait_port_free(port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket() as s:
            s.settimeout(0.5)
            if s.connect_ex(("localhost", port)) != 0:
                return
        time.sleep(0.5)


def _gpu_used_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        return int(out.splitlines()[0].strip())
    except (subprocess.SubprocessError, ValueError, IndexError):
        return 0


def _wait_gpu_free() -> None:
    # Without this the next vllm can OOM during CUDA-context init.
    deadline = time.monotonic() + _GPU_RELEASE_TIMEOUT
    while _gpu_used_mib() >= 1000 and time.monotonic() < deadline:
        time.sleep(2)


def _stop(port: int) -> None:
    _kill_vllm()
    _wait_port_free(port)
    _wait_gpu_free()


def _tail(path: Path, n: int = 40) -> str:
    try:
        return "\n".join(path.read_text().splitlines()[-n:])
    except OSError:
        return ""


@contextmanager
def initialize_server(url: str):
    """Start a fresh vLLM on enter, kill it on exit; yield the served model."""
    port = urlparse(url).port or 8000
    logger.info("starting vllm at {}", url)
    _stop(port)  # clear any stale server first
    proc = subprocess.Popen(
        ["bash", str(SERVER_SH)],
        stdout=LOG.open("w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    deadline = time.monotonic() + _READY_TIMEOUT
    while not http_utils.check_server_initialized(f"{url}/v1/models", 2.0):
        if proc.poll() is not None:
            raise RuntimeError(f"vllm died on startup; tail of {LOG}:\n{_tail(LOG)}")
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"vllm not ready after {_READY_TIMEOUT:.0f}s (see {LOG})"
            )

    model = http_utils.get_model_name(url)
    logger.info("vllm ready at {}, serving {}", url, model)
    try:
        yield model
    finally:
        logger.info("stopping vllm at {}", url)
        _stop(port)
