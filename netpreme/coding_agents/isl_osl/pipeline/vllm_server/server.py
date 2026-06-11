"""vLLM server lifecycle.

``Server`` is a context manager that owns the server for the duration of
the ``with`` block: on enter it clears any stale vLLM, launches a fresh one
(``server.sh``) and blocks until it serves; on exit it kills it. Each
problem gets its own clean, empty-cache server.

    with Server(VLLM_URL, model=...) as served:     # start fresh vLLM
        with Proxy(save_dir, iid, vllm_url=VLLM_URL, ...) as proxy:
            agent.solve(task=task, model=served, base_url=proxy.base_url)
    # vLLM killed here

Launch knobs (model, tensor-parallel size, …) are constructor args; any left
``None`` fall back to ``server.sh``'s .env/defaults, with a one-time warning.
All process handling is pure Python (psutil); the only shell file is
``server.sh``, the vLLM launch command. Stateless helpers (HTTP probes, GPU
queries, .env parsing) live in ``utils.py``.
"""

from __future__ import annotations

import contextlib
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import psutil
from loguru import logger

from pipeline.utils.processes import (
    process_family,
    process_listening_on_port,
    processes_with_env,
    terminate_processes,
    unique_processes,
)
from pipeline.vllm_server.utils import (
    LOG,
    SERVER_SH,
    _read_env_file,
    check_server_initialized,
    get_model_name,
    gpu_used_mib,
    read_tail,
)


class Server:
    """Start a fresh vLLM on enter, kill it on exit; yields the served model.

    Launch knobs map to the env vars server.sh reads; any left ``None`` fall
    back to server.sh's .env/defaults (warned once). ``server_args`` forward
    straight to `vllm serve`."""

    _ENV_PORT = "CODING_AGENTS_VLLM_PORT"
    _GPU_RELEASE_TIMEOUT = 60.0
    _READY_TIMEOUT = 600.0
    _TERM_GRACE_S = 5.0  # let vLLM unlink its /dev/shm IPC before we SIGKILL

    def __init__(
        self,
        url: str,
        *,
        remote: bool = False,
        model: str | None = None,
        tensor_parallel_size: int | None = None,
        max_model_len: int | None = None,
        gpu_memory_utilization: float | None = None,
        tool_call_parser: str | None = None,
        server_args: tuple[str, ...] = (),
    ) -> None:
        # remote=True (Anthropic backend): no local vLLM. claude-cli talks
        # directly to Anthropic via its own OAuth; we only carry `url` (the
        # Anthropic API URL) and `model` through so the uniform with-block
        # works for both backends.
        self.remote = remote
        self.url = url
        self.port = urlparse(url).port or 8000
        self.model = model
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tool_call_parser = tool_call_parser
        self.server_args = list(server_args)
        self._proc: subprocess.Popen | None = None

    # -- lifecycle ---------------------------------------------------------
    def __enter__(self) -> Server:
        if self.remote:
            logger.info(
                "anthropic backend at {} — model={} (no local vllm)",
                self.url,
                self.model,
            )
            return self
        logger.info("starting vllm at {}", self.url)
        self._stop()  # clear any stale server first
        self._proc = subprocess.Popen(
            ["bash", str(SERVER_SH), *self.server_args],
            env=self._env(),
            stdout=LOG.open("w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        deadline = time.monotonic() + self._READY_TIMEOUT
        while not check_server_initialized(f"{self.url}/v1/models", 2.0):
            if self._proc.poll() is not None:
                raise RuntimeError(
                    f"vllm died on startup; tail of {LOG}:\n{read_tail()}"
                )
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"vllm not ready after {self._READY_TIMEOUT:.0f}s (see {LOG})"
                )

        # Pin the actual served model so callers can read `server.model`.
        self.model = self.model or get_model_name(self.url)
        logger.info(
            "vllm ready at {} — {}",
            self.url,
            " ".join(f"{key}={value}" for key, value in self.serving_config().items()),
        )
        return self

    def __exit__(self, *exc) -> bool:
        if self.remote:
            return False
        logger.info("stopping vllm at {}", self.url)
        self._stop()
        return False

    def _env(self) -> dict[str, str]:
        """server.sh env: set each provided knob; the rest fall back to
        server.sh's .env/defaults. PORT always comes from the url."""
        env = os.environ.copy()
        knobs = {
            "MODEL_NAME": self.model,
            "TENSOR_PARALLEL_SIZE": self.tensor_parallel_size,
            "MAX_MODEL_LEN": self.max_model_len,
            "GPU_MEMORY_UTILIZATION": self.gpu_memory_utilization,
            "TOOL_CALL_PARSER": self.tool_call_parser,
        }
        for env_name, value in knobs.items():
            if value is not None:
                env[env_name] = str(value)
        env["PORT"] = str(self.port)
        env[self._ENV_PORT] = str(self.port)
        # Launch vLLM with the SAME interpreter that's running this process —
        # i.e. whatever env the user invoked main.py with — so server.sh never
        # hardcodes a venv path. (Our anthropic-serving patches live in that
        # env's editable vLLM.)
        env["VLLM_PYTHON"] = sys.executable
        return env

    def serving_config(self) -> dict[str, str]:
        """Effective serving knobs (caller arg → os env → .env) — for logging
        and run metadata."""
        dotenv = _read_env_file()

        def pick(env_name: str, value: object) -> str:
            if value is not None:
                return str(value)
            return (
                os.environ.get(env_name)
                or dotenv.get(env_name)
                or "(server.sh default)"
            )

        return {
            "model": pick("MODEL_NAME", self.model),
            "tp": pick("TENSOR_PARALLEL_SIZE", self.tensor_parallel_size),
            "max_model_len": pick("MAX_MODEL_LEN", self.max_model_len),
            "gpu_util": pick("GPU_MEMORY_UTILIZATION", self.gpu_memory_utilization),
            "tool_call_parser": pick("TOOL_CALL_PARSER", self.tool_call_parser),
            "port": str(self.port),
        }

    # -- process control ---------------------------------------------------
    def _stop(self) -> None:
        killed = self._kill_vllm()
        self._wait_port_free()
        self._wait_gpu_free()
        if killed:
            self._clean_shm()

    def _kill_vllm(self) -> int:
        processes = self._server_processes()
        if not processes:
            return 0

        # SIGTERM first so vLLM can unlink its own /dev/shm/psm_* IPC segments;
        # SIGKILL whatever ignores it within the grace window.
        killed_pids = terminate_processes(processes, self._TERM_GRACE_S)
        logger.info("killed vllm pids {}", killed_pids)
        return len(killed_pids)

    def _server_processes(self) -> list[psutil.Process]:
        """Processes owned by this harness on this configured server port.

        Three discovery strategies (unioned): the launched pid, processes
        env-tagged with this port, and whoever holds the LISTEN socket — then
        each expanded to its full family and deduplicated."""
        roots: list[psutil.Process] = []
        if self._proc is not None and self._proc.poll() is None:
            with contextlib.suppress(psutil.NoSuchProcess, psutil.AccessDenied):
                roots.append(psutil.Process(self._proc.pid))
        roots.extend(processes_with_env(self._ENV_PORT, str(self.port)))
        port_owner = process_listening_on_port(self.port)
        if port_owner is not None:
            roots.append(port_owner)
        return unique_processes(
            process for root in roots for process in process_family(root)
        )

    def _clean_shm(self) -> None:
        """Remove orphaned vLLM IPC segments from /dev/shm.

        vLLM uses POSIX shared memory (/dev/shm/psm_*) for EngineCore↔worker
        IPC and leaks it when SIGKILLed. Left to pile up they fill /dev/shm (a
        64 MiB container default is common), and the NEXT vLLM dies on its
        first request — the engine inits on the GPU fine, then can't allocate
        IPC shm. We run this only after killing a server owned by this harness."""
        removed = 0
        for shared_memory_segment in Path("/dev/shm").glob("psm_*"):
            try:
                shared_memory_segment.unlink()
                removed += 1
            except OSError:
                pass
        if removed:
            logger.info("cleaned {} orphaned /dev/shm vllm segment(s)", removed)

    def _wait_port_free(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with socket.socket() as sock:
                sock.settimeout(0.5)
                if sock.connect_ex(("localhost", self.port)) != 0:
                    return
            time.sleep(0.5)

    def _wait_gpu_free(self) -> None:
        # Without this the next vllm can OOM during CUDA-context init.
        deadline = time.monotonic() + self._GPU_RELEASE_TIMEOUT
        while gpu_used_mib() >= 1000 and time.monotonic() < deadline:
            time.sleep(2)
