"""Stateless utils for the vLLM server: HTTP probes, GPU/NVML queries,
version/log/.env readers. No dependency on ``Server`` — the lifecycle
class imports from here, never the other way around.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pynvml

# coding_agents/ — pipeline/ is two levels up from this file's package.
SERVER_SH = Path(__file__).resolve().parents[3] / "server.sh"
ENV_PATH = SERVER_SH.parent / ".env"
LOG = Path("/tmp/vllm_server.log")


# HTTP helpers — stdlib urllib (no `requests` dep for a couple one-shot calls).


def check_server_initialized(url: str, timeout: float) -> bool:
    """Poll `url` until it returns a 2xx (service up), or `timeout` s elapse."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if 200 <= r.status < 300:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.1)
    return False


def get_model_name(url: str) -> str:
    """Ask vLLM which model it's serving — that's what claude-cli sends."""
    with urllib.request.urlopen(f"{url}/v1/models", timeout=2.0) as r:
        return json.loads(r.read())["data"][0]["id"]


# GPU / NVML helpers.


@contextmanager
def _nvml():
    """NVML init/shutdown guard; yields the pynvml module. Raises NVMLError if
    the driver/library is unavailable — callers decide the fallback."""
    pynvml.nvmlInit()
    try:
        yield pynvml
    finally:
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


def gpu_used_mib() -> int:
    """GPU 0 memory in use (MiB), via NVML; 0 if it can't be read."""
    try:
        with _nvml() as nv:
            handle = nv.nvmlDeviceGetHandleByIndex(0)
            return nv.nvmlDeviceGetMemoryInfo(handle).used // (1024 * 1024)
    except pynvml.NVMLError:
        return 0


def gpu_info() -> dict:
    """GPU name / count / total-memory (MiB) of device 0, via NVML; {} if
    unavailable."""
    try:
        with _nvml() as nv:
            count = nv.nvmlDeviceGetCount()
            if not count:
                return {}
            name = nv.nvmlDeviceGetName(nv.nvmlDeviceGetHandleByIndex(0))
            if isinstance(name, bytes):  # older nvidia-ml-py returns bytes
                name = name.decode()
            total = nv.nvmlDeviceGetMemoryInfo(nv.nvmlDeviceGetHandleByIndex(0)).total
            return {"name": name, "count": count, "memory_mib": total // (1024 * 1024)}
    except pynvml.NVMLError:
        return {}


# Version / log / .env readers.


def vllm_version() -> str:
    """Installed vLLM distribution version (cheap — no package import); '' if
    not installed."""
    try:
        return version("vllm")
    except PackageNotFoundError:
        return ""


def read_tail(n: int = 40) -> str:
    """Last `n` lines of the vLLM server log."""
    try:
        return "\n".join(LOG.read_text().splitlines()[-n:])
    except OSError:
        return ""


def _read_env_file() -> dict[str, str]:
    """Parse server.sh's .env (KEY=VALUE, ignoring comments) — for display."""
    out: dict[str, str] = {}
    try:
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            out[key.strip()] = val.split("#", 1)[0].strip()
    except OSError:
        pass
    return out
