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
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if 200 <= response.status < 300:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.1)
    return False


def get_model_name(url: str) -> str:
    """Ask vLLM which model it's serving — that's what claude-cli sends."""
    with urllib.request.urlopen(f"{url}/v1/models", timeout=2.0) as response:
        return json.loads(response.read())["data"][0]["id"]


# GPU / NVML helpers.


@contextmanager
def nvml_context():
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
        with nvml_context() as nv:
            handle = nv.nvmlDeviceGetHandleByIndex(0)
            return nv.nvmlDeviceGetMemoryInfo(handle).used // (1024 * 1024)
    except pynvml.NVMLError:
        return 0


def gpu_info() -> dict:
    """GPU name / count / total-memory (MiB) of device 0, via NVML; {} if
    unavailable."""
    try:
        with nvml_context() as nv:
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


def get_package_version(package: str) -> str:
    """Installed distribution version of `package` (cheap — no import); '' if
    not installed."""
    try:
        return version(package)
    except PackageNotFoundError:
        return ""


def read_tail(line_count: int = 40) -> str:
    """Last `line_count` lines of the vLLM server log."""
    try:
        return "\n".join(LOG.read_text().splitlines()[-line_count:])
    except OSError:
        return ""


def _read_env_file() -> dict[str, str]:
    """Parse server.sh's .env (KEY=VALUE, ignoring comments) — for display."""
    env_values: dict[str, str] = {}
    try:
        for line in ENV_PATH.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env_values[key.strip()] = value.split("#", 1)[0].strip()
    except OSError:
        pass
    return env_values
