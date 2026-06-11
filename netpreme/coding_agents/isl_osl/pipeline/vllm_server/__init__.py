"""vLLM server: lifecycle context manager (``server.Server``) + stateless
helpers (``utils``). Import from here; the split into submodules is an
implementation detail.
"""

from __future__ import annotations

from pipeline.vllm_server.server import Server
from pipeline.vllm_server.utils import (
    SERVER_SH,
    _read_env_file,
    check_server_initialized,
    get_model_name,
    get_package_version,
    gpu_info,
    gpu_used_mib,
    read_tail,
)

__all__ = [
    "Server",
    "check_server_initialized",
    "get_model_name",
    "get_package_version",
    "gpu_info",
    "gpu_used_mib",
    "read_tail",
    "_read_env_file",
    "SERVER_SH",
]
