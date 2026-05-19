"""Server-lifecycle helpers: vLLM, Prometheus, GPU exporter, MTier reset.

All functions re-exported from the original bench_concurrent_users module —
imports here so callers can use a single `utils.lifecycle.start_vllm(...)`
form without depending on the entrypoint script.
"""
from . import bench_concurrent_users as _bcu


# vLLM lifecycle
start_vllm  = _bcu.start_vllm
stop_vllm   = _bcu.stop_vllm

# Prometheus lifecycle
start_prometheus = _bcu.start_prometheus
stop_prometheus  = _bcu.stop_prometheus
take_snapshot    = _bcu.take_snapshot

# GPU exporter lifecycle
ensure_gpu_exporter = _bcu.ensure_gpu_exporter
stop_gpu_exporter   = _bcu.stop_gpu_exporter

# Analyzer wrapper
run_analyzer = _bcu.run_analyzer


__all__ = [
    "start_vllm", "stop_vllm",
    "start_prometheus", "stop_prometheus", "take_snapshot",
    "ensure_gpu_exporter", "stop_gpu_exporter",
    "run_analyzer",
]
