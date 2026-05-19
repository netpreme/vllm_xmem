"""Per-task claude subprocess runner + capture-proxy spawning."""
from . import bench_concurrent_users as _bcu


run_claude_task     = _bcu.run_claude_task
start_capture_proxy = _bcu.start_capture_proxy

# Internal shutdown helpers — exposed for callers that want clean teardown
_claudes_kill_all  = _bcu._claudes_kill_all
_proxies_kill_all  = _bcu._proxies_kill_all
_alloc_free_port   = _bcu._alloc_free_port
_log_session_start = _bcu._log_session_start
_stop_capture_proxy = _bcu._stop_capture_proxy


__all__ = [
    "run_claude_task",
    "start_capture_proxy", "_stop_capture_proxy", "_log_session_start",
    "_alloc_free_port",
    "_claudes_kill_all", "_proxies_kill_all",
]
