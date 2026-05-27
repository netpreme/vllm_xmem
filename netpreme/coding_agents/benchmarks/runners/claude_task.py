"""One Claude Code session against a vLLM backend, optionally tee'd to JSONL.

Three roles in this module:

  * ``CaptureProxy``   — wraps record_proxy.py; tees /v1/messages traffic
                         from claude to the upstream vLLM into a JSONL trace.
  * ``ClaudeTask``     — runs one ``claude -p <prompt>`` subprocess; routes
                         through a CaptureProxy if a CaptureConfig is set.
  * registry helpers   — kill_all_claudes() / kill_all_proxies(), called by
                         the per-level orchestrator on shutdown so worker
                         threads stuck in proc.wait() don't pin Python exit.

CaptureConfig owns the per-level capture state (trace dir, sessions.jsonl
handle, level-start timestamps). Pass it explicitly to ClaudeTask — no
module globals.
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


_SCRIPT_DIR = Path(__file__).resolve().parent
_RECORD_PROXY_SCRIPT = _SCRIPT_DIR / "record_proxy.py"


# ── live process registries (for end-of-level cleanup) ───────────────────────
_running_claudes: set[subprocess.Popen] = set()
_running_claudes_lock = threading.Lock()
_running_proxies: set[subprocess.Popen] = set()
_running_proxies_lock = threading.Lock()


def kill_all_claudes() -> None:
    """Kill every in-flight claude subprocess. Used at end-of-level cleanup —
    otherwise worker threads stuck in proc.wait() block Python exit."""
    with _running_claudes_lock:
        procs = list(_running_claudes)
        _running_claudes.clear()
    for p in procs:
        try: p.kill()
        except Exception: pass


def kill_all_proxies() -> None:
    """Kill every in-flight capture proxy."""
    with _running_proxies_lock:
        procs = list(_running_proxies)
        _running_proxies.clear()
    for p in procs:
        try: p.terminate()
        except Exception: pass
    for p in procs:
        try: p.wait(timeout=3)
        except Exception:
            try: p.kill()
            except Exception: pass


def _alloc_free_port() -> int:
    """OS-assigned ephemeral port. Caller binds soon after — record_proxy.py
    uses SO_REUSEADDR/SO_REUSEPORT, so the small race window is acceptable."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CaptureConfig:
    """Per-level capture state. One instance per `_run_level`."""
    trace_dir: Path
    sessions_file: Path
    sessions_lock: threading.Lock
    t_level_start_mono: float = 0.0
    t_level_start_unix: float = 0.0

    def log_session_start(self, instance_id: str, trace_filename: str,
                          workdir: Path) -> None:
        """Append one record to sessions.jsonl (used for replay scheduling)."""
        t_rel = time.monotonic() - self.t_level_start_mono
        rec = {
            "instance_id":     instance_id,
            "trace_file":      trace_filename,
            "t_session_start": round(t_rel, 4),
            "workdir":         workdir.name,
            "t_wall_start":    time.time(),
        }
        with self.sessions_lock:
            with open(self.sessions_file, "a") as f:
                f.write(json.dumps(rec) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
class CaptureProxy:
    """One per-session record_proxy.py subprocess.

    Spawns record_proxy.py listening on a free local port, forwarding to the
    real vLLM at ``upstream`` and writing JSONL into ``trace_file``. Retries
    up to 3 times with a fresh port if the proxy doesn't come up.
    """

    def __init__(self, trace_file: Path, session_id: str, upstream: str):
        self.trace_file = trace_file
        self.session_id = session_id
        self.upstream = upstream
        self.proc: subprocess.Popen | None = None
        self.port: int | None = None

    def start(self, timeout_s: float = 8.0) -> "CaptureProxy":
        for _attempt in range(3):
            port = _alloc_free_port()
            proc = subprocess.Popen(
                [sys.executable, str(_RECORD_PROXY_SCRIPT),
                 "--port", str(port),
                 "--upstream", self.upstream,
                 "--trace", str(self.trace_file),
                 "--session-id", self.session_id],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )
            t0 = time.monotonic()
            ready = False
            while time.monotonic() - t0 < timeout_s:
                if proc.poll() is not None:
                    break
                line = proc.stdout.readline() if proc.stdout else ""
                if not line:
                    time.sleep(0.05)
                    continue
                if "capture_proxy ready" in line:
                    ready = True
                    break
            if ready:
                self.proc = proc
                self.port = port
                with _running_proxies_lock:
                    _running_proxies.add(proc)
                return self
            try: proc.kill()
            except Exception: pass
            try: proc.wait(timeout=2)
            except Exception: pass
        raise RuntimeError(
            f"capture_proxy failed to start "
            f"(upstream={self.upstream}, trace={self.trace_file})"
        )

    def stop(self) -> None:
        if self.proc is None:
            return
        with _running_proxies_lock:
            _running_proxies.discard(self.proc)
        try: self.proc.terminate()
        except Exception: pass
        try: self.proc.wait(timeout=5)
        except Exception:
            try: self.proc.kill()
            except Exception: pass
        self.proc = None


# ─────────────────────────────────────────────────────────────────────────────
class ClaudeTask:
    """One ``claude -p <problem_statement>`` session against vLLM.

    If ``capture_config`` is provided, spawns a CaptureProxy and routes
    claude through 127.0.0.1:<proxy_port> instead of straight to the vLLM
    base_url. On proxy startup failure, falls back to direct connection
    (best-effort capture).
    """

    def __init__(
        self,
        instance: dict,
        workdir: Path,
        model: str,
        base_url: str,
        capture_config: CaptureConfig | None = None,
    ):
        self.instance = instance
        self.workdir = workdir
        self.model = model
        self.base_url = base_url
        self.capture_config = capture_config

    def run(self) -> tuple[str, bool]:
        instance_id = self.instance["instance_id"]
        proxy: CaptureProxy | None = None
        claude_url = self.base_url

        if self.capture_config is not None:
            trace_path = self.capture_config.trace_dir / f"{instance_id}.jsonl"
            try:
                proxy = CaptureProxy(
                    trace_file=trace_path,
                    session_id=instance_id,
                    upstream=self.base_url,
                ).start()
                claude_url = f"http://127.0.0.1:{proxy.port}"
                self.capture_config.log_session_start(
                    instance_id, trace_path.name, self.workdir
                )
            except Exception:
                # Proxy failed — fall back to direct so the bench still produces
                # results (partial capture is better than aborting the task).
                proxy = None
                claude_url = self.base_url

        env = {
            **os.environ,
            "ANTHROPIC_BASE_URL":             claude_url,
            "ANTHROPIC_API_KEY":              "dummy",
            "ANTHROPIC_AUTH_TOKEN":           "dummy",
            "ANTHROPIC_DEFAULT_OPUS_MODEL":   self.model,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": self.model,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL":  self.model,
        }
        # stdin MUST be /dev/null — claude waits 3s for stdin data otherwise.
        log_path = Path(f"/tmp/claude_{Path(self.workdir).name}.log")
        try:
            with open(log_path, "wb") as flog:
                proc = subprocess.Popen(
                    ["claude", "--model", self.model,
                     "--dangerously-skip-permissions",
                     "-p", self.instance["problem_statement"]],
                    cwd=str(self.workdir),
                    env=env,
                    stdin=subprocess.DEVNULL,
                    stdout=flog,
                    stderr=subprocess.STDOUT,
                )
                with _running_claudes_lock:
                    _running_claudes.add(proc)
                try:
                    rc = proc.wait()
                finally:
                    with _running_claudes_lock:
                        _running_claudes.discard(proc)
            return instance_id, (rc == 0)
        except Exception:
            return instance_id, False
        finally:
            if proxy is not None:
                proxy.stop()
