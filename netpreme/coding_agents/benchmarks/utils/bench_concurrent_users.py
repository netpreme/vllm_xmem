#!/usr/bin/env python3
"""
Concurrency sweep with per-level Prometheus TSDB snapshots.

For every concurrency level C:
  1. Wipe + start Prometheus (with admin API enabled)
  2. Start vLLM server (cold cache; mtier reset before mtier setups)
  3. Launch C concurrent `claude -p "<swe_bench_problem>"` users; replace each
     user as it finishes; keep N users active for --sustained-mins
  4. POST /api/v1/admin/tsdb/snapshot and copy it into results/<setup>/c<N>/
  5. Stop vLLM + mtier reset
  6. Stop Prometheus

Each level's results dir contains exactly two things:
    config.json     — setup, concurrency, model, t_start_unix, t_end_unix, ...
    prom_snapshot/  — portable TSDB; open with `prometheus --storage.tsdb.path=...`

Determinism: server uses temperature=0, seed=42, VLLM_BATCH_INVARIANT=1
(set in start_server.sh).

Usage:
    python3 bench_concurrent_users.py --setup hybrid-mtier --concurrency 1 10 12 14 16 --sustained-mins 30
    python3 bench_concurrent_users.py --setup hybrid-cpu   --concurrency 1 10 12 14 16 --sustained-mins 30
    python3 bench_concurrent_users.py --concurrency 1 10 12 14 16   # both setups back-to-back
"""

import argparse
import atexit
import concurrent.futures
import itertools
import json
import os
import queue as _queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR      = Path(__file__).resolve().parent
BENCH_ROOT      = SCRIPT_DIR.parent
AGENT_ROOT      = BENCH_ROOT.parent
ENV_FILE        = AGENT_ROOT / ".env"
WORKSPACE_ROOT  = Path("/tmp/swe_workspaces")
RESULTS_DIR     = BENCH_ROOT / "results_benchmarks"
START_SCRIPT    = AGENT_ROOT / "start_server.sh"
MONITORING_DIR  = AGENT_ROOT / "monitoring"
PROM_CONFIG     = MONITORING_DIR / "prometheus.yml"
PROM_DATA_DIR   = Path("/tmp/prometheus_data")
PROM_URL        = "http://localhost:9090"

DEFAULT_MODEL            = "qwen/qwen3-coder-30b-a3b-instruct-fp8"
DEFAULT_PORT             = "8000"
DEFAULT_SUSTAINED_MIN_S  = 1800     # 30 min
SERVER_STARTUP_TIMEOUT_S = 360
PROM_STARTUP_TIMEOUT_S   = 30
CLONE_WORKERS            = 16

# ── live progress counter ─────────────────────────────────────────────────────
_live: dict = {}
_live_lock  = threading.Lock()

# ── active resources (set in main, used by cleanup) ───────────────────────────
_active_port:    "int | None"               = None
_active_setup:   "str | None"               = None
_prom_proc:      "subprocess.Popen | None"  = None
_gpu_exp_proc:   "subprocess.Popen | None"  = None


def _redraw_live() -> None:
    with _live_lock:
        c      = _live.get("c", "?")
        cfg    = _live.get("cfg", "")
        elap   = _live.get("elapsed", 0.0)
        done   = _live.get("done", 0)
        ok     = _live.get("ok", 0)
        fail   = _live.get("fail", 0)
        active = _live.get("active", 0)
        cloned = _live.get("cloned", 0)
    print(
        f"\r  C={c} [{cfg}]  {elap:>5.0f}s  "
        f"done={done}  ok={ok}  fail={fail}  active={active}  cloned={cloned}   ",
        end="", flush=True,
    )


# ── env loader ────────────────────────────────────────────────────────────────

def load_env(path: Path) -> None:
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            if key.strip() and key.strip() not in os.environ:
                os.environ[key.strip()] = value.strip()


# ── GPU exporter lifecycle (started once, runs for the whole sweep) ──────────

def ensure_gpu_exporter() -> "subprocess.Popen | None":
    """Start gpu_exporter.py on :9092 if not already running. Returns Popen or None."""
    try:
        r = requests.get("http://localhost:9092/metrics", timeout=1)
        if r.status_code == 200:
            print("  [gpu_exp] Already running on :9092", flush=True)
            return None
    except Exception:
        pass
    exporter = MONITORING_DIR / "gpu_exporter.py"
    if not exporter.exists():
        print(f"  [gpu_exp] WARNING: {exporter} not found — GPU metrics will be missing",
              flush=True)
        return None
    log = Path("/tmp/gpu_exporter.log")
    proc = subprocess.Popen(
        [sys.executable, str(exporter), "--port", "9092", "--interval", "1.0"],
        stdout=open(log, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    # Wait briefly for the HTTP server to come up
    for _ in range(10):
        time.sleep(0.5)
        try:
            r = requests.get("http://localhost:9092/metrics", timeout=1)
            if r.status_code == 200:
                print(f"  [gpu_exp] Started on :9092  PID={proc.pid}  log={log}", flush=True)
                return proc
        except Exception:
            pass
    print(f"  [gpu_exp] WARNING: did not become ready (see {log})", flush=True)
    return proc


def stop_gpu_exporter(proc: "subprocess.Popen | None") -> None:
    if proc is None:
        return
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass


# ── analyzer runner (called once per level after the snapshot is saved) ──────

ANALYZER_SCRIPT = AGENT_ROOT / "analysis" / "analyze_snapshot.py"
VENV_PYTHON    = Path("/home/ubuntu/vllm_xmem/.venv/bin/python")


def run_analyzer(level_dir: Path, port: int = 9099) -> None:
    """Invoke analyze_snapshot.py on the just-saved level directory.
    Best-effort: failures are logged but don't abort the sweep."""
    if not ANALYZER_SCRIPT.exists():
        return
    py = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
    print(f"  [analyze] running on {level_dir.name} ...", flush=True)
    try:
        r = subprocess.run(
            [py, str(ANALYZER_SCRIPT), str(level_dir), "--port", str(port)],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode == 0:
            print(f"  [analyze] → {level_dir}/analysis/  (fig1, fig2)", flush=True)
        else:
            tail = "\n".join((r.stdout + r.stderr).splitlines()[-5:])
            print(f"  [analyze] FAILED rc={r.returncode}:\n{tail}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"  [analyze] timed out after 120s", flush=True)
    except Exception as e:
        print(f"  [analyze] error: {e}", flush=True)


# ── Prometheus lifecycle (per level: wipe → start → snapshot → stop) ──────────

def start_prometheus() -> "subprocess.Popen | None":
    """Reuse an existing Prometheus on :9090 if one is healthy (so an external
    monitoring stack keeps running between bench invocations). Otherwise start
    a fresh one with admin API enabled."""
    try:
        r = requests.get(f"{PROM_URL}/-/ready", timeout=2)
        if r.status_code == 200:
            print(f"  [prom] Reusing existing Prometheus on :9090", flush=True)
            return None
    except Exception:
        pass

    subprocess.run(["pkill", "-f", "prometheus --config.file"], capture_output=True)
    time.sleep(2)
    if PROM_DATA_DIR.exists():
        shutil.rmtree(PROM_DATA_DIR, ignore_errors=True)
    PROM_DATA_DIR.mkdir(parents=True, exist_ok=True)

    log = Path("/tmp/prometheus.log")
    print(f"  [prom] Starting fresh Prometheus  data={PROM_DATA_DIR}  log={log}", flush=True)
    proc = subprocess.Popen(
        [
            "prometheus",
            f"--config.file={PROM_CONFIG}",
            f"--storage.tsdb.path={PROM_DATA_DIR}",
            "--web.enable-admin-api",
            "--web.listen-address=:9090",
        ],
        stdout=open(log, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    t0 = time.monotonic()
    while True:
        try:
            r = requests.get(f"{PROM_URL}/-/ready", timeout=2)
            if r.status_code == 200:
                print(f"  [prom] Ready  ({time.monotonic()-t0:.0f}s)  PID={proc.pid}", flush=True)
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            raise RuntimeError(f"Prometheus died during startup (see {log})")
        if time.monotonic() - t0 > PROM_STARTUP_TIMEOUT_S:
            proc.terminate()
            raise RuntimeError(f"Prometheus startup timed out after {PROM_STARTUP_TIMEOUT_S}s")
        time.sleep(1)


def stop_prometheus(proc: "subprocess.Popen | None") -> None:
    if proc is None:
        # External Prometheus (e.g. from monitoring stack) — leave it alone.
        return
    print(f"  [prom] Stopping ...", flush=True)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(15):
        time.sleep(1)
        try:
            os.killpg(os.getpgid(proc.pid), 0)
        except ProcessLookupError:
            break
    subprocess.run(["pkill", "-9", "-f", "prometheus --config.file"], capture_output=True)


def take_snapshot(out_dir: Path) -> str:
    """POST snapshot endpoint, copy hardlinked snapshot dir into out_dir/prom_snapshot."""
    r = requests.post(f"{PROM_URL}/api/v1/admin/tsdb/snapshot", timeout=60)
    r.raise_for_status()
    name = r.json()["data"]["name"]
    src  = PROM_DATA_DIR / "snapshots" / name
    dst  = out_dir / "prom_snapshot"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return name


# ── vLLM server lifecycle ─────────────────────────────────────────────────────

_GREEN  = "\033[32m"
_RED    = "\033[31m"
_DIM    = "\033[2m"
_OFF    = "\033[0m"
# Per-setup colors used everywhere — purple for mtier, orange for cpu.
_PURPLE = "\033[38;5;141m"
_ORANGE = "\033[38;5;208m"


def _setup_short(setup: str) -> str:
    if "mtier" in setup: return "Mtier"
    if "cpu"   in setup: return "CPU"
    if "hbm"   in setup: return "HBM"
    return setup


def _setup_color(setup: str) -> str:
    if "mtier" in setup: return _PURPLE
    if "cpu"   in setup: return _ORANGE
    return ""


def _vllm_label(port: int, gpus: str | None, setup: str) -> str:
    """`[vllm:8001:GPU0+Mtier]` colored by setup."""
    color = _setup_color(setup)
    g = f"GPU{gpus}" if gpus is not None else "GPU?"
    return f"{color}[vllm:{port}:{g}+{_setup_short(setup)}]{_OFF}"


def start_vllm(
    setup: str,
    port: int,
    tp: int | None = None,
    gpu_util: float | None = None,
    gpus: str | None = None,
    max_num_seqs: int | None = None,
) -> subprocess.Popen:
    if not START_SCRIPT.exists():
        raise FileNotFoundError(f"Server script not found: {START_SCRIPT}")
    log = Path(f"/tmp/vllm_server_{port}.log")
    label = _vllm_label(port, gpus, setup)
    print(f"  {label} starting  max_num_seqs={max_num_seqs or 'default'}  "
          f"{_DIM}(log: {log}){_OFF}", flush=True)

    env = {**os.environ}
    env["PORT"] = str(port)
    if tp is not None:           env["TENSOR_PARALLEL_SIZE"]   = str(tp)
    if gpu_util is not None:     env["GPU_MEMORY_UTILIZATION"] = str(gpu_util)
    if gpus is not None:         env["CUDA_VISIBLE_DEVICES"]   = gpus
    if max_num_seqs is not None: env["MAX_NUM_SEQS"]           = str(max_num_seqs)

    proc = subprocess.Popen(
        ["bash", str(START_SCRIPT), f"--{setup}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )

    t0 = time.monotonic()
    last_print = 0.0
    while True:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                elapsed = time.monotonic() - t0
                print(f"\r  {label}  {_GREEN}✓ ready{_OFF} in {elapsed:.0f}s  "
                      f"PID={proc.pid}                              ", flush=True)
                return proc
        except Exception:
            pass
        if proc.poll() is not None:
            print(f"\r  {label}  {_RED}✗ died during startup{_OFF}  "
                  f"see log: {log}", flush=True)
            raise RuntimeError(f"vLLM server died during startup (see {log})")
        if time.monotonic() - t0 > SERVER_STARTUP_TIMEOUT_S:
            proc.terminate()
            print(f"\r  {label}  {_RED}✗ startup timed out "
                  f"after {SERVER_STARTUP_TIMEOUT_S}s{_OFF}  see log: {log}",
                  flush=True)
            raise RuntimeError(f"vLLM server startup timed out after {SERVER_STARTUP_TIMEOUT_S}s")
        now = time.monotonic()
        if now - last_print >= 3.0:
            print(f"\r  {label}  {int(now - t0):>3}s waiting  "
                  f"{_DIM}(waiting for /health){_OFF}     ", end="", flush=True)
            last_print = now
        time.sleep(1)


def stop_vllm(proc: subprocess.Popen, setup: str, port: int) -> None:
    print(f"  [vllm] Stopping  setup={setup}  port={port} ...", flush=True)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass
    for _ in range(30):
        time.sleep(1)
        try:
            os.killpg(os.getpgid(proc.pid), 0)
        except ProcessLookupError:
            break
    subprocess.run(
        ["sh", "-c", f"ss -tlnp 'sport = :{port}' | grep -oP 'pid=\\K[0-9]+' | xargs -r kill -9"],
        capture_output=True,
    )
    time.sleep(2)
    if "mtier" in setup:
        subprocess.run(["sh", "-c", "echo yes | mtier_service reset 2>/dev/null || true"],
                       capture_output=True)
        time.sleep(2)
    print(f"  [vllm] Stopped.", flush=True)


# ── workspace setup ───────────────────────────────────────────────────────────

def setup_workspace(instance: dict, workspace_root: Path | None = None) -> Path:
    """Clone or refresh a SWE-bench workspace.
    workspace_root may be passed explicitly (required for thread-safe parallel
    setups that each use their own workspace dir)."""
    root = workspace_root if workspace_root is not None else WORKSPACE_ROOT
    instance_id = instance["instance_id"]
    repo        = instance["repo"]
    base_commit = instance["base_commit"]
    workspace   = root / instance_id
    if workspace.exists():
        subprocess.run(["git", "checkout", "-f", base_commit],
                       cwd=workspace, check=True, capture_output=True)
        subprocess.run(["git", "clean", "-fd"],
                       cwd=workspace, check=True, capture_output=True)
    else:
        root.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth=50",
                        f"https://github.com/{repo}.git", str(workspace)],
                       check=True, capture_output=True)
        subprocess.run(["git", "fetch", "--depth=50", "origin", base_commit],
                       cwd=workspace, check=True, capture_output=True)
        subprocess.run(["git", "checkout", base_commit],
                       cwd=workspace, check=True, capture_output=True)
    return workspace


# ── claude task runner ────────────────────────────────────────────────────────

# Track every Popen spawned by run_claude_task so the level orchestrator can
# kill them on shutdown (otherwise their worker threads block Python exit and
# the bench process hangs after the level finishes).
_RUNNING_CLAUDES: set = set()
_RUNNING_CLAUDES_LOCK = threading.Lock()

# Capture-mode state. Populated by main() when --capture-traces is set.
# When None, run_claude_task points claude directly at the vLLM base_url.
# When set, run_claude_task spawns a per-session proxy and routes claude through it.
_CAPTURE_CFG: "dict | None" = None
# Capture proxy lifetime is per-task. Track them so cleanup can kill stragglers.
_RUNNING_PROXIES: set = set()
_RUNNING_PROXIES_LOCK = threading.Lock()


def _claudes_kill_all() -> None:
    """Kill every in-flight claude subprocess. Used at end-of-level cleanup."""
    with _RUNNING_CLAUDES_LOCK:
        procs = list(_RUNNING_CLAUDES)
        _RUNNING_CLAUDES.clear()
    for p in procs:
        try:
            p.kill()
        except Exception:
            pass


def _proxies_kill_all() -> None:
    """Kill every in-flight capture proxy. Used at end-of-level cleanup."""
    with _RUNNING_PROXIES_LOCK:
        procs = list(_RUNNING_PROXIES)
        _RUNNING_PROXIES.clear()
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass
    for p in procs:
        try:
            p.wait(timeout=3)
        except Exception:
            try: p.kill()
            except Exception: pass


def _alloc_free_port() -> int:
    """Get an OS-assigned ephemeral port. Caller binds soon after — small race
    window is acceptable; capture_proxy uses SO_REUSEADDR/SO_REUSEPORT."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


_CAPTURE_PROXY_SCRIPT = SCRIPT_DIR / "record_proxy.py"


def start_capture_proxy(trace_file: Path, session_id: str, upstream: str,
                        timeout_s: float = 8.0) -> tuple[subprocess.Popen, int]:
    """Spawn capture_proxy.py and return (process, port). Raises on startup failure."""
    for attempt in range(3):
        port = _alloc_free_port()
        proc = subprocess.Popen(
            [sys.executable, str(_CAPTURE_PROXY_SCRIPT),
             "--port", str(port),
             "--upstream", upstream,
             "--trace", str(trace_file),
             "--session-id", session_id],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        # Read until we see the "ready" line (or process dies).
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
            with _RUNNING_PROXIES_LOCK:
                _RUNNING_PROXIES.add(proc)
            return proc, port
        # Failed — kill and retry with a new port.
        try: proc.kill()
        except Exception: pass
        try: proc.wait(timeout=2)
        except Exception: pass
    raise RuntimeError(f"capture_proxy failed to start (upstream={upstream}, trace={trace_file})")


def _stop_capture_proxy(proc: subprocess.Popen) -> None:
    with _RUNNING_PROXIES_LOCK:
        _RUNNING_PROXIES.discard(proc)
    try:
        proc.terminate()
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        try: proc.kill()
        except Exception: pass


def _log_session_start(capture_cfg: dict, instance_id: str,
                       trace_filename: str, workdir: Path) -> None:
    """Append a session-start record to sessions.jsonl for replay scheduling."""
    t_rel = time.monotonic() - capture_cfg["t_level_start_mono"]
    rec = {
        "instance_id":      instance_id,
        "trace_file":       trace_filename,
        "t_session_start":  round(t_rel, 4),
        "workdir":          workdir.name,
        "t_wall_start":     time.time(),
    }
    with capture_cfg["sessions_lock"]:
        with open(capture_cfg["sessions_file"], "a") as f:
            f.write(json.dumps(rec) + "\n")


def run_claude_task(instance: dict, workdir: Path, model: str, base_url: str) -> tuple[str, bool]:
    """Run a single claude -p session on a SWE-bench problem. Return (instance_id, ok).

    If _CAPTURE_CFG is set, spawn a per-session capture proxy and route claude
    through it. The proxy forwards to `base_url` (the real vLLM) and writes a
    JSONL trace to <trace_dir>/<instance_id>.jsonl."""
    proxy_proc: "subprocess.Popen | None" = None
    claude_url = base_url
    instance_id = instance["instance_id"]

    if _CAPTURE_CFG is not None:
        trace_dir = Path(_CAPTURE_CFG["trace_dir"])
        trace_path = trace_dir / f"{instance_id}.jsonl"
        try:
            proxy_proc, port = start_capture_proxy(
                trace_file=trace_path,
                session_id=instance_id,
                upstream=base_url,
            )
            claude_url = f"http://127.0.0.1:{port}"
            _log_session_start(_CAPTURE_CFG, instance_id, trace_path.name, workdir)
        except Exception:
            # If proxy fails to start, fall back to direct connection rather than
            # killing the task — the bench is still useful even with partial capture.
            proxy_proc = None
            claude_url = base_url

    env = {
        **os.environ,
        "ANTHROPIC_BASE_URL":             claude_url,
        "ANTHROPIC_API_KEY":              "dummy",
        "ANTHROPIC_AUTH_TOKEN":           "dummy",
        "ANTHROPIC_DEFAULT_OPUS_MODEL":   model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL":  model,
    }
    # stdin MUST be redirected from /dev/null — without it claude waits 3s for
    # stdin data ("Warning: no stdin data received in 3s") and may hang longer
    # when launched via a pipeline. Log stdout+stderr per task for diagnosis.
    log_path = Path(f"/tmp/claude_{Path(workdir).name}.log")
    try:
        with open(log_path, "wb") as flog:
            proc = subprocess.Popen(
                ["claude", "--model", model,
                 "--dangerously-skip-permissions",
                 "-p", instance["problem_statement"]],
                cwd=str(workdir),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=flog,
                stderr=subprocess.STDOUT,
            )
            with _RUNNING_CLAUDES_LOCK:
                _RUNNING_CLAUDES.add(proc)
            try:
                rc = proc.wait()
            finally:
                with _RUNNING_CLAUDES_LOCK:
                    _RUNNING_CLAUDES.discard(proc)
        return instance_id, (rc == 0)
    except Exception:
        return instance_id, False
    finally:
        if proxy_proc is not None:
            _stop_capture_proxy(proxy_proc)


# ── per-level orchestration ───────────────────────────────────────────────────

def _run_setup_pool(
    spec: dict,
    concurrency: int,
    instances: list[dict],
    duration_s: float,
    model: str,
    t_start_mono: float,
    no_clone: bool,
    shared_state: dict | None = None,
) -> dict:
    """Clone workspaces + run claude pool for ONE setup. Designed to be called
    from a thread so multiple setups can run their pools in parallel.
    `shared_state[setup_label]` (if given) is updated with running counters."""
    setup          = spec["setup"]
    base_url       = spec["base_url"]
    workspace_root = spec["workspace_root"]

    initial = instances[:concurrency]
    rest    = instances[concurrency:]
    work_q: "_queue.Queue[dict | None]" = _queue.Queue()
    stop_clone = threading.Event()

    if no_clone:
        initial_specs = [{"instance": inst, "workdir": Path.cwd()} for inst in initial]
        for inst in rest:
            work_q.put({"instance": inst, "workdir": Path.cwd()})
    else:
        initial_specs = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(CLONE_WORKERS, max(1, len(initial)))
        ) as cex:
            fmap = {cex.submit(setup_workspace, inst, workspace_root): inst for inst in initial}
            for f in concurrent.futures.as_completed(fmap):
                inst = fmap[f]
                try:
                    wd = f.result()
                except Exception:
                    wd = Path.cwd()
                initial_specs.append({"instance": inst, "workdir": wd})

        def _bg_clone(all_inst=rest or instances, q=work_q, stop=stop_clone):
            idx = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=CLONE_WORKERS) as cex:
                while not stop.is_set():
                    if not all_inst:
                        break
                    inst = all_inst[idx % len(all_inst)]
                    idx += 1
                    try:
                        wd = cex.submit(setup_workspace, inst, workspace_root).result()
                    except Exception:
                        wd = Path.cwd()
                    if stop.is_set():
                        break
                    q.put({"instance": inst, "workdir": wd})

        threading.Thread(target=_bg_clone, daemon=True).start()

    n_done = n_ok = n_fail = 0
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=concurrency)
    active: dict = {}
    spec_iter = iter(initial_specs)

    def _next_spec() -> "dict | None":
        s = next(spec_iter, None)
        if s is not None:
            return s
        if time.monotonic() - t_start_mono >= duration_s:
            return None
        try:
            return work_q.get(timeout=1.0)
        except _queue.Empty:
            return None

    def _submit(s: dict):
        f = ex.submit(run_claude_task, s["instance"], s["workdir"], model, base_url)
        active[f] = s

    for _ in range(concurrency):
        s = _next_spec()
        if s is None:
            break
        _submit(s)

    def _update_shared():
        if shared_state is not None:
            shared_state[setup] = {"n_done": n_done, "n_ok": n_ok, "n_fail": n_fail,
                                   "n_active": len(active)}

    _update_shared()
    try:
        while active and time.monotonic() - t_start_mono < duration_s:
            done, _ = concurrent.futures.wait(
                list(active.keys()), timeout=1.0,
                return_when=concurrent.futures.FIRST_COMPLETED,
            )
            for fut in done:
                try:
                    _, ok = fut.result()
                except Exception:
                    ok = False
                del active[fut]
                n_done += 1
                if ok: n_ok  += 1
                else:  n_fail += 1
                if time.monotonic() - t_start_mono < duration_s:
                    s = _next_spec()
                    if s is not None:
                        _submit(s)
            _update_shared()
    finally:
        if not no_clone:
            stop_clone.set()
        for f in list(active.keys()):
            f.cancel()
        # Kill in-flight claude subprocesses BEFORE shutdown(wait=...).
        # Otherwise worker threads stuck in proc.wait() prevent Python exit.
        _claudes_kill_all()
        _proxies_kill_all()
        ex.shutdown(wait=True, cancel_futures=True)
        _update_shared()

    return {
        "setup":             setup,
        "n_tasks_started":   n_done,
        "n_tasks_completed": n_ok,
        "n_tasks_failed":    n_fail,
    }


def _q_one(query: str):
    """Single-value Prometheus query. Returns float or None."""
    try:
        r = requests.get(f"{PROM_URL}/api/v1/query",
                         params={"query": query}, timeout=3)
        d = r.json().get("data", {}).get("result", [])
        if not d:
            return None
        v = d[0]["value"][1]
        return None if v in ("NaN", "+Inf", "-Inf") else float(v)
    except Exception:
        return None


def _query_setup_metrics(spec: dict, window: str = "30s") -> dict:
    """Query all live metrics for one setup over `window`."""
    from urllib.parse import urlparse as _up
    sel = '{instance="localhost:%s"}' % _up(spec["base_url"]).port
    return {
        "ttft":      _q_one(f'histogram_quantile(0.5,  sum by (le) (rate(vllm:time_to_first_token_seconds_bucket{sel}[{window}])))'),
        "itl":       _q_one(f'histogram_quantile(0.5,  sum by (le) (rate(vllm:inter_token_latency_seconds_bucket{sel}[{window}])))'),
        "e2e":       _q_one(f'histogram_quantile(0.5,  sum by (le) (rate(vllm:e2e_request_latency_seconds_bucket{sel}[{window}])))'),
        "out_tps":   _q_one(f'sum(rate(vllm:generation_tokens_total{sel}[{window}]))'),
        "hbm_use":   _q_one(f'avg(vllm:kv_cache_usage_perc{sel})'),
        "hbm_hit":   _q_one(f'rate(vllm:prefix_cache_hits_total{sel}[{window}]) / clamp_min(rate(vllm:prefix_cache_queries_total{sel}[{window}]), 1e-9)'),
        "off_hit":   _q_one(f'rate(vllm:external_prefix_cache_hits_total{sel}[{window}]) / clamp_min(rate(vllm:prefix_cache_queries_total{sel}[{window}]), 1e-9)'),
        "turns":     _q_one(f'sum(vllm:e2e_request_latency_seconds_count{sel})'),
    }


def _fmt_status_line(spec: dict, m: dict, task_counters: dict) -> str:
    """`[GPU0+Mtier]  tasks=4 ok=2 turns=37 ttft=89ms itl=12ms e2e=2.3s ...`"""
    color = _setup_color(spec["setup"])
    gpu   = f"GPU{spec.get('gpus', '?')}"
    name  = _setup_short(spec["setup"])
    def ms(v):  return f"{int(v*1000)}ms" if v is not None else "-"
    def s(v):   return f"{v:.1f}s"        if v is not None else "-"
    def tps(v): return f"{v:.0f}"         if v is not None else "-"
    def pct(v): return f"{v*100:.1f}%"    if v is not None else "-"
    turns = int(m["turns"]) if m["turns"] is not None else 0
    return (
        f"{color}[{gpu}+{name:5s}]{_OFF}  "
        f"tasks={task_counters.get('n_active',0):>2}/{task_counters.get('n_done',0):>3}  "
        f"turns={turns:>5}  "
        f"ttft={ms(m['ttft']):>6}  itl={ms(m['itl']):>5}  e2e={s(m['e2e']):>6}  "
        f"out={tps(m['out_tps']):>4}t/s  "
        f"HBM_use={pct(m['hbm_use']):>6}  HBM_hit={pct(m['hbm_hit']):>6}  off_hit={pct(m['off_hit']):>5}"
    )


def _live_status_thread(setup_specs: list[dict], shared_state: dict,
                        t_start_mono: float, stop: threading.Event,
                        interval: float = 1.0) -> None:
    """Print all setups' live status periodically. Uses ANSI cursor-up to
    overwrite in place if stdout is a TTY; falls back to non-overwriting lines
    every 10 s when piped to a file (so logs remain readable)."""
    is_tty = sys.stdout.isatty()
    if not is_tty:
        interval = 10.0  # less spam when logging to a file
    n = len(setup_specs)
    first = True
    while not stop.wait(interval):
        elapsed = int(time.monotonic() - t_start_mono)
        if is_tty:
            if not first:
                sys.stdout.write(f"\033[{n+1}A")
            first = False
            sys.stdout.write(f"\033[2K\r  {_DIM}── live  t={elapsed:>3}s  (window=30s rolling){_OFF}\n")
            for spec in setup_specs:
                m  = _query_setup_metrics(spec)
                tc = shared_state.get(spec["setup"], {})
                sys.stdout.write(f"\033[2K\r  {_fmt_status_line(spec, m, tc)}\n")
            sys.stdout.flush()
        else:
            # File log: emit a small block (no cursor games)
            print(f"\n  ── live  t={elapsed:>3}s  (window=30s rolling)", flush=True)
            for spec in setup_specs:
                m  = _query_setup_metrics(spec)
                tc = shared_state.get(spec["setup"], {})
                print(f"  {_fmt_status_line(spec, m, tc)}", flush=True)


def _print_averages(setup_specs: list[dict], t_start_unix: float, t_end_unix: float,
                    per_setup: dict) -> None:
    """End-of-level: print per-setup AVERAGES over the run window."""
    duration_s = max(15, int(t_end_unix - t_start_unix))
    window = f"{duration_s}s"
    print(f"\n  {_DIM}── Averages  (over {duration_s}s window){_OFF}", flush=True)
    for spec in setup_specs:
        # Use sum/count style averages over the full window so they reflect
        # the WHOLE level, not just the last 30 s.
        from urllib.parse import urlparse as _up
        sel = '{instance="localhost:%s"}' % _up(spec["base_url"]).port
        avg_ttft = _q_one(f'increase(vllm:time_to_first_token_seconds_sum{sel}[{window}]) / clamp_min(increase(vllm:time_to_first_token_seconds_count{sel}[{window}]), 1)')
        avg_itl  = _q_one(f'increase(vllm:inter_token_latency_seconds_sum{sel}[{window}]) / clamp_min(increase(vllm:inter_token_latency_seconds_count{sel}[{window}]), 1)')
        avg_e2e  = _q_one(f'increase(vllm:e2e_request_latency_seconds_sum{sel}[{window}]) / clamp_min(increase(vllm:e2e_request_latency_seconds_count{sel}[{window}]), 1)')
        avg_tps  = _q_one(f'rate(vllm:generation_tokens_total{sel}[{window}])')
        avg_hbm  = _q_one(f'avg_over_time(vllm:kv_cache_usage_perc{sel}[{window}])')
        hbm_hit  = _q_one(f'increase(vllm:prefix_cache_hits_total{sel}[{window}]) / clamp_min(increase(vllm:prefix_cache_queries_total{sel}[{window}]), 1)')
        off_hit  = _q_one(f'increase(vllm:external_prefix_cache_hits_total{sel}[{window}]) / clamp_min(increase(vllm:prefix_cache_queries_total{sel}[{window}]), 1)')
        turns    = _q_one(f'sum(increase(vllm:e2e_request_latency_seconds_count{sel}[{window}]))')
        n_ok     = per_setup.get(spec["setup"], {}).get("n_tasks_completed", 0)
        color = _setup_color(spec["setup"])
        gpu   = f"GPU{spec.get('gpus', '?')}"
        name  = _setup_short(spec["setup"])
        def ms(v):  return f"{int(v*1000)}ms" if v is not None else "-"
        def s_(v):  return f"{v:.1f}s"        if v is not None else "-"
        def tps(v): return f"{v:.0f}"         if v is not None else "-"
        def pct(v): return f"{v*100:.1f}%"    if v is not None else "-"
        print(
            f"  {color}[{gpu}+{name:5s}]{_OFF}  "
            f"tasks_done={n_ok:>3}  turns={int(turns) if turns else 0:>5}  "
            f"avg ttft={ms(avg_ttft):>6}  itl={ms(avg_itl):>5}  e2e={s_(avg_e2e):>6}  "
            f"out={tps(avg_tps):>4}t/s  "
            f"HBM_use={pct(avg_hbm):>6}  HBM_hit={pct(hbm_hit):>6}  off_hit={pct(off_hit):>5}",
            flush=True,
        )


def run_level(
    setup_specs: list[dict],
    concurrency: int,
    instances: list[dict],
    duration_s: float,
    model: str,
    out_dir: Path,
    no_clone: bool,
) -> dict:
    """Run all setups in parallel for one concurrency level.
    Shows per-setup live status every 1s while running, then per-setup averages."""
    labels = " | ".join(_setup_short(s["setup"]) for s in setup_specs)
    print(f"\n  ── C={concurrency:>3}  parallel: [{labels}]  "
          f"duration={duration_s:.0f}s ──", flush=True)

    t_start_unix = time.time()
    t_start_mono = time.monotonic()

    # If capture is enabled, anchor session-start timestamps to this level's start.
    if _CAPTURE_CFG is not None:
        _CAPTURE_CFG["t_level_start_mono"] = t_start_mono
        _CAPTURE_CFG["t_level_start_unix"] = t_start_unix

    shared_state: dict[str, dict] = {s["setup"]: {} for s in setup_specs}

    # Live status thread (1s, in-place updates)
    status_stop = threading.Event()
    threading.Thread(target=_live_status_thread,
                     args=(setup_specs, shared_state, t_start_mono, status_stop),
                     daemon=True).start()

    # Per-setup claude pools (parallel)
    results_lock = threading.Lock()
    per_setup: dict[str, dict] = {}

    def _run(spec=None):
        r = _run_setup_pool(spec, concurrency, instances, duration_s,
                            model, t_start_mono, no_clone, shared_state)
        with results_lock:
            per_setup[spec["setup"]] = r

    threads = []
    for spec in setup_specs:
        t = threading.Thread(target=_run, kwargs={"spec": spec})
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    status_stop.set()
    time.sleep(0.5)  # let the status thread drain its last redraw
    t_end_unix = time.time()
    print("", flush=True)  # newline after the in-place status

    _print_averages(setup_specs, t_start_unix, t_end_unix, per_setup)

    return {
        "t_start_unix":      t_start_unix,
        "t_end_unix":        t_end_unix,
        "duration_s":        round(t_end_unix - t_start_unix, 2),
        "setups":            per_setup,
    }




# ── cleanup on exit / Ctrl-C ──────────────────────────────────────────────────

def _cleanup() -> None:
    if _active_port is not None:
        print(f"\n  [cleanup] Killing vLLM on port {_active_port} ...", flush=True)
        subprocess.run(["fuser", "-k", f"{_active_port}/tcp"], capture_output=True)
    if _active_setup and "mtier" in _active_setup:
        print("  [cleanup] Resetting MTier ...", flush=True)
        subprocess.run(["sh", "-c", "echo yes | mtier_service reset 2>/dev/null || true"],
                       capture_output=True)
    if _prom_proc is not None:
        print("  [cleanup] Stopping Prometheus ...", flush=True)
        try:
            stop_prometheus(_prom_proc)
        except Exception:
            pass
    if _gpu_exp_proc is not None:
        print("  [cleanup] Stopping GPU exporter ...", flush=True)
        try:
            stop_gpu_exporter(_gpu_exp_proc)
        except Exception:
            pass
    print("  [cleanup] Done.", flush=True)


def _signal_handler(sig, frame):
    _cleanup()
    sys.exit(0)


atexit.register(_cleanup)
signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1, 10, 12, 14, 16],
                    help="Concurrency levels to sweep")
    ap.add_argument("--setup", nargs="+", default=["hybrid-cpu", "hybrid-mtier"],
                    help="Server setup label(s); multiple values run sequentially")
    ap.add_argument("--sustained-mins", type=float, default=DEFAULT_SUSTAINED_MIN_S / 60,
                    help="Wall-clock cap per level (default: 30 min)")
    ap.add_argument("--start",   type=int, default=0,    help="SWE-bench dataset start index")
    ap.add_argument("--end",     type=int, default=None, help="SWE-bench dataset end index")
    ap.add_argument("--difficulty", nargs="+", default=None,
                    help="Filter by SWE-bench difficulty: easy/medium/hard/vhard or full labels")
    ap.add_argument("--port",    default=None)
    ap.add_argument("--model",   default=None)
    ap.add_argument("--tp",      type=int,   default=None)
    ap.add_argument("--gpu-util",type=float, default=None)
    ap.add_argument("--gpus",    type=str,   default=None)
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="vLLM --max-num-seqs; defaults to current concurrency level")
    ap.add_argument("--workspace-root", type=str, default=None,
                    help="Base dir for cloned task repos (default: /tmp/swe_workspaces_p<port>)")
    ap.add_argument("--no-clone", action="store_true",
                    help="Skip repo setup; run claude in cwd")
    ap.add_argument("--capture-traces", type=str, default=None,
                    help="If set, spawn a per-claude capture proxy and write trace JSONLs into "
                         "this directory (one file per session + sessions.jsonl + capture_meta.json). "
                         "Only meaningful with a single --setup and --concurrency level.")
    args = ap.parse_args()

    duration_s = args.sustained_mins * 60

    load_env(ENV_FILE)
    model = args.model or os.environ.get("MODEL", DEFAULT_MODEL)

    # Per-setup defaults: mtier → port 8001 / GPU 0, cpu → port 8002 / GPU 1.
    # Used to pick (port, gpus) per setup inside the sweep loop so that running
    # both setups in one invocation gives Prometheus distinct instance labels.
    _SETUP_DEFAULTS = {
        "hybrid-mtier": (8001, "0"),
        "mtier-only":   (8001, "0"),
        "hybrid-cpu":   (8002, "1"),
        "cpu-only":     (8002, "1"),
    }

    # ── SWE-bench Verified, ordered v-hard → hard → medium → easy ─────────────
    from datasets import load_dataset
    print("Loading SWE-bench Verified ...", flush=True)
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

    _DIFF_ALIASES = {
        "easy":   "<15 min fix",
        "medium": "15 min - 1 hour",
        "hard":   "1-4 hours",
        "vhard":  ">4 hours",
    }
    _DIFF_RANK = {
        ">4 hours":        0,   # v-hard first
        "1-4 hours":       1,
        "15 min - 1 hour": 2,
        "<15 min fix":     3,
    }
    if args.difficulty:
        wanted = {_DIFF_ALIASES.get(d, d) for d in args.difficulty}
        ds = ds.filter(lambda row: row["difficulty"] in wanted)
        print(f"  Difficulty filter: {wanted}  →  {len(ds)} tasks", flush=True)

    rows = sorted(
        ds,
        key=lambda r: (_DIFF_RANK.get(r["difficulty"], 9), -len(r["problem_statement"])),
    )
    print(f"  Task order: v-hard → hard → medium → easy, longest problem first within tier",
          flush=True)

    end = args.end if args.end is not None else len(rows)
    instances = rows[args.start:end]

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Start GPU exporter once for the whole sweep (per-level Prometheus
    # scrapes :9092 to record GPU SM util / mem util / power per second).
    global _gpu_exp_proc
    _gpu_exp_proc = ensure_gpu_exporter()

    # ── Build per-setup specs (port, gpus, base_url, workspace_root) ─────────
    setup_specs: list[dict] = []
    for setup in args.setup:
        auto_port, auto_gpus = _SETUP_DEFAULTS.get(setup, (None, None))
        if args.port and len(args.setup) == 1:
            port = int(args.port)
        else:
            port = int(auto_port or os.environ.get("PORT", DEFAULT_PORT))
        if args.gpus and len(args.setup) == 1:
            gpus_for_setup = args.gpus
        else:
            gpus_for_setup = auto_gpus
        setup_specs.append({
            "setup":          setup,
            "port":           port,
            "gpus":           gpus_for_setup,
            "base_url":       f"http://localhost:{port}",
            "workspace_root": Path(args.workspace_root or f"/tmp/swe_workspaces_p{port}"),
        })

    completed_levels: list[dict] = []

    # Set cleanup state. _active_port/_setup are used by atexit cleanup; with
    # parallel mode they point to "the most recent" setup, which is fine.
    global _active_port, _active_setup
    _active_port  = setup_specs[0]["port"] if setup_specs else None
    _active_setup = setup_specs[0]["setup"] if setup_specs else None

    # Reset MTier once up-front if any setup uses it
    if any("mtier" in s["setup"] for s in setup_specs):
        print(f"\n  [mtier] Resetting MTier memory ...", flush=True)
        subprocess.run(["sh", "-c", "echo yes | mtier_service reset 2>/dev/null || true"],
                       capture_output=True)
        time.sleep(2)

    run_dir = RESULTS_DIR / f"bench_sweep_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    _tp_disp       = args.tp       if args.tp       is not None else "default (1)"
    _gpu_util_disp = args.gpu_util if args.gpu_util is not None else "default (0.9)"
    print(f"\n{'═'*70}")
    print(f"  Setup        : {' | '.join(s['setup'] for s in setup_specs)}")
    print(f"  Model        : {model}")
    print(f"  TP           : {_tp_disp}")
    print(f"  GPU util     : {_gpu_util_disp}")
    print(f"  Levels       : {args.concurrency}")
    print(f"  Duration/lvl : {args.sustained_mins:.1f} min")
    print(f"  Difficulty   : {args.difficulty or 'all'}")
    print(f"  Dataset      : SWE-bench Verified  ({len(instances)} tasks)")
    print(f"  Output       : {run_dir}/")
    _bi = os.environ.get("VLLM_BATCH_INVARIANT", "1")
    print(f"  Determinism  : VLLM_BATCH_INVARIANT={_bi}  seed=42  temperature=0")
    if args.capture_traces:
        print(f"  Capture      : ON  →  {args.capture_traces}/")
        if len(setup_specs) != 1:
            print(f"  {_RED}WARN{_OFF}: --capture-traces with >1 setup is unusual — only one will be replayable")
    print(f"{'═'*70}")

    for concurrency in args.concurrency:
        level_dir = run_dir / f"c{concurrency:03d}"
        level_dir.mkdir(parents=True, exist_ok=True)

        # ── Capture config: per-level traces/ dir + sessions.jsonl ──
        global _CAPTURE_CFG
        capture_dir: "Path | None" = None
        if args.capture_traces:
            capture_dir = Path(args.capture_traces).expanduser().resolve()
            if len(args.concurrency) > 1:
                capture_dir = capture_dir / f"c{concurrency:03d}"
            capture_dir.mkdir(parents=True, exist_ok=True)
            (capture_dir / "traces").mkdir(exist_ok=True)
            sessions_file = capture_dir / "sessions.jsonl"
            sessions_file.write_text("")  # truncate
            _CAPTURE_CFG = {
                "trace_dir":           capture_dir / "traces",
                "sessions_file":       sessions_file,
                "sessions_lock":       threading.Lock(),
                "t_level_start_mono":  0.0,  # set when run_level starts
                "t_level_start_unix":  0.0,
            }
        else:
            _CAPTURE_CFG = None

        # 1) Fresh Prometheus (admin API enabled)
        global _prom_proc
        _prom_proc = start_prometheus()

        # 2) Start all vLLMs IN PARALLEL (one per GPU). If any fails we abort.
        max_seqs = args.max_num_seqs if args.max_num_seqs is not None else concurrency
        vllm_procs: dict[str, subprocess.Popen] = {}
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(setup_specs)) as ex:
                fmap = {
                    ex.submit(start_vllm, s["setup"], s["port"],
                              tp=args.tp, gpu_util=args.gpu_util,
                              gpus=s["gpus"], max_num_seqs=max_seqs): s
                    for s in setup_specs
                }
                for f in concurrent.futures.as_completed(fmap):
                    s = fmap[f]
                    vllm_procs[s["setup"]] = f.result()  # raises on startup failure
        except Exception as e:
            print(f"  {_RED}✗ vLLM startup failed:{_OFF} {e}", flush=True)
            for proc in vllm_procs.values():
                try: proc.terminate()
                except Exception: pass
            stop_prometheus(_prom_proc)
            _prom_proc = None
            raise

        try:
            # 3) Run claude pools for all setups in parallel
            level_meta = run_level(
                setup_specs, concurrency, instances, duration_s,
                model, level_dir, args.no_clone,
            )

            # 4) Snapshot (single Prometheus contains all setups, distinguishable by `setup` label)
            print(f"  [prom] Snapshotting TSDB → {level_dir}/prom_snapshot/ ...", flush=True)
            snap_name = take_snapshot(level_dir)
            print(f"  [prom] Snapshot saved (name={snap_name})", flush=True)

            # 5) config.json
            config = {
                "concurrency":    concurrency,
                "model":          model,
                "tp":             args.tp,
                "gpu_util":       args.gpu_util,
                "max_num_seqs":   max_seqs,
                "sustained_mins": args.sustained_mins,
                "difficulty":     args.difficulty,
                "swe_bench_pool_size": len(instances),
                "determinism": {
                    "VLLM_BATCH_INVARIANT": int(os.environ.get("VLLM_BATCH_INVARIANT", "1") or 0),
                    "seed":                  42,
                    "temperature":           0,
                },
                "snapshot_name": snap_name,
                "setups": [
                    {"setup": s["setup"], "port": s["port"], "gpus": s["gpus"]}
                    for s in setup_specs
                ],
                **level_meta,
            }
            (level_dir / "config.json").write_text(json.dumps(config, indent=2))
            print(f"  → {level_dir}/  (config.json + prom_snapshot/)", flush=True)

            # If capture was on, emit capture_meta.json next to traces/.
            if capture_dir is not None and _CAPTURE_CFG is not None:
                # Count sessions actually captured (lines in sessions.jsonl).
                sf = _CAPTURE_CFG["sessions_file"]
                try:
                    n_sessions = sum(1 for _ in open(sf))
                except Exception:
                    n_sessions = 0
                capture_meta = {
                    "kind":            "capture_level",
                    "concurrency":     concurrency,
                    "model":           model,
                    "setup":           setup_specs[0]["setup"] if setup_specs else None,
                    "vllm_port":       setup_specs[0]["port"]  if setup_specs else None,
                    "gpus":            setup_specs[0]["gpus"]  if setup_specs else None,
                    "sustained_mins":  args.sustained_mins,
                    "n_sessions":      n_sessions,
                    "t_level_start_unix": _CAPTURE_CFG["t_level_start_unix"],
                    "t_level_end_unix":   level_meta["t_end_unix"],
                    "duration_s":      level_meta["duration_s"],
                    "determinism": {
                        "VLLM_BATCH_INVARIANT": int(os.environ.get("VLLM_BATCH_INVARIANT", "1") or 0),
                        "seed":                  42,
                        "temperature":           0,
                    },
                    "sweep_run_dir":   str(level_dir),
                }
                (capture_dir / "capture_meta.json").write_text(
                    json.dumps(capture_meta, indent=2))
                print(f"  → capture: {capture_dir}/  "
                      f"({n_sessions} sessions, traces/, capture_meta.json)", flush=True)

            completed_levels.append({
                "concurrency": concurrency,
                "dir":         level_dir,
                "duration_s":  level_meta["duration_s"],
            })
        finally:
            # 6) Stop all vLLMs (parallel)
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(vllm_procs)) as ex:
                for s in setup_specs:
                    proc = vllm_procs.get(s["setup"])
                    if proc is not None:
                        ex.submit(stop_vllm, proc, s["setup"], s["port"])
            # 7) Stop Prometheus
            stop_prometheus(_prom_proc)
            _prom_proc = None
            # 8) Auto-run analyzer on the just-saved snapshot
            run_analyzer(level_dir)

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'═'*72}")
    print(f"  Sweep complete  ({len(completed_levels)} level(s))")
    print(f"{'═'*72}")
    for L in completed_levels:
        fig1 = L["dir"] / "analysis" / "fig1_timeseries.png"
        fig2 = L["dir"] / "analysis" / "fig2_offload_dominant.png"
        ok = fig1.exists() and fig2.exists()
        mark = f"{_GREEN}✓{_OFF}" if ok else f"{_RED}✗{_OFF}"
        print(f"  {mark}  C={L['concurrency']:>3}  "
              f"{L['duration_s']:>5.0f}s  →  {L['dir']}")
    if completed_levels:
        print(f"\n  Figures: <level>/analysis/fig1_timeseries.png  +  fig2_offload_dominant.png")
        print(f"  Re-analyze any level:  "
              f"{VENV_PYTHON} {ANALYZER_SCRIPT} <level_dir>")
    print(f"{'═'*72}\n")


if __name__ == "__main__":
    main()
