"""Lifecycle classes for the long-lived subprocesses the bench spawns.

Each class owns one Popen and exposes start()/stop(). State is encapsulated
on the instance — no module globals.

    VLLMServer            — one vLLM server bound to (setup, port, gpus)
    Prometheus            — the per-level Prometheus instance + TSDB snapshot
    GPUMetricsRecorder    — the sweep-wide gpu_metrics_recorder.py exporter
    MTier                 — stateless reset() helper for the MTier device
"""
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import requests

from utils.colors import DIM, GREEN, OFF, RED, vllm_label

# ── shared paths ──────────────────────────────────────────────────────────────
_BENCH_ROOT     = Path(__file__).resolve().parent        # netpreme/.../benchmarks
_AGENT_ROOT     = _BENCH_ROOT.parent                     # netpreme/.../coding_agents
START_SCRIPT    = _AGENT_ROOT / "server.sh"
MONITORING_DIR  = _AGENT_ROOT / "monitoring"
PROM_CONFIG     = MONITORING_DIR / "prometheus.yml"
PROM_DATA_DIR   = Path(os.environ.get(
    "PROM_DATA_DIR",
    str(Path.home() / "monitoring_state" / "prometheus_data"),
))
PROM_URL        = "http://localhost:9090"

SERVER_STARTUP_TIMEOUT_S = 360
PROM_STARTUP_TIMEOUT_S   = 30


# ─────────────────────────────────────────────────────────────────────────────
class VLLMServer:
    """One vLLM server. Spawn via server.sh, block until /health is 200."""

    def __init__(
        self,
        setup: str,
        port: int,
        tp: int | None = None,
        gpu_util: float | None = None,
        gpus: str | None = None,
        max_num_seqs: int | None = None,
    ):
        self.setup = setup
        self.port = port
        self.tp = tp
        self.gpu_util = gpu_util
        self.gpus = gpus
        self.max_num_seqs = max_num_seqs
        self.proc: subprocess.Popen | None = None
        self.log = Path(f"/tmp/vllm_server_{port}.log")
        self.label = vllm_label(port, gpus, setup)

    def start(self) -> "VLLMServer":
        if not START_SCRIPT.exists():
            raise FileNotFoundError(f"Server script not found: {START_SCRIPT}")
        print(f"  {self.label} starting  "
              f"max_num_seqs={self.max_num_seqs or 'default'}  "
              f"{DIM}(log: {self.log}){OFF}", flush=True)

        env = {**os.environ, "PORT": str(self.port)}
        if self.tp           is not None: env["TENSOR_PARALLEL_SIZE"]   = str(self.tp)
        if self.gpu_util     is not None: env["GPU_MEMORY_UTILIZATION"] = str(self.gpu_util)
        if self.gpus         is not None: env["CUDA_VISIBLE_DEVICES"]   = self.gpus
        if self.max_num_seqs is not None: env["MAX_NUM_SEQS"]           = str(self.max_num_seqs)

        self.proc = subprocess.Popen(
            ["bash", str(START_SCRIPT), f"--{self.setup}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            env=env,
        )
        self._wait_until_healthy()
        return self

    def _wait_until_healthy(self) -> None:
        t0 = time.monotonic()
        last_print = 0.0
        while True:
            try:
                r = requests.get(f"http://localhost:{self.port}/health", timeout=2)
                if r.status_code == 200:
                    elapsed = time.monotonic() - t0
                    print(f"\r  {self.label}  {GREEN}✓ ready{OFF} in {elapsed:.0f}s  "
                          f"PID={self.proc.pid}                              ",
                          flush=True)
                    return
            except Exception:
                pass
            if self.proc.poll() is not None:
                print(f"\r  {self.label}  {RED}✗ died during startup{OFF}  "
                      f"see log: {self.log}", flush=True)
                raise RuntimeError(f"vLLM server died during startup (see {self.log})")
            if time.monotonic() - t0 > SERVER_STARTUP_TIMEOUT_S:
                self.proc.terminate()
                print(f"\r  {self.label}  {RED}✗ startup timed out "
                      f"after {SERVER_STARTUP_TIMEOUT_S}s{OFF}  see log: {self.log}",
                      flush=True)
                raise RuntimeError(
                    f"vLLM server startup timed out after {SERVER_STARTUP_TIMEOUT_S}s"
                )
            now = time.monotonic()
            if now - last_print >= 3.0:
                print(f"\r  {self.label}  {int(now - t0):>3}s waiting  "
                      f"{DIM}(waiting for /health){OFF}     ", end="", flush=True)
                last_print = now
            time.sleep(1)

    def stop(self) -> None:
        if self.proc is None:
            return
        print(f"  [vllm] Stopping  setup={self.setup}  port={self.port} ...", flush=True)
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(30):
            time.sleep(1)
            try:
                os.killpg(os.getpgid(self.proc.pid), 0)
            except ProcessLookupError:
                break
        subprocess.run(
            ["sh", "-c",
             f"ss -tlnp 'sport = :{self.port}' | "
             f"grep -oP 'pid=\\K[0-9]+' | xargs -r kill -9"],
            capture_output=True,
        )
        time.sleep(2)
        if "mtier" in self.setup:
            MTier.reset()
            time.sleep(2)
        print(f"  [vllm] Stopped.", flush=True)
        self.proc = None


# ─────────────────────────────────────────────────────────────────────────────
class Prometheus:
    """Per-level Prometheus instance + TSDB snapshot.

    If an external Prometheus is already running on :9090 (e.g. the monitoring
    stack started by benchmark.sh), reuse it: start()/stop() become no-ops and
    `external=True`. Otherwise spawn a fresh one with the admin API enabled.
    """

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.external: bool = False
        self.log = Path("/tmp/prometheus.log")

    def start(self) -> "Prometheus":
        if self._is_healthy():
            print(f"  [prom] Reusing existing Prometheus on :9090", flush=True)
            self.external = True
            return self

        subprocess.run(["pkill", "-f", "prometheus --config.file"], capture_output=True)
        time.sleep(2)
        if PROM_DATA_DIR.exists():
            shutil.rmtree(PROM_DATA_DIR, ignore_errors=True)
        PROM_DATA_DIR.mkdir(parents=True, exist_ok=True)

        print(f"  [prom] Starting fresh Prometheus  data={PROM_DATA_DIR}  "
              f"log={self.log}", flush=True)
        self.proc = subprocess.Popen(
            [
                "prometheus",
                f"--config.file={PROM_CONFIG}",
                f"--storage.tsdb.path={PROM_DATA_DIR}",
                "--web.enable-admin-api",
                "--web.listen-address=:9090",
            ],
            stdout=open(self.log, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        t0 = time.monotonic()
        while True:
            if self._is_healthy():
                print(f"  [prom] Ready  ({time.monotonic()-t0:.0f}s)  "
                      f"PID={self.proc.pid}", flush=True)
                return self
            if self.proc.poll() is not None:
                raise RuntimeError(f"Prometheus died during startup (see {self.log})")
            if time.monotonic() - t0 > PROM_STARTUP_TIMEOUT_S:
                self.proc.terminate()
                raise RuntimeError(
                    f"Prometheus startup timed out after {PROM_STARTUP_TIMEOUT_S}s"
                )
            time.sleep(1)

    @staticmethod
    def _is_healthy() -> bool:
        try:
            r = requests.get(f"{PROM_URL}/-/ready", timeout=2)
            return r.status_code == 200
        except Exception:
            return False

    def snapshot(self, out_dir: Path) -> str:
        """POST snapshot endpoint; copy hardlinked snapshot dir into out_dir/prom_snapshot."""
        r = requests.post(f"{PROM_URL}/api/v1/admin/tsdb/snapshot", timeout=60)
        r.raise_for_status()
        name = r.json()["data"]["name"]
        src = PROM_DATA_DIR / "snapshots" / name
        dst = out_dir / "prom_snapshot"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return name

    def stop(self) -> None:
        if self.external or self.proc is None:
            return
        print(f"  [prom] Stopping ...", flush=True)
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(15):
            time.sleep(1)
            try:
                os.killpg(os.getpgid(self.proc.pid), 0)
            except ProcessLookupError:
                break
        subprocess.run(["pkill", "-9", "-f", "prometheus --config.file"],
                       capture_output=True)
        self.proc = None


# ─────────────────────────────────────────────────────────────────────────────
class GPUMetricsRecorder:
    """Sweep-wide gpu_metrics_recorder.py on :9092.

    Started once at the top of a sweep, scrapes nvidia-smi every second, and
    publishes Prometheus gauges. If something is already listening on :9092
    (e.g. the monitoring stack already started it), this becomes a no-op.
    """

    PORT = 9092
    SCRIPT = MONITORING_DIR / "gpu_metrics_recorder.py"

    def __init__(self):
        self.proc: subprocess.Popen | None = None

    def start(self) -> "GPUMetricsRecorder":
        if self._is_listening():
            print("  [gpu_exp] Already running on :9092", flush=True)
            return self
        if not self.SCRIPT.exists():
            print(f"  [gpu_exp] WARNING: {self.SCRIPT} not found — "
                  f"GPU metrics will be missing", flush=True)
            return self
        log = Path("/tmp/gpu_metrics_recorder.log")
        self.proc = subprocess.Popen(
            [sys.executable, str(self.SCRIPT),
             "--port", str(self.PORT), "--interval", "1.0"],
            stdout=open(log, "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        for _ in range(10):
            time.sleep(0.5)
            if self._is_listening():
                print(f"  [gpu_exp] Started on :{self.PORT}  "
                      f"PID={self.proc.pid}  log={log}", flush=True)
                return self
        print(f"  [gpu_exp] WARNING: did not become ready (see {log})", flush=True)
        return self

    @classmethod
    def _is_listening(cls) -> bool:
        try:
            r = requests.get(f"http://localhost:{cls.PORT}/metrics", timeout=1)
            return r.status_code == 200
        except Exception:
            return False

    def stop(self) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        self.proc = None


# ─────────────────────────────────────────────────────────────────────────────
class MTier:
    """Static helpers for the MTier KV-offload device."""

    @staticmethod
    def reset() -> None:
        subprocess.run(
            ["sh", "-c", "echo yes | mtier_service reset 2>/dev/null || true"],
            capture_output=True,
        )
