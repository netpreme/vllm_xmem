"""Capture-mode benchmark: drive N concurrent Claude agents against one
or more vLLM backends, snapshot Prometheus per concurrency level, and
optionally record per-session JSONL traces for later replay.

For every concurrency level C:
  1. Start a fresh Prometheus with admin API enabled
  2. Start all vLLM servers in parallel (one per setup/GPU)
  3. Launch C concurrent ``claude -p <swe_bench_problem>`` agents per setup;
     keep N active for ``--sustained-mins`` by replacing each as it finishes
  4. POST /api/v1/admin/tsdb/snapshot, copy into ``<level_dir>/prom_snapshot/``
  5. Stop all vLLMs + Prometheus
  6. Run analyzer over the snapshot
"""
import argparse
import atexit
import concurrent.futures
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from lifecycle import GPUMetricsRecorder, MTier, Prometheus, VLLMServer
from runners.claude_task import CaptureConfig
from runners.dataset import SWEBenchDataset
from runners.setup_pool import SetupPool
from utils.analyzer import ANALYZER_SCRIPT, VENV_PYTHON, run_analyzer
from utils.colors import GREEN, OFF, RED
from utils.config import SETUP_DEFAULTS
from utils.status import LiveStatus, print_averages


# ── paths / defaults ──────────────────────────────────────────────────────────
# capture.py is at .../benchmarks/modes/capture.py
BENCH_ROOT      = Path(__file__).resolve().parent.parent     # .../benchmarks
AGENT_ROOT      = BENCH_ROOT.parent                          # .../coding_agents
ENV_FILE        = AGENT_ROOT / ".env"
RESULTS_DIR     = BENCH_ROOT / "results_benchmarks"

DEFAULT_MODEL           = "qwen/qwen3-coder-30b-a3b-instruct-fp8"
DEFAULT_PORT            = "8000"
DEFAULT_SUSTAINED_MIN_S = 1800     # 30 min


def load_env(path: Path) -> None:
    """Load KEY=VALUE lines from `path` into os.environ (does not overwrite)."""
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


# ── atexit/signal cleanup — tracks the currently-running benchmark ───────────
_active_bench: "CaptureBenchmark | None" = None


def _atexit_cleanup() -> None:
    if _active_bench is not None:
        _active_bench.cleanup_on_exit()


def _signal_handler(_sig, _frame):
    _atexit_cleanup()
    sys.exit(0)


atexit.register(_atexit_cleanup)
signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ─────────────────────────────────────────────────────────────────────────────
class CaptureBenchmark:
    """One full capture-mode run: load dataset, sweep concurrency levels,
    snapshot Prometheus per level. Optionally record JSONL traces.

    Public:
        run()  — top-level entry; returns the run directory
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.duration_s = args.sustained_mins * 60
        self.model = args.model or os.environ.get("MODEL", DEFAULT_MODEL)
        # Mutable run state — populated in run().
        self.run_dir: Path | None = None
        self.dataset: SWEBenchDataset | None = None
        self.setup_specs: list[dict] = []
        self.prom: Prometheus | None = None
        self.gpu_recorder: GPUMetricsRecorder | None = None
        self.capture_config: CaptureConfig | None = None
        self.completed_levels: list[dict] = []
        # Last active port/setup — used only by the cleanup handler.
        self._active_port: int | None = None
        self._active_setup: str | None = None

    # ─── public entry ─────────────────────────────────────────────────────
    def run(self) -> Path:
        global _active_bench
        _active_bench = self
        try:
            self._prepare()
            for concurrency in self.args.concurrency:
                self._run_level(concurrency)
            self._print_summary()
        finally:
            _active_bench = None
        return self.run_dir

    # ─── setup phase ─────────────────────────────────────────────────────
    def _prepare(self) -> None:
        load_env(ENV_FILE)
        self.dataset = SWEBenchDataset.load(
            difficulty=self.args.difficulty,
            no_shuffle=self.args.no_shuffle,
            shuffle_seed=self.args.shuffle_seed,
            start=self.args.start,
            end=self.args.end,
        )
        self.gpu_recorder = GPUMetricsRecorder().start()
        self.setup_specs = self._build_setup_specs()
        self._active_port  = self.setup_specs[0]["port"]  if self.setup_specs else None
        self._active_setup = self.setup_specs[0]["setup"] if self.setup_specs else None

        if any("mtier" in s["setup"] for s in self.setup_specs):
            print(f"\n  [mtier] Resetting MTier memory ...", flush=True)
            MTier.reset()
            time.sleep(2)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = RESULTS_DIR / f"bench_sweep_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._print_banner()

    def _build_setup_specs(self) -> list[dict]:
        specs: list[dict] = []
        for setup in self.args.setup:
            auto_port, auto_gpus = SETUP_DEFAULTS.get(setup, (None, None))
            if self.args.port and len(self.args.setup) == 1:
                port = int(self.args.port)
            else:
                port = int(auto_port or os.environ.get("PORT", DEFAULT_PORT))
            if self.args.gpus and len(self.args.setup) == 1:
                gpus = self.args.gpus
            else:
                gpus = auto_gpus
            specs.append({
                "setup":          setup,
                "port":           port,
                "gpus":           gpus,
                "base_url":       f"http://localhost:{port}",
                "workspace_root": Path(self.args.workspace_root
                                       or f"/tmp/swe_workspaces_p{port}"),
            })
        return specs

    def _print_banner(self) -> None:
        tp_disp       = self.args.tp       if self.args.tp       is not None else "default (1)"
        gpu_util_disp = self.args.gpu_util if self.args.gpu_util is not None else "default (0.9)"
        print(f"\n{'═'*70}")
        print(f"  Setup        : {' | '.join(s['setup'] for s in self.setup_specs)}")
        print(f"  Model        : {self.model}")
        print(f"  TP           : {tp_disp}")
        print(f"  GPU util     : {gpu_util_disp}")
        print(f"  Levels       : {self.args.concurrency}")
        print(f"  Duration/lvl : {self.args.sustained_mins:.1f} min")
        print(f"  Difficulty   : {self.args.difficulty or 'all'}")
        print(f"  Dataset      : SWE-bench Verified  ({len(self.dataset)} tasks)")
        print(f"  Output       : {self.run_dir}/")
        bi = os.environ.get("VLLM_BATCH_INVARIANT", "1")
        print(f"  Determinism  : VLLM_BATCH_INVARIANT={bi}  seed=42  temperature=0")
        if self.args.capture_traces:
            print(f"  Capture      : ON  →  {self.args.capture_traces}/")
            if len(self.setup_specs) != 1:
                print(f"  {RED}WARN{OFF}: --capture-traces with >1 setup is unusual — "
                      f"only one will be replayable")
        print(f"{'═'*70}")

    # ─── per-level phase ─────────────────────────────────────────────────
    def _run_level(self, concurrency: int) -> None:
        level_dir = self.run_dir / f"c{concurrency:03d}"
        level_dir.mkdir(parents=True, exist_ok=True)

        capture_dir = self._init_capture_config_for_level(level_dir)
        self.prom = Prometheus().start()
        vllm_servers = self._start_vllms_parallel(concurrency)
        try:
            level_meta = self._run_pools(level_dir, concurrency)
            print(f"  [prom] Snapshotting TSDB → {level_dir}/prom_snapshot/ ...",
                  flush=True)
            snap_name = self.prom.snapshot(level_dir)
            print(f"  [prom] Snapshot saved (name={snap_name})", flush=True)
            self._write_level_config(level_dir, concurrency, snap_name, level_meta)
            if capture_dir is not None:
                self._write_capture_meta(capture_dir, concurrency, level_meta)
            self.completed_levels.append({
                "concurrency": concurrency,
                "dir":         level_dir,
                "duration_s":  level_meta["duration_s"],
            })
        finally:
            self._stop_vllms_parallel(vllm_servers)
            self.prom.stop()
            self.prom = None
            run_analyzer(level_dir)

    def _init_capture_config_for_level(self, level_dir: Path) -> Path | None:
        """Returns the capture dir if capture is on, otherwise None."""
        if not self.args.capture_traces:
            self.capture_config = None
            return None
        capture_dir = level_dir
        (capture_dir / "traces").mkdir(exist_ok=True)
        sessions_file = capture_dir / "sessions.jsonl"
        sessions_file.write_text("")  # truncate
        self.capture_config = CaptureConfig(
            trace_dir=capture_dir / "traces",
            sessions_file=sessions_file,
            sessions_lock=threading.Lock(),
        )
        return capture_dir

    def _start_vllms_parallel(self, concurrency: int) -> dict[str, VLLMServer]:
        max_seqs = self.args.max_num_seqs if self.args.max_num_seqs is not None else concurrency
        servers: dict[str, VLLMServer] = {}
        try:
            def _start(spec):
                return VLLMServer(spec["setup"], spec["port"],
                                  tp=self.args.tp, gpu_util=self.args.gpu_util,
                                  gpus=spec["gpus"], max_num_seqs=max_seqs).start()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(self.setup_specs)
            ) as ex:
                fmap = {ex.submit(_start, s): s for s in self.setup_specs}
                for f in concurrent.futures.as_completed(fmap):
                    s = fmap[f]
                    servers[s["setup"]] = f.result()
            return servers
        except Exception as e:
            print(f"  {RED}✗ vLLM startup failed:{OFF} {e}", flush=True)
            for v in servers.values():
                try: v.stop()
                except Exception: pass
            self.prom.stop()
            self.prom = None
            raise

    def _stop_vllms_parallel(self, servers: dict[str, VLLMServer]) -> None:
        if not servers:
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(servers)) as ex:
            for s in self.setup_specs:
                v = servers.get(s["setup"])
                if v is not None:
                    ex.submit(v.stop)

    def _run_pools(self, level_dir: Path, concurrency: int) -> dict:
        """Run all setups' Claude pools in parallel for one concurrency level.
        Returns level_meta dict with t_start/end + per-setup counters."""
        labels = " | ".join(s["setup"] for s in self.setup_specs)
        print(f"\n  ── C={concurrency:>3}  parallel: [{labels}]  "
              f"duration={self.duration_s:.0f}s ──", flush=True)

        t_start_unix = time.time()
        t_start_mono = time.monotonic()

        if self.capture_config is not None:
            self.capture_config.t_level_start_mono = t_start_mono
            self.capture_config.t_level_start_unix = t_start_unix

        shared_state = {s["setup"]: {} for s in self.setup_specs}
        status = LiveStatus(self.setup_specs, shared_state, t_start_mono).start()

        results_lock = threading.Lock()
        per_setup: dict[str, dict] = {}

        def _run_pool_for(spec):
            pool = SetupPool(spec, concurrency, self.dataset.instances,
                             self.duration_s, self.model, self.args.no_clone,
                             capture_config=self.capture_config)
            r = pool.run(t_start_mono, shared_state)
            with results_lock:
                per_setup[spec["setup"]] = r

        threads = [threading.Thread(target=_run_pool_for, args=(s,))
                   for s in self.setup_specs]
        for t in threads: t.start()
        for t in threads: t.join()

        status.stop()
        t_end_unix = time.time()
        print("", flush=True)
        print_averages(self.setup_specs, t_start_unix, t_end_unix, per_setup)

        return {
            "t_start_unix": t_start_unix,
            "t_end_unix":   t_end_unix,
            "duration_s":   round(t_end_unix - t_start_unix, 2),
            "setups":       per_setup,
        }

    # ─── per-level outputs ───────────────────────────────────────────────
    def _write_level_config(self, level_dir: Path, concurrency: int,
                            snap_name: str, level_meta: dict) -> None:
        config = {
            "concurrency":    concurrency,
            "model":          self.model,
            "tp":             self.args.tp,
            "gpu_util":       self.args.gpu_util,
            "max_num_seqs":   self.args.max_num_seqs or concurrency,
            "sustained_mins": self.args.sustained_mins,
            "difficulty":     self.args.difficulty,
            "swe_bench_pool_size": len(self.dataset),
            "shuffle_seed":   self.dataset.shuffle_seed,
            "determinism": {
                "VLLM_BATCH_INVARIANT": int(os.environ.get("VLLM_BATCH_INVARIANT", "1") or 0),
                "seed":                  42,
                "temperature":           0,
            },
            "snapshot_name": snap_name,
            "setups": [
                {"setup": s["setup"], "port": s["port"], "gpus": s["gpus"]}
                for s in self.setup_specs
            ],
            **level_meta,
        }
        (level_dir / "config.json").write_text(json.dumps(config, indent=2))
        print(f"  → {level_dir}/  (config.json + prom_snapshot/)", flush=True)

    def _write_capture_meta(self, capture_dir: Path, concurrency: int,
                            level_meta: dict) -> None:
        assert self.capture_config is not None
        try:
            n_sessions = sum(1 for _ in open(self.capture_config.sessions_file))
        except Exception:
            n_sessions = 0
        capture_meta = {
            "kind":            "capture_level",
            "concurrency":     concurrency,
            "model":           self.model,
            "setup":           self.setup_specs[0]["setup"] if self.setup_specs else None,
            "vllm_port":       self.setup_specs[0]["port"]  if self.setup_specs else None,
            "gpus":            self.setup_specs[0]["gpus"]  if self.setup_specs else None,
            "sustained_mins":  self.args.sustained_mins,
            "n_sessions":      n_sessions,
            "t_level_start_unix": self.capture_config.t_level_start_unix,
            "t_level_end_unix":   level_meta["t_end_unix"],
            "duration_s":      level_meta["duration_s"],
            "determinism": {
                "VLLM_BATCH_INVARIANT": int(os.environ.get("VLLM_BATCH_INVARIANT", "1") or 0),
                "seed":                  42,
                "temperature":           0,
            },
            "sweep_run_dir":   str(capture_dir),
        }
        (capture_dir / "capture_meta.json").write_text(
            json.dumps(capture_meta, indent=2))
        print(f"  → capture: {capture_dir}/  "
              f"({n_sessions} sessions, traces/, capture_meta.json)", flush=True)

    # ─── summary ─────────────────────────────────────────────────────────
    def _print_summary(self) -> None:
        print(f"\n{'═'*72}")
        print(f"  Sweep complete  ({len(self.completed_levels)} level(s))")
        print(f"{'═'*72}")
        for L in self.completed_levels:
            fig1 = L["dir"] / "analysis" / "fig1_timeseries.png"
            fig2 = L["dir"] / "analysis" / "fig2_offload_dominant.png"
            ok = fig1.exists() and fig2.exists()
            mark = f"{GREEN}✓{OFF}" if ok else f"{RED}✗{OFF}"
            print(f"  {mark}  C={L['concurrency']:>3}  "
                  f"{L['duration_s']:>5.0f}s  →  {L['dir']}")
        if self.completed_levels:
            print(f"\n  Figures: <level>/analysis/fig1_timeseries.png  +  fig2_offload_dominant.png")
            print(f"  Re-analyze any level:  "
                  f"{VENV_PYTHON} {ANALYZER_SCRIPT} <level_dir>")
        print(f"{'═'*72}\n")

    # ─── cleanup ─────────────────────────────────────────────────────────
    def cleanup_on_exit(self) -> None:
        """Called by atexit/signal handlers. Kill leftover server, reset MTier
        if mtier was used, stop Prometheus + GPU recorder if still running."""
        if self._active_port is not None:
            print(f"\n  [cleanup] Killing vLLM on port {self._active_port} ...",
                  flush=True)
            subprocess.run(["fuser", "-k", f"{self._active_port}/tcp"],
                           capture_output=True)
        if self._active_setup and "mtier" in self._active_setup:
            print("  [cleanup] Resetting MTier ...", flush=True)
            MTier.reset()
        if self.prom is not None:
            print("  [cleanup] Stopping Prometheus ...", flush=True)
            try: self.prom.stop()
            except Exception: pass
        if self.gpu_recorder is not None:
            print("  [cleanup] Stopping GPU exporter ...", flush=True)
            try: self.gpu_recorder.stop()
            except Exception: pass
        print("  [cleanup] Done.", flush=True)


# ─── CLI entry (preserves the old `python -m utils.capture` invocation) ──────
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--concurrency", type=int, nargs="+",
                    default=[1, 10, 12, 14, 16],
                    help="Concurrency levels to sweep")
    ap.add_argument("--setup", nargs="+",
                    default=["hybrid-cpu", "hybrid-mtier"],
                    help="Server setup label(s); multiple values run sequentially")
    ap.add_argument("--sustained-mins", type=float,
                    default=DEFAULT_SUSTAINED_MIN_S / 60,
                    help="Wall-clock cap per level (default: 30 min)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end",   type=int, default=None)
    ap.add_argument("--difficulty", nargs="+", default=None)
    ap.add_argument("--no-shuffle", action="store_true")
    ap.add_argument("--shuffle-seed", type=int, default=None)
    ap.add_argument("--port",    default=None)
    ap.add_argument("--model",   default=None)
    ap.add_argument("--tp",      type=int,   default=None)
    ap.add_argument("--gpu-util",type=float, default=None)
    ap.add_argument("--gpus",    type=str,   default=None)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--workspace-root", type=str, default=None)
    ap.add_argument("--no-clone", action="store_true")
    ap.add_argument("--capture-traces", type=str, default=None)
    return ap.parse_args(argv)


def main() -> None:
    args = _parse_args()
    CaptureBenchmark(args).run()


if __name__ == "__main__":
    main()
