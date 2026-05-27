#!/usr/bin/env python3
"""Replay-mode benchmark: replay a captured /v1/messages trace against both
backends in parallel.

What this guarantees:
  - Both backends receive byte-identical request bodies in byte-identical
    order at byte-identical absolute times (relative to the run start).
  - With VLLM_BATCH_INVARIANT=1 + temperature=0 + seed=42, both also generate
    bit-identical outputs, so per-turn KV block hashes match across backends.
  - Server-side prefix cache lookups, allocations, evictions, and offload
    traffic are therefore deterministic and identical between mtier and cpu.

What this does NOT do:
  - Execute tool calls or modify any workspace. The agent loop is replaced
    by raw HTTP replay — the response is drained and discarded.
  - Adjust for slow backends. Requests fire on schedule regardless of
    whether the previous response arrived. This is open-loop replay.
"""
import argparse
import asyncio
import concurrent.futures
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import aiohttp

from lifecycle import GPUMetricsRecorder, MTier, Prometheus, VLLMServer
from modes.capture import DEFAULT_MODEL, DEFAULT_PORT, ENV_FILE, RESULTS_DIR, load_env
from modes.replay_session import (
    load_session_meta,
    load_session_trace,
    replay_one_session,
)
from utils.analyzer import run_analyzer
from utils.config import SETUP_DEFAULTS
from utils.status import LiveStatus, print_averages


# ── async replay engine (one backend) ─────────────────────────────────────────
async def _replay_all_for_backend(
    capture_dir: Path,
    base_url: str,
    setup_label: str,
    shared_state: dict,
    duration_cap_s: float | None,
) -> dict:
    """Replay every captured session against one backend. Returns summary stats."""
    _meta, session_records = load_session_meta(capture_dir)
    sessions = [load_session_trace(capture_dir, r) for r in session_records]
    sessions = [s for s in sessions if s.turns]

    timeout = aiohttp.ClientTimeout(total=None, sock_read=None, sock_connect=15)
    connector = aiohttp.TCPConnector(limit=0, force_close=False)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as hs:
        t0 = asyncio.get_event_loop().time()
        tasks = [
            asyncio.create_task(replay_one_session(s, base_url, t0, hs,
                                                   label=setup_label))
            for s in sessions
        ]

        async def _poke():
            while True:
                done = sum(1 for t in tasks if t.done())
                shared_state[setup_label] = {
                    "n_done":   done,
                    "n_active": len(tasks) - done,
                    "n_total":  len(tasks),
                }
                if done == len(tasks):
                    return
                await asyncio.sleep(0.5)

        poke = asyncio.create_task(_poke())

        if duration_cap_s is not None:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=duration_cap_s,
                )
            except asyncio.TimeoutError:
                for t in tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        else:
            await asyncio.gather(*tasks, return_exceptions=True)

        poke.cancel()
        try:
            await poke
        except (asyncio.CancelledError, Exception):
            pass

        results = []
        for t in tasks:
            if t.cancelled():
                continue
            try:
                results.append(t.result())
            except Exception:
                continue

        return {
            "setup":          setup_label,
            "base_url":       base_url,
            "n_sessions":     len(sessions),
            "n_completed":    len(results),
            "n_turns":        sum(r.n_turns for r in results),
            "n_ok":           sum(r.n_ok    for r in results),
            "n_error":        sum(r.n_error for r in results),
            "bytes_received": sum(r.bytes_received for r in results),
        }


def _run_replay_for_backend_blocking(*args, **kwargs) -> dict:
    """Thread entry — runs an asyncio replay loop in this thread."""
    return asyncio.run(_replay_all_for_backend(*args, **kwargs))


# ─────────────────────────────────────────────────────────────────────────────
class ReplayBenchmark:
    """Replay one captured trace dir against N backends in parallel.

    Public:
        run() — top-level entry; returns the run directory.
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.capture_dir = Path(args.capture_dir).expanduser().resolve()
        if not (self.capture_dir / "capture_meta.json").exists():
            raise FileNotFoundError(
                f"{self.capture_dir}/capture_meta.json not found"
            )
        self.capture_meta = json.loads(
            (self.capture_dir / "capture_meta.json").read_text()
        )
        self.n_sessions  = self.capture_meta.get("n_sessions", 0)
        self.concurrency = args.concurrency or self.capture_meta.get("concurrency") or 16
        self.duration_cap_s = (args.duration_cap_mins or 20.0) * 60
        self.model = self.capture_meta.get("model", DEFAULT_MODEL)

        # Mutable run state — populated in run().
        self.setup_specs: list[dict] = []
        self.prom: Prometheus | None = None
        self.gpu_recorder: GPUMetricsRecorder | None = None
        self.run_dir: Path | None = None
        self.level_dir: Path | None = None

    def run(self) -> Path:
        load_env(ENV_FILE)
        self.setup_specs = self._build_setup_specs()
        self._setup_run_dir()
        self._print_banner()

        self.gpu_recorder = GPUMetricsRecorder().start()
        self.prom         = Prometheus().start()
        if any("mtier" in s["setup"] for s in self.setup_specs):
            print(f"  [mtier] Resetting MTier memory ...", flush=True)
            MTier.reset()
            time.sleep(2)

        vllm_servers = self._start_vllms_parallel()
        try:
            per_setup, t_start_unix, t_end_unix = self._run_replay_pools()
            print(f"\n  [prom] Snapshotting TSDB → "
                  f"{self.level_dir}/prom_snapshot/ ...", flush=True)
            snap_name = self.prom.snapshot(self.level_dir)
            print(f"  [prom] Snapshot saved (name={snap_name})", flush=True)
            self._write_config(per_setup, t_start_unix, t_end_unix, snap_name)
        finally:
            self._stop_vllms_parallel(vllm_servers)
            self.prom.stop()
            self.prom = None

        run_analyzer(self.level_dir)
        print(f"\n  Replay complete → {self.run_dir}/", flush=True)
        return self.run_dir

    # ─── setup ───────────────────────────────────────────────────────────
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
                "setup":    setup,
                "port":     port,
                "gpus":     gpus,
                "base_url": f"http://localhost:{port}",
            })
        return specs

    def _setup_run_dir(self) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = RESULTS_DIR / f"from_trace_{ts}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.level_dir = self.run_dir / f"c{self.concurrency:03d}"
        self.level_dir.mkdir(parents=True, exist_ok=True)

    def _print_banner(self) -> None:
        bi = os.environ.get("VLLM_BATCH_INVARIANT", "1")
        print(f"\n{'═'*70}")
        print(f"  Replay run   : {self.capture_dir}")
        print(f"  Sessions     : {self.n_sessions}  (from capture)")
        print(f"  Setups       : {' | '.join(s['setup'] for s in self.setup_specs)}")
        print(f"  Model        : {self.model}")
        print(f"  Concurrency  : {self.concurrency}  (→ max_num_seqs)")
        print(f"  Duration cap : {self.duration_cap_s/60:.1f} min")
        print(f"  Capture sustained: {self.capture_meta.get('sustained_mins', '?')} min")
        print(f"  Determinism  : VLLM_BATCH_INVARIANT={bi}  seed=42  temperature=0")
        print(f"  Output       : {self.run_dir}/")
        print(f"{'═'*70}\n")

    def _start_vllms_parallel(self) -> dict[str, VLLMServer]:
        max_seqs = (self.args.max_num_seqs
                    if self.args.max_num_seqs is not None
                    else self.concurrency)
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
            print(f"  vLLM startup failed: {e}", flush=True)
            for v in servers.values():
                try: v.stop()
                except Exception: pass
            self.prom.stop()
            raise

    def _stop_vllms_parallel(self, servers: dict[str, VLLMServer]) -> None:
        if not servers:
            return
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(servers)) as ex:
            for s in self.setup_specs:
                v = servers.get(s["setup"])
                if v is not None:
                    ex.submit(v.stop)

    # ─── workload ────────────────────────────────────────────────────────
    def _run_replay_pools(self) -> tuple[dict[str, dict], float, float]:
        per_setup: dict[str, dict] = {}
        shared_state = {s["setup"]: {} for s in self.setup_specs}

        t_start_unix = time.time()
        t_start_mono = time.monotonic()
        status = LiveStatus(self.setup_specs, shared_state, t_start_mono).start()

        try:
            results_lock = threading.Lock()

            def _run_for_setup(spec):
                try:
                    r = _run_replay_for_backend_blocking(
                        self.capture_dir, spec["base_url"], spec["setup"],
                        shared_state, self.duration_cap_s,
                    )
                except Exception as e:
                    r = {"setup": spec["setup"], "error": f"{type(e).__name__}: {e}"}
                with results_lock:
                    per_setup[spec["setup"]] = r

            threads = [threading.Thread(target=_run_for_setup, args=(s,))
                       for s in self.setup_specs]
            for t in threads: t.start()
            for t in threads: t.join()
        finally:
            status.stop()

        t_end_unix = time.time()
        print("", flush=True)
        print_averages(self.setup_specs, t_start_unix, t_end_unix, {
            s["setup"]: {"n_tasks_completed": per_setup.get(s["setup"], {}).get("n_completed", 0)}
            for s in self.setup_specs
        })
        return per_setup, t_start_unix, t_end_unix

    # ─── output ──────────────────────────────────────────────────────────
    def _write_config(self, per_setup: dict, t_start_unix: float,
                      t_end_unix: float, snap_name: str) -> None:
        max_seqs = (self.args.max_num_seqs
                    if self.args.max_num_seqs is not None
                    else self.concurrency)
        cfg = {
            "concurrency":    self.concurrency,
            "model":          self.model,
            "tp":             self.args.tp,
            "gpu_util":       self.args.gpu_util,
            "max_num_seqs":   max_seqs,
            "sustained_mins": self.duration_cap_s / 60.0,
            "swe_bench_pool_size": self.n_sessions,
            "determinism": {
                "VLLM_BATCH_INVARIANT": int(os.environ.get("VLLM_BATCH_INVARIANT", "1") or 0),
                "seed":                  42,
                "temperature":           0,
            },
            "snapshot_name": snap_name,
            "kind":          "replay",
            "capture_dir":   str(self.capture_dir),
            "capture_meta":  self.capture_meta,
            "setups": [
                {"setup": s["setup"], "port": s["port"], "gpus": s["gpus"]}
                for s in self.setup_specs
            ],
            "per_setup":     per_setup,
            "t_start_unix":  t_start_unix,
            "t_end_unix":    t_end_unix,
            "duration_s":    round(t_end_unix - t_start_unix, 2),
        }
        (self.level_dir / "config.json").write_text(json.dumps(cfg, indent=2))
        print(f"  → {self.level_dir}/  (config.json + prom_snapshot/)", flush=True)


# ─── CLI entry (also exposed as `python -m utils.replay`) ────────────────────
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--capture-dir", required=True, type=str)
    ap.add_argument("--setup", nargs="+", default=["hybrid-mtier", "hybrid-cpu"])
    ap.add_argument("--duration-cap-mins", type=float, default=None)
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--tp", type=int, default=None)
    ap.add_argument("--gpu-util", type=float, default=None)
    ap.add_argument("--gpus", type=str, default=None)
    ap.add_argument("--port", default=None)
    return ap.parse_args(argv)


def main() -> None:
    args = _parse_args()
    try:
        ReplayBenchmark(args).run()
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
