#!/usr/bin/env python3
"""Replay a captured /v1/messages trace against both backends in parallel.

Usage:
    # Capture once (single setup, single concurrency level):
    python3 bench_concurrent_users.py \\
        --setup hybrid-mtier --concurrency 16 --sustained-mins 20 \\
        --capture-traces /path/to/capture_c016/

    # Replay against mtier (GPU0:8001) and cpu (GPU1:8002) in parallel:
    python3 bench_replay.py \\
        --capture-dir /path/to/capture_c016/ \\
        --setup hybrid-mtier hybrid-cpu

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
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from . import bench_concurrent_users as bcu
from .from_trace_session import (
    load_session_meta,
    load_session_trace,
    replay_one_session,
)

import aiohttp
import requests


SCRIPT_DIR  = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "results_benchmarks"


async def _replay_all_for_backend(
    capture_dir: Path,
    base_url: str,
    setup_label: str,
    shared_state: dict,
    duration_cap_s: float | None,
) -> dict:
    """Replay every captured session against one backend. Returns summary stats."""
    meta, session_records = load_session_meta(capture_dir)
    # Pre-load all sessions' traces.
    sessions = [load_session_trace(capture_dir, r) for r in session_records]
    sessions = [s for s in sessions if s.turns]

    timeout = aiohttp.ClientTimeout(total=None, sock_read=None, sock_connect=15)
    connector = aiohttp.TCPConnector(limit=0, force_close=False)
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as hs:
        t0 = asyncio.get_event_loop().time()
        tasks = []
        for s in sessions:
            tasks.append(asyncio.create_task(
                replay_one_session(s, base_url, t0, hs, label=setup_label)
            ))

        # Update shared_state periodically until tasks done.
        async def _poke():
            while True:
                done = sum(1 for t in tasks if t.done())
                active = len(tasks) - done
                shared_state[setup_label] = {
                    "n_done":   done,
                    "n_active": active,
                    "n_total":  len(tasks),
                }
                if done == len(tasks):
                    return
                await asyncio.sleep(0.5)

        poke = asyncio.create_task(_poke())

        if duration_cap_s is not None:
            # Cap total replay duration. Cancel any laggards past the cap.
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=duration_cap_s,
                )
            except asyncio.TimeoutError:
                for t in tasks:
                    if not t.done():
                        t.cancel()
                # Wait for cancellations to settle.
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
            "setup":            setup_label,
            "base_url":         base_url,
            "n_sessions":       len(sessions),
            "n_completed":      len(results),
            "n_turns":          sum(r.n_turns for r in results),
            "n_ok":             sum(r.n_ok    for r in results),
            "n_error":          sum(r.n_error for r in results),
            "bytes_received":   sum(r.bytes_received for r in results),
        }


def _run_replay_for_backend_blocking(*args, **kwargs) -> dict:
    """Thread entry point — runs an asyncio replay loop in this thread."""
    return asyncio.run(_replay_all_for_backend(*args, **kwargs))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--capture-dir", required=True, type=str,
                    help="Directory produced by bench_concurrent_users.py --capture-traces")
    ap.add_argument("--setup", nargs="+", default=["hybrid-mtier", "hybrid-cpu"],
                    help="Setups to replay against, in parallel. Default: both")
    ap.add_argument("--duration-cap-mins", type=float, default=None,
                    help="Hard cap on replay duration. Default: 1.5x capture duration")
    ap.add_argument("--concurrency", type=int, default=None,
                    help="Sets vLLM --max-num-seqs. Default: capture's concurrency")
    ap.add_argument("--max-num-seqs", type=int, default=None,
                    help="Override --max-num-seqs explicitly")
    ap.add_argument("--tp", type=int, default=None)
    ap.add_argument("--gpu-util", type=float, default=None)
    ap.add_argument("--gpus", type=str, default=None,
                    help="Override CUDA_VISIBLE_DEVICES for the single-setup case")
    ap.add_argument("--port", default=None,
                    help="Override port for the single-setup case")
    args = ap.parse_args()

    capture_dir = Path(args.capture_dir).expanduser().resolve()
    if not (capture_dir / "capture_meta.json").exists():
        print(f"ERROR: {capture_dir}/capture_meta.json not found", file=sys.stderr)
        sys.exit(2)

    capture_meta = json.loads((capture_dir / "capture_meta.json").read_text())
    n_sessions = capture_meta.get("n_sessions", 0)
    concurrency = args.concurrency or capture_meta.get("concurrency") or 16
    duration_cap_s = (args.duration_cap_mins or 20.0) * 60
    model = capture_meta.get("model", bcu.DEFAULT_MODEL)

    bcu.load_env(bcu.ENV_FILE)

    # Per-setup defaults (mtier → 8001/GPU0, cpu → 8002/GPU1). Mirrors the
    # mapping in bench_concurrent_users.main().
    _SETUP_DEFAULTS = {
        "hybrid-mtier": (8001, "0"),
        "mtier-only":   (8001, "0"),
        "hybrid-cpu":   (8002, "1"),
        "cpu-only":     (8002, "1"),
    }

    # Build per-setup specs.
    setup_specs: list[dict] = []
    for setup in args.setup:
        auto_port, auto_gpus = _SETUP_DEFAULTS.get(setup, (None, None))
        if args.port and len(args.setup) == 1:
            port = int(args.port)
        else:
            port = int(auto_port or os.environ.get("PORT", bcu.DEFAULT_PORT))
        if args.gpus and len(args.setup) == 1:
            gpus_for_setup = args.gpus
        else:
            gpus_for_setup = auto_gpus
        setup_specs.append({
            "setup":    setup,
            "port":     port,
            "gpus":     gpus_for_setup,
            "base_url": f"http://localhost:{port}",
        })

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / f"from_trace_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)
    level_dir = run_dir / f"c{concurrency:03d}"
    level_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═'*70}")
    print(f"  Replay run   : {capture_dir}")
    print(f"  Sessions     : {n_sessions}  (from capture)")
    print(f"  Setups       : {' | '.join(s['setup'] for s in setup_specs)}")
    print(f"  Model        : {model}")
    print(f"  Concurrency  : {concurrency}  (→ max_num_seqs)")
    print(f"  Duration cap : {duration_cap_s/60:.1f} min")
    print(f"  Capture sustained: {capture_meta.get('sustained_mins', '?')} min")
    _bi = os.environ.get("VLLM_BATCH_INVARIANT", "1")
    print(f"  Determinism  : VLLM_BATCH_INVARIANT={_bi}  seed=42  temperature=0")
    print(f"  Output       : {run_dir}/")
    print(f"{'═'*70}\n")

    # ── Start exporters + Prometheus + both vLLMs (mirrors bench_concurrent_users) ──
    bcu._gpu_exp_proc = bcu.ensure_gpu_exporter()
    bcu._prom_proc    = bcu.start_prometheus()
    if any("mtier" in s["setup"] for s in setup_specs):
        print(f"  [mtier] Resetting MTier memory ...", flush=True)
        subprocess.run(["sh", "-c", "echo yes | mtier_service reset 2>/dev/null || true"],
                       capture_output=True)
        time.sleep(2)

    max_seqs = args.max_num_seqs if args.max_num_seqs is not None else concurrency
    vllm_procs: dict[str, subprocess.Popen] = {}
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(setup_specs)) as ex:
            fmap = {
                ex.submit(bcu.start_vllm, s["setup"], s["port"],
                          tp=args.tp, gpu_util=args.gpu_util,
                          gpus=s["gpus"], max_num_seqs=max_seqs): s
                for s in setup_specs
            }
            for f in concurrent.futures.as_completed(fmap):
                s = fmap[f]
                vllm_procs[s["setup"]] = f.result()
    except Exception as e:
        print(f"  vLLM startup failed: {e}", flush=True)
        for p in vllm_procs.values():
            try: p.terminate()
            except Exception: pass
        bcu.stop_prometheus(bcu._prom_proc)
        raise

    per_setup: dict[str, dict] = {}
    shared_state: dict[str, dict] = {s["setup"]: {} for s in setup_specs}

    # Live status thread (reuses bench_concurrent_users layout).
    t_start_unix = time.time()
    t_start_mono = time.monotonic()
    status_stop = threading.Event()
    threading.Thread(
        target=bcu._live_status_thread,
        args=(setup_specs, shared_state, t_start_mono, status_stop),
        daemon=True,
    ).start()

    # Launch one replay thread per setup; each thread runs its own asyncio loop.
    try:
        results_lock = threading.Lock()

        def _run_for_setup(spec):
            try:
                r = _run_replay_for_backend_blocking(
                    capture_dir, spec["base_url"], spec["setup"],
                    shared_state, duration_cap_s,
                )
            except Exception as e:
                r = {"setup": spec["setup"], "error": f"{type(e).__name__}: {e}"}
            with results_lock:
                per_setup[spec["setup"]] = r

        threads = [threading.Thread(target=_run_for_setup, args=(s,)) for s in setup_specs]
        for t in threads: t.start()
        for t in threads: t.join()
    finally:
        status_stop.set()
        time.sleep(0.5)

    t_end_unix = time.time()
    print("", flush=True)
    bcu._print_averages(setup_specs, t_start_unix, t_end_unix, {
        s["setup"]: {"n_tasks_completed": per_setup.get(s["setup"], {}).get("n_completed", 0)}
        for s in setup_specs
    })

    # Snapshot Prometheus.
    print(f"\n  [prom] Snapshotting TSDB → {level_dir}/prom_snapshot/ ...", flush=True)
    snap_name = bcu.take_snapshot(level_dir)
    print(f"  [prom] Snapshot saved (name={snap_name})", flush=True)

    # config.json (compatible with the analyzer in bench_concurrent_users).
    cfg = {
        "concurrency":    concurrency,
        "model":          model,
        "tp":             args.tp,
        "gpu_util":       args.gpu_util,
        "max_num_seqs":   max_seqs,
        "sustained_mins": duration_cap_s / 60.0,
        "swe_bench_pool_size": n_sessions,
        "determinism": {
            "VLLM_BATCH_INVARIANT": int(os.environ.get("VLLM_BATCH_INVARIANT", "1") or 0),
            "seed":                  42,
            "temperature":           0,
        },
        "snapshot_name": snap_name,
        "kind":          "replay",
        "capture_dir":   str(capture_dir),
        "capture_meta":  capture_meta,
        "setups": [
            {"setup": s["setup"], "port": s["port"], "gpus": s["gpus"]}
            for s in setup_specs
        ],
        "per_setup":     per_setup,
        "t_start_unix":  t_start_unix,
        "t_end_unix":    t_end_unix,
        "duration_s":    round(t_end_unix - t_start_unix, 2),
    }
    (level_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"  → {level_dir}/  (config.json + prom_snapshot/)", flush=True)

    # Stop vLLMs + Prometheus.
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(vllm_procs)) as ex:
        for s in setup_specs:
            proc = vllm_procs.get(s["setup"])
            if proc is not None:
                ex.submit(bcu.stop_vllm, proc, s["setup"], s["port"])

    bcu.stop_prometheus(bcu._prom_proc)
    bcu._prom_proc = None

    bcu.run_analyzer(level_dir)
    print(f"\n  Replay complete → {run_dir}/", flush=True)


if __name__ == "__main__":
    main()
