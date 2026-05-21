"""Benchmark entrypoint. Always runs hybrid-mtier + hybrid-cpu in parallel.

Modes:
  (default)              run with concurrency, no traces
  --save-trace <dir>     run + record per-session traces to <dir>
  --from-trace <dir>     replay <dir> against fresh vLLMs (OSL = captured)
  --from-trace --osl N   replay with every turn's OSL forced to N

Analysis figures + per-turn ISL/OSL/uncached/timings CSV emitted automatically.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BENCH_ROOT = SCRIPT_DIR.parent

if str(BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCH_ROOT))

from utils import CodingAgents


SETUPS_DUAL = ["hybrid-mtier", "hybrid-cpu"]
SETUPS_CAPTURE = ["hybrid-mtier"]   # capture uses one backend (the trace source)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[16])
    ap.add_argument("--sustained-mins", type=float, default=20.0,
        help="Wall-clock cap per concurrency level in run / save-trace mode")
    ap.add_argument("--duration-cap-mins", type=float, default=20.0,
        help="Hard cap for replay (--from-trace) mode")
    ap.add_argument("--save-trace", action="store_true",
        help="Record per-session traces alongside the Prometheus snapshot "
             "(written inside each <run>/c<NN>/ folder).")
    ap.add_argument("--from-trace", default=None,
        help="Replay this previously-captured trace directory. OSL is "
             "automatically pinned to each turn's captured value.")
    ap.add_argument("--osl", type=int, default=None,
        help="OSL target. With --from-trace: overrides each turn's pinned OSL. "
             "With --save-trace AND --isl: target OSL for synthetic capture (default 110).")
    # Synthetic-capture knobs (only meaningful with --save-trace; trigger when --isl is set)
    ap.add_argument("--isl",     type=int, default=None,
        help="Initial ISL target for synthetic capture. Enables synthetic capture mode "
             "when combined with --save-trace (no agents, no GPU needed).")
    ap.add_argument("--isl-new", type=int, default=500,
        help="Target uncached input tokens per turn (synthetic capture only).")
    ap.add_argument("--n-turns", type=int, default=50,
        help="Turns per session (synthetic capture only).")
    ap.add_argument("--n-sessions", type=int, default=30,
        help="Total sessions (synthetic capture only).")
    ap.add_argument("--deterministic", action="store_true",
        help="Pin VLLM_BATCH_INVARIANT=1 + greedy decoding (temp=0, seed=42). "
             "Default: OFF — non-deterministic kernels, model-default sampling.")
    ap.add_argument("--model",          default=None)
    ap.add_argument("--tp",   type=int, default=None)
    ap.add_argument("--gpu-util", type=float, default=None)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--workspace-root", default=None)
    ap.add_argument("--no-clone", action="store_true")
    ap.add_argument("--difficulty", nargs="+", default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end",   type=int, default=None)
    args = ap.parse_args()

    if args.save_trace and args.from_trace:
        ap.error("--save-trace and --from-trace are mutually exclusive")
    # Synthetic mode: --save-trace + --isl. Pure file generation, no GPU/agents.
    if args.save_trace and args.isl is not None:
        from .synthetic_capture import gen_synthetic_capture
        out_dir = gen_synthetic_capture(
            isl=args.isl, isl_new=args.isl_new,
            osl=(args.osl if args.osl is not None else 110),
            n_turns=args.n_turns, n_sessions=args.n_sessions,
            seed=42)
        print(f"\nSynthetic capture written → {out_dir}")
        print(f"Replay with: bash bench.sh --from-trace {out_dir} --concurrency <C> --deterministic")
        return
    if args.osl is not None and not args.from_trace:
        ap.error("--osl requires --from-trace (or --save-trace with --isl)")
    if args.osl is not None and args.from_trace:
        os.environ["REPLAY_OSL_OVERRIDE"] = str(args.osl)

    # Determinism toggle (default OFF). Controls both VLLM_BATCH_INVARIANT
    # and the temperature/seed override forwarded by start_server.sh.
    if args.deterministic:
        os.environ["VLLM_BATCH_INVARIANT"] = "1"
        os.environ["OVERRIDE_GEN_CONFIG"] = '{"temperature":0,"seed":42}'
    else:
        os.environ["VLLM_BATCH_INVARIANT"] = "0"
        os.environ["OVERRIDE_GEN_CONFIG"] = ""

    # In capture mode (--save-trace) use only mtier — the trace is the source-of-truth
    # workload that the from-trace replay will then drive against both backends.
    setups = SETUPS_CAPTURE if args.save_trace else SETUPS_DUAL

    agents = CodingAgents(
        concurrency=args.concurrency,
        setups=setups,
        sustained_mins=args.sustained_mins,
        duration_cap_mins=args.duration_cap_mins,
        model=args.model,
        tp=args.tp,
        gpu_util=args.gpu_util,
        max_num_seqs=args.max_num_seqs,
        difficulty=args.difficulty,
        swe_range=(args.start, args.end),
        workspace_root=args.workspace_root,
        no_clone=args.no_clone,
    )

    if args.from_trace:
        # OSL pinning is tied to --deterministic: ON → pin to captured; OFF → model decides.
        agents.benchmark(from_traces=args.from_trace, force_osl=args.deterministic)
    elif args.save_trace:
        # Pass a sentinel — the bench writes traces inside the run dir.
        agents.benchmark(capture_to="enabled")
        # Auto-extract per_turn.csv inside each level dir produced
        for level_dir in agents.last_level_dirs:
            _emit_per_turn(level_dir)
    else:
        agents.benchmark()

    agents.analyze()


def _emit_per_turn(level_dir: Path) -> None:
    """Write per_turn.csv with ISL/OSL/uncached/timings inside the level dir."""
    extract = SCRIPT_DIR / "extract_per_turn.py"
    if not extract.exists() or not (Path(level_dir) / "capture_meta.json").exists():
        return
    subprocess.run(
        [sys.executable, str(extract),
         "--capture-dir", str(level_dir),
         "--output", str(Path(level_dir) / "per_turn.csv")],
        check=False,
    )


if __name__ == "__main__":
    main()
