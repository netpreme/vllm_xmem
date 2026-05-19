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


SETUPS = ["hybrid-mtier", "hybrid-cpu"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[16])
    ap.add_argument("--sustained-mins", type=float, default=20.0,
        help="Wall-clock cap per concurrency level in run / save-trace mode")
    ap.add_argument("--duration-cap-mins", type=float, default=20.0,
        help="Hard cap for replay (--from-trace) mode")
    ap.add_argument("--save-trace", default=None,
        help="Output directory for per-session traces (run + capture)")
    ap.add_argument("--from-trace", default=None,
        help="Replay this previously-captured trace directory. OSL is "
             "automatically pinned to each turn's captured value.")
    ap.add_argument("--osl", type=int, default=None,
        help="Override OSL to a fixed value for every turn "
             "(only meaningful with --from-trace).")
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
    if args.osl is not None and not args.from_trace:
        ap.error("--osl requires --from-trace")
    if args.osl is not None:
        os.environ["REPLAY_OSL_OVERRIDE"] = str(args.osl)

    agents = CodingAgents(
        concurrency=args.concurrency,
        setups=SETUPS,
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
        # Replay always pins OSL to the captured value; --osl overrides to a constant.
        agents.benchmark(from_traces=args.from_trace, force_osl=True)
    elif args.save_trace:
        agents.benchmark(capture_to=args.save_trace)
        _emit_per_turn(args.save_trace)
    else:
        agents.benchmark()

    agents.analyze()


def _emit_per_turn(capture_dir: str) -> None:
    """Always-on: write per_turn.csv with ISL/OSL/uncached/timings."""
    extract = SCRIPT_DIR / "extract_per_turn.py"
    if not extract.exists():
        return
    bases = sorted(p for p in Path(capture_dir).iterdir()
                   if p.is_dir() and p.name.startswith("c"))
    targets = bases if bases else [Path(capture_dir)]
    for t in targets:
        if not (t / "capture_meta.json").exists():
            continue
        subprocess.run(
            [sys.executable, str(extract),
             "--capture-dir", str(t),
             "--output", str(t / "per_turn.csv")],
            check=False,
        )


if __name__ == "__main__":
    main()
