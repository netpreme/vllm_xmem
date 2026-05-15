#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════════
#  Repeat a concurrency sweep N iterations.
#  Each iteration runs `run_sweep.sh` once with the same args, producing a
#  fresh timestamped results dir under benchmarks/results_benchmarks/.
#
#  Usage:
#    bash run_sweep_iter.sh <N_ITERATIONS> -- <run_sweep.sh args ...>
#
#  Example (the one the user requested):
#    bash run_sweep_iter.sh 20 -- --concurrency 1 10 12 13 14 15 16 --sustained-mins 20
# ═══════════════════════════════════════════════════════════════════════════
set -uo pipefail

if [[ $# -lt 3 || "$2" != "--" ]]; then
    echo "Usage: $0 <N_ITERATIONS> -- <run_sweep.sh args ...>"
    echo "Example: $0 20 -- --concurrency 1 10 12 13 14 15 16 --sustained-mins 20"
    exit 2
fi

N_ITER="$1"; shift; shift
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SWEEP="$SCRIPT_DIR/run_sweep.sh"

START_UNIX=$(date +%s)
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  Iterated sweep:  $N_ITER iterations  args: $*"
echo "  Started at:      $(date -u +%FT%TZ)  unix=$START_UNIX"
echo "═══════════════════════════════════════════════════════════════════════════"

for ((i = 1; i <= N_ITER; i++)); do
    echo ""
    echo "▶ Iteration $i / $N_ITER  ($(date -u +%FT%TZ))"
    bash "$SWEEP" "$@"
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "  ✗ iteration $i exited with code $rc — continuing"
    fi
done

END_UNIX=$(date +%s)
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"
echo "  All $N_ITER iterations done."
echo "  Total wall time: $(( (END_UNIX - START_UNIX) / 60 )) min"
echo "═══════════════════════════════════════════════════════════════════════════"
