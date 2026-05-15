#!/usr/bin/env bash
# Master sequencer: runs all 4 {agent, dataset} combos against the currently-
# loaded vLLM model. Each phase blocks until completion.
#
# Phases:
#   1. claude × SWE-bench Verified
#   2. codex  × SWE-bench Verified
#   3. claude × SWE-bench Pro
#   4. codex  × SWE-bench Pro
#
# Each phase writes to its own runs/ or runs_codex/ directory; results are
# never overwritten. If a phase fails the script keeps going to the next.
#
# vLLM must already be running. To swap models, stop, edit .env, restart, then
# re-run this script — output dirs from prior runs are preserved.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

phase() {
    local name="$1"; shift
    echo
    echo "=========================================="
    echo "[run_all] PHASE: $name  $(date)"
    echo "=========================================="
    "$@" || echo "[run_all] phase '$name' exited non-zero — continuing"
}

cd "$HERE"

# 1. claude × Verified
SWE_DATASET="princeton-nlp/SWE-bench_Verified" SWE_SPLIT="test" \
    phase "claude × Verified" bash run.sh

# 2. codex × Verified
SWE_DATASET="princeton-nlp/SWE-bench_Verified" SWE_SPLIT="test" \
    phase "codex  × Verified" bash run_codex.sh

# 3. claude × Pro
SWE_DATASET="ScaleAI/SWE-bench_Pro" SWE_SPLIT="test" \
    phase "claude × Pro" bash run.sh

# 4. codex × Pro
SWE_DATASET="ScaleAI/SWE-bench_Pro" SWE_SPLIT="test" \
    phase "codex  × Pro" bash run_codex.sh

echo
echo "[run_all] all phases finished $(date)"
