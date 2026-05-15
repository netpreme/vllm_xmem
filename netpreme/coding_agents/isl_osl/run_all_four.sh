#!/usr/bin/env bash
# Runs all four combos sequentially:
#   1. claude × SWE-bench Verified
#   2. claude × SWE-bench Pro
#   3. codex  × SWE-bench Verified
#   4. codex  × SWE-bench Pro
# vLLM is cold-restarted before every problem (run.sh / run_codex.sh handle it).
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

stamp() { date -u +%FT%TZ; }

echo "[wrapper] $(stamp) START — claude × Verified"
SWE_DATASET=princeton-nlp/SWE-bench_Verified bash run.sh
echo "[wrapper] $(stamp) DONE  — claude × Verified"

echo "[wrapper] $(stamp) START — claude × Pro"
SWE_DATASET=ScaleAI/SWE-bench_Pro bash run.sh
echo "[wrapper] $(stamp) DONE  — claude × Pro"

echo "[wrapper] $(stamp) START — codex × Verified"
SWE_DATASET=princeton-nlp/SWE-bench_Verified bash run_codex.sh
echo "[wrapper] $(stamp) DONE  — codex × Verified"

echo "[wrapper] $(stamp) START — codex × Pro"
SWE_DATASET=ScaleAI/SWE-bench_Pro bash run_codex.sh
echo "[wrapper] $(stamp) DONE  — codex × Pro"

echo "[wrapper] $(stamp) ALL DONE"
