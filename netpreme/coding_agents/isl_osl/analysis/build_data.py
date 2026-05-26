"""Build the canonical analytic dataset for a run.

Walks runs/<stamp>/per_problem/*.csv + problems.jsonl, joins difficulty
per problem, and writes one structured numpy array to <run-dir>/data.npz.
Every plot script reads from this file rather than re-parsing CSVs.

CSV schema is set by pipeline/metrics_watcher.py (the Prometheus-scraping
process that writes one row per turn).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from data import load_per_problem_rows, load_problem_field, num


# Qwen3-Coder-30B-A3B-Instruct-FP8 with default FP16 KV cache:
#   48 layers × 4 KV heads (GQA) × 128 head_dim × 2 (K+V) × 2 bytes = 96 KB/tok.
# Update if you swap the served model.
KV_BYTES_PER_TOKEN = 48 * 4 * 128 * 2 * 2

DTYPE = np.dtype([
    ("instance_id",             "U64"),
    ("difficulty",               "U24"),
    # Populated by pipeline/agent_labeler.py from the system-prompt size of
    # each /v1/messages request: "main" (claude-cli's outer loop, ~27 k char
    # system prompt) vs "sub" (Task-tool helper, ~3 k char system prompt).
    # If the labeler isn't running the column is always "main".
    ("agent",                   "U4"),
    ("num_tool_defs",           "i4"),
    ("num_messages",            "i4"),
    ("system_prompt_chars",     "i4"),
    ("turn",                    "i4"),
    ("isl",                     "i4"),
    ("osl",                     "i4"),
    ("isl_new",                 "i4"),
    ("isl_cached",              "i4"),
    ("cache_hit_rate",          "f4"),
    # KV-cache prefix accounting (per-problem rolling state).
    # prefix_kv_tokens        : tokens the previous turn left in KV at its
    #                           end = isl[N-1] + osl[N-1]. Max reuse this turn.
    # usable_prefix_kv_tokens : cache_hit_rate × prefix_kv_tokens — what the
    #                           cache actually returned this turn.
    # kv_cache_used_bytes     : isl_cached × KV_BYTES_PER_TOKEN — bytes of
    #                           prefix KV cache hit on this turn.
    ("prefix_kv_tokens",        "i4"),
    ("usable_prefix_kv_tokens", "f4"),
    ("kv_cache_used_bytes",     "i8"),
    # vLLM-internal GPU KV cache utilization at end of turn (0..100%).
    ("kv_cache_usage_pct",      "f4"),
    ("stop_reason",             "U16"),
    ("ttft_ms",                 "i4"),
    ("prefill_ms",              "i4"),
    ("decode_ms",               "i4"),
    ("itl_ms",                  "f4"),
    ("queue_ms",                "i4"),
    ("elapsed_ms",              "i4"),
])


def _int(v: float) -> int:
    """Coerce NaN/inf to 0; keep finite ints intact."""
    return int(v) if math.isfinite(v) else 0


def build_records(run_dir: Path) -> np.ndarray:
    """Walk per-problem CSVs in order; emit one record per turn."""
    difficulty = load_problem_field(run_dir, "difficulty")
    rows: list[tuple] = []
    for iid, turns in load_per_problem_rows(run_dir).items():
        prev_isl_plus_osl = 0
        for i, r in enumerate(turns, start=1):
            isl        = _int(num(r, "isl"))
            osl        = _int(num(r, "osl"))
            isl_cached = _int(num(r, "isl_cached"))
            hit        = num(r, "cache_hit_rate")
            prefix_kv  = prev_isl_plus_osl
            usable_kv  = hit * prefix_kv if math.isfinite(hit) else 0.0
            rows.append((
                iid,
                difficulty.get(iid) or "",
                r.get("agent") or "main",
                _int(num(r, "num_tool_defs")),
                _int(num(r, "num_messages")),
                _int(num(r, "system_prompt_chars")),
                i,
                isl, osl,
                _int(num(r, "isl_new")),
                isl_cached,
                hit,
                prefix_kv,
                usable_kv,
                isl_cached * KV_BYTES_PER_TOKEN,
                num(r, "kv_cache_usage_pct"),
                r.get("stop_reason") or "",
                _int(num(r, "ttft_ms")),
                _int(num(r, "prefill_ms")),
                _int(num(r, "decode_ms")),
                num(r, "itl_ms"),
                _int(num(r, "queue_ms")),
                _int(num(r, "elapsed_ms")),
            ))
            prev_isl_plus_osl = isl + osl
    return np.array(rows, dtype=DTYPE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args()

    turns = build_records(args.run_dir)
    out = args.run_dir / "data.npz"
    np.savez_compressed(out, turns=turns)
    print(f"wrote {out}  ({len(turns):,} turns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
