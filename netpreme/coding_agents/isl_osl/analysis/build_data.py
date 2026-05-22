"""Build the canonical analytic dataset for a run.

Walks runs/<stamp>/per_problem/*.csv + problems.jsonl, joins difficulty
per problem, and writes one structured numpy array to <run-dir>/data.npz.
Every plot script reads from this file rather than re-parsing CSVs.

  import numpy as np
  t = np.load("runs/<stamp>/data.npz")["turns"]
  t.dtype.names    # see schema
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

import json

from data import load_per_problem_rows, load_problem_field, num


# Sub-agent calls (claude's Task tool spawning a focused helper) ship a much
# smaller system prompt than the main agent. The main agent's system prompt
# is ~27k characters; sub-agents are ~3k. We classify each turn by reading
# the matching transcript line.
MAIN_SYSTEM_CHARS_MIN = 10_000


def classify_agent(run_dir: Path) -> dict[str, list[str]]:
    """Return {instance_id: [agent_kind_per_turn]} — one entry per /v1/messages
    call (matches the per_problem CSV row order, including empty turns).

    agent_kind is "main" (full agent loop) or "sub" (Task-tool sub-agent),
    derived from the size of the system prompt in the transcript.
    """
    out: dict[str, list[str]] = {}
    tx_dir = run_dir / "transcripts"
    if not tx_dir.exists():
        return out
    for tx in sorted(tx_dir.glob("*.jsonl")):
        kinds: list[str] = []
        for line in tx.open():
            ev = json.loads(line)
            sb = ev["request"].get("system")
            if isinstance(sb, list):
                sys_chars = sum(len(s.get("text", "")) for s in sb)
            else:
                sys_chars = len(sb or "")
            kinds.append("main" if sys_chars >= MAIN_SYSTEM_CHARS_MIN else "sub")
        out[tx.stem] = kinds
    return out

# Per-token KV-cache bytes for the served model.
# Qwen3-Coder-30B-A3B-Instruct-FP8 served on vLLM with default FP16 KV cache:
#   48 layers × 4 KV heads (GQA) × 128 head_dim × 2 (K+V) × 2 bytes (FP16)
# = 98_304 bytes/token (96 KB).
# Update if you swap the served model.
KV_BYTES_PER_TOKEN = 48 * 4 * 128 * 2 * 2

DTYPE = np.dtype([
    ("instance_id",             "U64"),
    ("difficulty",              "U24"),
    # "main" = claude's outer coding agent loop; "sub" = a Task-tool helper
    # spawned by the main agent (own conversation, smaller system prompt).
    ("agent",                   "U4"),
    ("turn",                    "i4"),
    ("isl",                     "i4"),
    ("osl",                     "i4"),
    ("isl_new",                 "i4"),
    ("isl_cached",              "i4"),
    ("cache_hit_rate",          "f4"),
    # KV-cache prefix accounting (per-problem rolling state).
    # prefix_kv_tokens        : tokens the previous turn left in KV at its
    #                           end = isl[N-1] + osl[N-1]. The maximum
    #                           the cache could reuse this turn.
    # usable_prefix_kv_tokens : cache_hit_rate * prefix_kv_tokens — what
    #                           the cache actually returned this turn.
    # kv_cache_used_bytes     : isl_cached * KV_BYTES_PER_TOKEN — bytes
    #                           of prefix KV cache hit on this turn.
    ("prefix_kv_tokens",        "i4"),
    ("usable_prefix_kv_tokens", "f4"),
    ("kv_cache_used_bytes",     "i8"),
    ("category",                "U10"),
    ("num_tool_calls",          "i4"),
    ("ttft_ms",                 "i4"),
    ("decode_ms",               "i4"),
    ("itl_ms",                  "f4"),
    ("elapsed_ms",              "i4"),
])


def _int(v: float) -> int:
    """Coerce NaN/inf to 0; keep finite ints intact."""
    return int(v) if math.isfinite(v) else 0


def build_records(run_dir: Path) -> np.ndarray:
    """Walk per-problem CSVs in order; emit one structured record per turn.
    Track per-problem rolling state to fill `prefix_kv_tokens` (= prev
    isl+osl) and `usable_prefix_kv_tokens` (= cache_hit_rate × prev)."""
    difficulty = load_problem_field(run_dir, "difficulty")
    agents     = classify_agent(run_dir)
    rows: list[tuple] = []
    for iid, turns in load_per_problem_rows(run_dir).items():
        agent_kinds = agents.get(iid, ["main"] * len(turns))
        prev_isl_plus_osl = 0
        for i, r in enumerate(turns, start=1):
            isl        = _int(num(r, "isl"))
            osl        = _int(num(r, "osl"))
            isl_cached = _int(num(r, "isl_cached"))
            hit        = num(r, "cache_hit_rate")
            prefix_kv  = prev_isl_plus_osl
            usable_kv  = hit * prefix_kv if math.isfinite(hit) else 0.0
            agent = agent_kinds[i - 1] if i - 1 < len(agent_kinds) else "main"
            rows.append((
                iid,
                difficulty.get(iid) or "",
                agent,
                i,
                isl, osl,
                _int(num(r, "isl_new")),
                isl_cached,
                hit,
                prefix_kv,
                usable_kv,
                isl_cached * KV_BYTES_PER_TOKEN,
                r.get("category") or "",
                _int(num(r, "num_tool_calls")),
                _int(num(r, "ttft_ms")),
                _int(num(r, "decode_ms")),
                num(r, "itl_ms"),
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
