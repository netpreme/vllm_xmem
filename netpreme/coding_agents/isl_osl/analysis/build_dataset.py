"""Build the canonical analytic dataset for a run.

Joins the two raw capture streams per problem into one structured array:
    per_problem/<iid>.vllm.jsonl   — one row per vLLM completion (watcher)
    per_problem/<iid>.proxy.jsonl  — one row per /v1/messages   (proxy)
    per_problem/<iid>.meta.json    — per-problem metadata

Output is `<run-dir>/data.npz` with one record per turn. The schema is
RAW MEASUREMENTS ONLY; derivations (cache_hit_rate, isl_cached, ttft_ms,
agent main-vs-sub, response category, KV bytes, …) are computed at plot
time via helpers in analysis/dataset.py.

Schema is set by pipeline/metrics_watcher.py (vLLM-side fields) and
pipeline/agent_labeler.py (proxy-side fields).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# `>4 hours` has ~3 problems in Verified — fold into `1+ hours` with `1-4 hours`.
_DIFFICULTY_REMAP = {">4 hours": "1+ hours", "1-4 hours": "1+ hours"}

DTYPE = np.dtype(
    [
        # Identity / orchestration.
        ("instance_id", "U64"),
        ("difficulty", "U24"),
        ("turn", "i4"),
        ("ts", "f8"),
        # vLLM-side raw measurements (metrics_watcher).
        ("isl", "i4"),
        ("osl", "i4"),
        ("isl_new", "i4"),
        ("prefill_ms", "f4"),
        ("decode_ms", "f4"),
        ("queue_ms", "f4"),
        ("e2e_ms", "f4"),
        ("itl_ms", "f4"),  # NaN when osl=0
        ("kv_cache_usage_pct", "f4"),
        ("stop_reason", "U16"),
        # Proxy-side raw measurements (agent_labeler; zero/empty if not captured).
        ("system_prompt_chars", "i4"),
        ("num_tool_defs", "i4"),
        ("num_messages", "i4"),
        ("num_tool_calls", "i4"),
        ("claude_stop_reason", "U16"),
        ("tool_names", "U256"),
        ("has_thinking", "?"),
        ("response_text_chars", "i4"),
    ]
)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _f(v) -> float:
    """Coerce to float; NaN for None/missing."""
    try:
        return float(v) if v is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _i(v) -> int:
    try:
        return int(v) if v is not None else 0
    except (TypeError, ValueError):
        return 0


def build_records(run_dir: Path) -> np.ndarray:
    per_problem_dir = run_dir / "per_problem"
    rows: list[tuple] = []
    for meta_path in sorted(per_problem_dir.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        iid = meta["instance_id"]
        raw_diff = meta.get("difficulty") or ""
        difficulty = _DIFFICULTY_REMAP.get(raw_diff, raw_diff)

        vllm_rows = _load_jsonl(per_problem_dir / f"{iid}.vllm.jsonl")
        proxy_rows = _load_jsonl(per_problem_dir / f"{iid}.proxy.jsonl")

        # At concurrency=1 the two streams should align row-for-row. If
        # they don't, truncate to the shorter and warn — better than
        # silently misaligning turns.
        n = min(len(vllm_rows), len(proxy_rows)) if proxy_rows else len(vllm_rows)
        if proxy_rows and len(vllm_rows) != len(proxy_rows):
            print(
                f"[build_dataset] {iid}: vllm={len(vllm_rows)} "
                f"proxy={len(proxy_rows)} — truncating to {n}"
            )

        for turn, v in enumerate(vllm_rows[:n], start=1):
            p = proxy_rows[turn - 1] if proxy_rows else {}
            rows.append(
                (
                    iid,
                    difficulty,
                    turn,
                    _f(v.get("ts")),
                    _i(v.get("isl")),
                    _i(v.get("osl")),
                    _i(v.get("isl_new")),
                    _f(v.get("prefill_ms")),
                    _f(v.get("decode_ms")),
                    _f(v.get("queue_ms")),
                    _f(v.get("e2e_ms")),
                    _f(v.get("itl_ms")),
                    _f(v.get("kv_cache_usage_pct")),
                    v.get("stop_reason") or "",
                    _i(p.get("system_prompt_chars")),
                    _i(p.get("num_tool_defs")),
                    _i(p.get("num_messages")),
                    _i(p.get("num_tool_calls")),
                    p.get("claude_stop_reason") or "",
                    ",".join(p.get("tool_names") or [])[:256],
                    bool(p.get("has_thinking")),
                    _i(p.get("response_text_chars")),
                )
            )
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
