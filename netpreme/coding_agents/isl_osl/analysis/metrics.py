"""The analytic dataset: build it, load it, derive from it.

The canonical per-turn dataset (`<save-dir>/data.npz`) stores RAW
MEASUREMENTS ONLY. This module owns its whole lifecycle:

  * `build_records` / `main` — join the raw per-problem capture streams
    (`*.vllm.jsonl` + `*.proxy.jsonl` + `*.meta.json`) into data.npz.
    Run as a script:  `python metrics.py --save-dir <save-dir>`.
  * `load_data` — read the structured array back.
  * Derivation helpers — compute non-stored fields (`isl_cached`,
    `cache_hit_rate`, per-tier hit rates, `ttft_ms`, `agent`) on the fly.
    Use these from plot scripts instead of expecting the field on the array.

Raw schema is set by pipeline/metrics_watcher.py (vLLM-side fields) and
pipeline/proxy.py (proxy-side fields).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# SWE-bench Verified difficulty buckets. The ">4 hours" bucket has only ~3
# problems in the 500-problem split, too few to be meaningful, so we fold
# it into "1+ hours" together with "1-4 hours".
VERIFIED_BUCKETS = ["<15 min fix", "15 min - 1 hour", "1+ hours"]
DIFFICULTY_REMAP = {">4 hours": "1+ hours", "1-4 hours": "1+ hours"}

# main = claude's outer loop (~27 k char system prompt)
# sub  = Task-tool sub-agent  (~3 k char system prompt)
# 10 k sits cleanly between the two clusters.
SUB_AGENT_SYSTEM_PROMPT_THRESHOLD = 10_000


def load_data(save_dir: Path) -> np.ndarray:
    """Load the canonical per-turn structured array from <save-dir>/data.npz.
    Build it first with `python metrics.py --save-dir <save-dir>`."""
    return np.load(save_dir / "data.npz")["turns"]


# ---------------------------------------------------------------------------
# Derivation helpers — compute on-the-fly from raw fields.
# ---------------------------------------------------------------------------


def isl_cached(t: np.ndarray) -> np.ndarray:
    """Tokens served from prefix cache this turn = isl - isl_new."""
    return np.maximum(t["isl"].astype(np.int64) - t["isl_new"].astype(np.int64), 0)


def cache_hit_rate(t: np.ndarray) -> np.ndarray:
    """isl_cached / isl; 0 where isl=0."""
    isl = t["isl"].astype(np.float64)
    cached = isl_cached(t).astype(np.float64)
    return np.divide(cached, isl, out=np.zeros_like(isl), where=isl > 0)


# Per-tier breakdown of where each input token came from. The three buckets
# are disjoint (vLLM only queries the offload tier for tokens the local cache
# missed), so over `isl` they partition the input:
#
#     hbm_hit_rate + offload_hit_rate + recompute_rate == 1   (where isl > 0)
#
# Token counts are the raw numerators: HBM = prefix_cache_hits, offload =
# external_prefix_cache_hits, recompute = isl - those two.


def _rate(num: np.ndarray, isl: np.ndarray) -> np.ndarray:
    return np.divide(num, isl, out=np.zeros_like(isl), where=isl > 0)


def hbm_hit_rate(t: np.ndarray) -> np.ndarray:
    """Fraction of input served from the local HBM/GPU prefix cache."""
    isl = t["isl"].astype(np.float64)
    return _rate(t["prefix_cache_hits"].astype(np.float64), isl)


def offload_hit_rate(t: np.ndarray) -> np.ndarray:
    """Fraction of input served from the CPU/offload tier (KV connector)."""
    isl = t["isl"].astype(np.float64)
    return _rate(t["external_prefix_cache_hits"].astype(np.float64), isl)


def recompute_tokens(t: np.ndarray) -> np.ndarray:
    """Input tokens that hit neither tier and were recomputed in prefill,
    as the residual `isl - HBM hits - offload hits` (clamped ≥ 0). Closes
    the partition exactly; cross-checks against the measured isl_new."""
    isl = t["isl"].astype(np.int64)
    hits = t["prefix_cache_hits"].astype(np.int64) + t[
        "external_prefix_cache_hits"
    ].astype(np.int64)
    return np.maximum(isl - hits, 0)


def recompute_rate(t: np.ndarray) -> np.ndarray:
    """Fraction of input recomputed (prefix-cache miss). By construction
    hbm_hit_rate + offload_hit_rate + recompute_rate == 1 where isl > 0."""
    isl = t["isl"].astype(np.float64)
    return _rate(recompute_tokens(t).astype(np.float64), isl)


def ttft_ms(t: np.ndarray) -> np.ndarray:
    """Time-to-first-token reconstructed = queue_ms + prefill_ms.
    (vLLM's TTFT histogram is observed at first-token-time, so its _sum
    delta against our completion-triggered scrapes is always 0.)"""
    return t["queue_ms"].astype(np.float64) + t["prefill_ms"].astype(np.float64)


def agent(
    t: np.ndarray, threshold: int = SUB_AGENT_SYSTEM_PROMPT_THRESHOLD
) -> np.ndarray:
    """Classify each turn as 'main' or 'sub' from system_prompt_chars."""
    return np.where(t["system_prompt_chars"] < threshold, "sub", "main")


# ---------------------------------------------------------------------------
# Raw-file loaders (for code that wants the per-turn dicts).
# ---------------------------------------------------------------------------


def load_per_problem_rows(save_dir: Path) -> dict[str, list[dict]]:
    """Read every `<save_dir>/per_problem/<iid>.vllm.jsonl` into
    `{instance_id: [vllm_row_dict, ...]}`. Use `load_data` for plotting;
    this is for scripts that want raw watcher dicts."""
    out: dict[str, list[dict]] = {}
    for f in sorted((save_dir / "per_problem").glob("*.vllm.jsonl")):
        iid = f.name[: -len(".vllm.jsonl")]
        out[iid] = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return out


def load_problem_field(save_dir: Path, field: str) -> dict[str, str]:
    """Pull `<field>` from per_problem/<iid>.meta.json, keyed by instance_id.
    `difficulty` is run through DIFFICULTY_REMAP."""
    out: dict[str, str] = {}
    for f in sorted((save_dir / "per_problem").glob("*.meta.json")):
        meta = json.loads(f.read_text())
        v = meta.get(field)
        if isinstance(v, list):
            v = ",".join(map(str, v))
        if field == "difficulty":
            v = DIFFICULTY_REMAP.get(v, v)
        out[meta["instance_id"]] = v
    return out


# ---------------------------------------------------------------------------
# Build — join the raw per-problem capture streams into data.npz.
# ---------------------------------------------------------------------------

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
        ("prefix_cache_hits", "i4"),  # tokens served from local HBM cache
        ("external_prefix_cache_hits", "i4"),  # tokens served from offload tier
        ("stop_reason", "U16"),
        # Proxy-side raw measurements (proxy; zero/empty if not captured).
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
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


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


def build_records(save_dir: Path) -> np.ndarray:
    """Join each problem's watcher + proxy turns (by index) into one
    structured array; difficulty is pulled from meta.json and remapped."""
    per_problem_dir = save_dir / "per_problem"
    rows: list[tuple] = []
    for meta_path in sorted(per_problem_dir.glob("*.meta.json")):
        meta = json.loads(meta_path.read_text())
        iid = meta["instance_id"]
        difficulty = DIFFICULTY_REMAP.get(meta.get("difficulty") or "", meta.get("difficulty") or "")

        vllm_rows = _load_jsonl(per_problem_dir / f"{iid}.vllm.jsonl")
        proxy_rows = _load_jsonl(per_problem_dir / f"{iid}.proxy.jsonl")

        # At concurrency=1 the two streams should align row-for-row. If
        # they don't, truncate to the shorter and warn.
        n = min(len(vllm_rows), len(proxy_rows)) if proxy_rows else len(vllm_rows)
        if proxy_rows and len(vllm_rows) != len(proxy_rows):
            print(
                f"[metrics] {iid}: vllm={len(vllm_rows)} "
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
                    _i(v.get("prefix_cache_hits")),
                    _i(v.get("external_prefix_cache_hits")),
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
    ap.add_argument("--save-dir", required=True, type=Path)
    args = ap.parse_args()

    turns = build_records(args.save_dir)
    out = args.save_dir / "data.npz"
    np.savez_compressed(out, turns=turns)
    print(f"wrote {out}  ({len(turns):,} turns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
