"""The analytic dataset: build it, load it, derive from it.

The canonical per-turn dataset (`<save-dir>/data.npz`) stores RAW
MEASUREMENTS ONLY. This module owns its whole lifecycle:

  * `build_records` / `main` — join each problem's raw capture streams
    (`<iid>/vllm.jsonl` + `<iid>/proxy.jsonl` + `<iid>/meta.json`) into
    data.npz. Run as a script:  `python metrics.py --save-dir <save-dir>`.
  * `load_data` — read the structured array back.
  * Derivation helpers — compute non-stored fields (`isl_cached`,
    `cache_hit_rate`, per-tier hit rates, `ttft_ms`, `agent`) on the fly.
    Use these from plot scripts instead of expecting the field on the array.

Raw schema is set by pipeline/metrics_watcher.py (vLLM-side fields) and
pipeline/proxy/ (proxy-side fields).
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


# Claude Code's sub-agent spawner is the tool named exactly "Task". The
# todo-list tools (TaskCreate / TaskUpdate / TaskOutput) are NOT spawners, so
# match the bare name — substring matching would false-positive on those.
SUBAGENT_TOOL = "Task"


def uses_subagents(t: np.ndarray) -> bool:
    """True iff any turn invoked the Task sub-agent spawner.

    The toolset is the harness's (claude-cli), not the model's, so Task is
    offered regardless of model — but a served model may simply never spawn a
    sub-agent. When it doesn't, there is no sub tier to label and callers
    should not draw a main/sub split at all."""
    return any(SUBAGENT_TOOL in names.split(",") for names in t["tool_names"])


def agent(
    t: np.ndarray, threshold: int = SUB_AGENT_SYSTEM_PROMPT_THRESHOLD
) -> np.ndarray:
    """Classify each turn as 'main' or 'sub'.

    A sub-agent runs on its own (smaller) system prompt, so within a run that
    actually spawns sub-agents the prompt size separates the tiers. But if the
    run never invokes the Task spawner, there is no sub tier — the size
    threshold would otherwise misfire (e.g. a model whose only system prompt
    is already below `threshold`), so label everything 'main'."""
    if not uses_subagents(t):
        return np.full(len(t), "main")
    return np.where(t["system_prompt_chars"] < threshold, "sub", "main")


# ---------------------------------------------------------------------------
# Raw-file loaders (for code that wants the per-turn dicts).
# ---------------------------------------------------------------------------


def load_per_problem_rows(save_dir: Path) -> dict[str, list[dict]]:
    """Read every `<save_dir>/telemetry/<iid>/vllm.jsonl` into
    `{instance_id: [vllm_row_dict, ...]}`. Use `load_data` for plotting;
    this is for scripts that want raw watcher dicts."""
    out: dict[str, list[dict]] = {}
    for f in sorted((save_dir / "telemetry").glob("*/vllm.jsonl")):
        iid = f.parent.name
        out[iid] = [json.loads(ln) for ln in f.read_text().splitlines() if ln.strip()]
    return out


def load_problem_field(save_dir: Path, field: str) -> dict[str, str]:
    """Pull `<field>` from telemetry/<iid>/meta.json, keyed by instance_id.
    `difficulty` is run through DIFFICULTY_REMAP."""
    out: dict[str, str] = {}
    for f in sorted((save_dir / "telemetry").glob("*/meta.json")):
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
        ("tools_chars", "i4"),
        ("messages_chars", "i4"),
        ("num_tool_defs", "i4"),
        ("num_messages", "i4"),
        ("num_tool_calls", "i4"),
        ("claude_stop_reason", "U16"),
        ("tool_names", "U256"),
        ("has_thinking", "?"),
        ("response_text_chars", "i4"),
    ]
)


def build_records(save_dir: Path) -> np.ndarray:
    """Join each problem's watcher + proxy turns (by index) into one
    structured array; difficulty is pulled from meta.json and remapped."""
    telemetry_dir = save_dir / "telemetry"
    rows: list[tuple] = []
    for meta_path in sorted(telemetry_dir.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        instance_id = meta["instance_id"]
        raw_difficulty = meta.get("difficulty") or ""
        difficulty = DIFFICULTY_REMAP.get(raw_difficulty, raw_difficulty)

        problem_dir = meta_path.parent
        vllm_rows = _load_jsonl(problem_dir / "vllm.jsonl")
        proxy_rows = _load_jsonl(problem_dir / "proxy.jsonl")

        # At concurrency=1 the two streams should align row-for-row. If
        # they don't, truncate to the shorter and warn.
        num_turns = (
            min(len(vllm_rows), len(proxy_rows)) if proxy_rows else len(vllm_rows)
        )
        if proxy_rows and len(vllm_rows) != len(proxy_rows):
            print(
                f"[metrics] {instance_id}: vllm={len(vllm_rows)} "
                f"proxy={len(proxy_rows)} — truncating to {num_turns}"
            )

        for turn, vllm in enumerate(vllm_rows[:num_turns], start=1):
            proxy = proxy_rows[turn - 1] if proxy_rows else {}
            rows.append(
                (
                    instance_id,
                    difficulty,
                    turn,
                    _as_float(vllm.get("ts")),
                    _as_int(vllm.get("isl")),
                    _as_int(vllm.get("osl")),
                    _as_int(vllm.get("isl_new")),
                    _as_float(vllm.get("prefill_ms")),
                    _as_float(vllm.get("decode_ms")),
                    _as_float(vllm.get("queue_ms")),
                    _as_float(vllm.get("e2e_ms")),
                    _as_float(vllm.get("itl_ms")),
                    _as_float(vllm.get("kv_cache_usage_pct")),
                    _as_int(vllm.get("prefix_cache_hits")),
                    _as_int(vllm.get("external_prefix_cache_hits")),
                    vllm.get("stop_reason") or "",
                    _as_int(proxy.get("system_prompt_chars")),
                    _as_int(proxy.get("tools_chars")),
                    _as_int(proxy.get("messages_chars")),
                    _as_int(proxy.get("num_tool_defs")),
                    _as_int(proxy.get("num_messages")),
                    _as_int(proxy.get("num_tool_calls")),
                    proxy.get("claude_stop_reason") or "",
                    ",".join(proxy.get("tool_names") or [])[:256],
                    bool(proxy.get("has_thinking")),
                    _as_int(proxy.get("response_text_chars")),
                )
            )
    return np.array(rows, dtype=DTYPE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", required=True, type=Path)
    args = parser.parse_args()

    turns = build_records(args.save_dir)
    out_path = args.save_dir / "data.npz"
    np.savez_compressed(out_path, turns=turns)
    print(f"wrote {out_path}  ({len(turns):,} turns)")
    return 0


def _rate(num: np.ndarray, isl: np.ndarray) -> np.ndarray:
    return np.divide(num, isl, out=np.zeros_like(isl), where=isl > 0)


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _as_float(value) -> float:
    """Coerce to float; NaN for None/missing."""
    try:
        return float(value) if value is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _as_int(value) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
