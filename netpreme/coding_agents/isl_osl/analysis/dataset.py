"""Shared data-loading + derivation helpers for analysis/plot_*.py.

The canonical analytic dataset (`<run-dir>/data.npz`) stores RAW
MEASUREMENTS ONLY. This module adds:

  * `load_data` — read the structured array
  * Derivation helpers — compute commonly used non-stored fields
    (`isl_cached`, `cache_hit_rate`, `ttft_ms`, `agent`) from raw
    measurements. Use these from plot scripts instead of expecting the
    field on the array.
"""

from __future__ import annotations

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


def load_data(run_dir: Path) -> np.ndarray:
    """Load the canonical per-turn structured array from <run-dir>/data.npz.
    Build it first with `python build_dataset.py --run-dir <run-dir>`."""
    return np.load(run_dir / "data.npz")["turns"]


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


def load_per_problem_rows(run_dir: Path) -> dict[str, list[dict]]:
    """Read every `<run_dir>/per_problem/<iid>.vllm.jsonl` into
    `{instance_id: [vllm_row_dict, ...]}`. Use `load_data` for plotting;
    this is for scripts that want raw watcher dicts."""
    out: dict[str, list[dict]] = {}
    for f in sorted((run_dir / "per_problem").glob("*.vllm.jsonl")):
        iid = f.name[: -len(".vllm.jsonl")]
        out[iid] = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return out


def load_problem_field(run_dir: Path, field: str) -> dict[str, str]:
    """Pull `<field>` from per_problem/<iid>.meta.json, keyed by instance_id.
    `difficulty` is run through DIFFICULTY_REMAP."""
    out: dict[str, str] = {}
    for f in sorted((run_dir / "per_problem").glob("*.meta.json")):
        meta = json.loads(f.read_text())
        v = meta.get(field)
        if isinstance(v, list):
            v = ",".join(map(str, v))
        if field == "difficulty":
            v = DIFFICULTY_REMAP.get(v, v)
        out[meta["instance_id"]] = v
    return out
