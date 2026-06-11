"""One `/metrics` scrape (``Snapshot``) and the per-turn row derived from two.

Pure data + math — no I/O. ``compute_turn_metrics(before, after)`` turns two snapshots
that bracket exactly one request completion (concurrency=1) into one raw row.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from pipeline.vllm_metrics.prometheus import (
    DECODE_SUM,
    E2E_SUM,
    EXTERNAL_PREFIX_CACHE_HITS,
    FINISH_REASON,
    FINISHED_REASONS,
    GEN_TOKENS_SUM,
    KV_USAGE_PCT,
    PREFILL_KV_COMPUTED_SUM,
    PREFILL_SUM,
    PREFIX_CACHE_HITS,
    PROMPT_TOKENS_SUM,
    QUEUE_SUM,
    REQUEST_COUNT,
    TPOT_SUM,
    extract_metric,
)


@dataclass(frozen=True)
class Snapshot:
    request_count: int
    timing_seconds_sum: dict[str, float]  # prefill | decode | queue | tpot | e2e
    prompt_tokens: int
    gen_tokens: int
    prefill_kv_computed: int
    finished_reason_counts: dict[str, int]
    kv_usage_pct: float
    prefix_cache_hits: int
    external_prefix_cache_hits: int
    wall_time: float

    @classmethod
    def from_metrics(cls, metrics: dict[str, float]) -> "Snapshot":
        timings = {
            "prefill": extract_metric(metrics=metrics, name_prefix=PREFILL_SUM),
            "decode": extract_metric(metrics=metrics, name_prefix=DECODE_SUM),
            "queue": extract_metric(metrics=metrics, name_prefix=QUEUE_SUM),
            "tpot": extract_metric(metrics=metrics, name_prefix=TPOT_SUM),
            "e2e": extract_metric(metrics=metrics, name_prefix=E2E_SUM),
        }
        finished = {
            reason: int(
                extract_metric(
                    metrics=metrics,
                    name_prefix=FINISH_REASON,
                    label_substring=f'finished_reason="{reason}"',
                )
            )
            for reason in FINISHED_REASONS
        }
        return cls(
            request_count=int(
                extract_metric(metrics=metrics, name_prefix=REQUEST_COUNT)
            ),
            timing_seconds_sum=timings,
            prompt_tokens=int(
                extract_metric(metrics=metrics, name_prefix=PROMPT_TOKENS_SUM)
            ),
            gen_tokens=int(extract_metric(metrics=metrics, name_prefix=GEN_TOKENS_SUM)),
            prefill_kv_computed=int(
                extract_metric(metrics=metrics, name_prefix=PREFILL_KV_COMPUTED_SUM)
            ),
            finished_reason_counts=finished,
            kv_usage_pct=extract_metric(metrics=metrics, name_prefix=KV_USAGE_PCT),
            prefix_cache_hits=int(
                extract_metric(metrics=metrics, name_prefix=PREFIX_CACHE_HITS)
            ),
            external_prefix_cache_hits=int(
                extract_metric(metrics=metrics, name_prefix=EXTERNAL_PREFIX_CACHE_HITS)
            ),
            wall_time=time.time(),
        )


def compute_turn_metrics(
    before: Snapshot, after: Snapshot, peak_kv_usage: float = 0.0
) -> dict:
    """Build one row of raw measurements from two consecutive snapshots
    that bracket exactly one request completion (concurrency=1).

    `peak_kv_usage` is the max of vLLM's instantaneous `kv_cache_usage_perc`
    gauge (0..1) seen across all polls *during* the turn. The gauge measures
    currently-allocated KV blocks, which vLLM frees the instant a request
    finishes — so `after.kv_usage_pct` (sampled at completion) is ~0 and
    useless. The peak, sampled mid-request, is the occupancy we actually want.
    """

    def delta_ms(name: str) -> float:
        return (
            after.timing_seconds_sum[name] - before.timing_seconds_sum[name]
        ) * 1000.0

    isl = after.prompt_tokens - before.prompt_tokens
    osl = after.gen_tokens - before.gen_tokens
    isl_new = after.prefill_kv_computed - before.prefill_kv_computed

    return {
        "ts": round(before.wall_time, 3),
        "isl": isl,
        "osl": osl,
        "isl_new": isl_new,
        "prefill_ms": round(delta_ms("prefill"), 2),
        "decode_ms": round(delta_ms("decode"), 2),
        "queue_ms": round(delta_ms("queue"), 2),
        "itl_ms": round(delta_ms("tpot"), 3) if osl > 0 else None,
        "e2e_ms": round(delta_ms("e2e"), 2),
        "stop_reason": _diff_finished_reason(before=before, after=after),
        # Peak KV-cache gauge across the turn's in-flight polls — the real
        # occupancy. (The instantaneous gauge sampled at completion is useless
        # here: vLLM frees the request's blocks on finish, so it reads ~0.)
        "kv_cache_usage_pct_peak": round(peak_kv_usage * 100, 3),
        # Prefix-cache hit-token deltas for this turn (rates derived in
        # analysis, over `isl` as the denominator):
        #   local (HBM) hit rate     = prefix_cache_hits / isl
        #   offload-tier hit rate    = external_prefix_cache_hits / isl
        "prefix_cache_hits": after.prefix_cache_hits - before.prefix_cache_hits,
        "external_prefix_cache_hits": (
            after.external_prefix_cache_hits - before.external_prefix_cache_hits
        ),
    }


def _diff_finished_reason(before: Snapshot, after: Snapshot) -> str:
    for reason in FINISHED_REASONS:
        if (
            after.finished_reason_counts.get(reason, 0)
            - before.finished_reason_counts.get(reason, 0)
        ) >= 1:
            return reason
    return ""
