"""Background per-turn vLLM metrics watcher.

Polls vLLM's `/metrics` Prometheus endpoint and, on each detected request
completion (concurrency must be 1), appends one JSON object — RAW
measurements only — to `<per_problem_dir>/<instance_id>.vllm.jsonl`.

The instance_id is read from `--control-file` on each completion, which
`coding_agent.py` rewrites before launching `claude -p` for each problem.

What this file does NOT do:
    - Compute derivations (isl_cached, cache_hit_rate, ttft_ms, …).
      Those live in the analysis layer.
    - Merge proxy data. The agent_labeler writes its own JSONL; the
      analysis layer joins on turn index.

How completion detection works:
    1. Snapshot `/metrics` every `poll_interval_s`.
    2. Whenever `vllm:request_prompt_tokens_count` increments, one
       request has just completed; the delta of every other counter
       across the two snapshots is that request's contribution.
    3. Write the row, attributed to the current control-file instance_id.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx


# ---------------------------------------------------------------------------
# vLLM Prometheus metric names. See `curl localhost:8000/metrics`.
# ---------------------------------------------------------------------------


class _Metric:
    """Per-request histogram _sum counters (seconds). These tick atomically
    at request completion, in lockstep with our completion trigger.

    We intentionally skip `vllm:time_to_first_token_seconds_sum`: vLLM
    observes TTFT at first-token-time (mid-request), so its _sum delta
    against our completion-triggered scrapes is 0. TTFT is recomputed at
    analysis time as `queue_ms + prefill_ms`.
    """

    PREFILL_SUM = "vllm:request_prefill_time_seconds_sum"
    DECODE_SUM = "vllm:request_decode_time_seconds_sum"
    QUEUE_SUM = "vllm:request_queue_time_seconds_sum"
    TPOT_SUM = "vllm:request_time_per_output_token_seconds_sum"
    E2E_SUM = "vllm:e2e_request_latency_seconds_sum"

    # Per-request token histograms — atomic at completion.
    PROMPT_TOKENS_SUM = "vllm:request_prompt_tokens_sum"
    GEN_TOKENS_SUM = "vllm:request_generation_tokens_sum"
    # Uncached input tokens that actually went through prefill compute.
    PREFILL_KV_COMPUTED_SUM = "vllm:request_prefill_kv_computed_tokens_sum"

    # Labelled by finished_reason — the label whose counter incremented
    # by 1 between scrapes is the stop_reason for this turn.
    FINISH_REASON = "vllm:request_success_total"

    # Instantaneous gauge.
    KV_USAGE_PCT = "vllm:kv_cache_usage_perc"

    # Completion trigger.
    REQUEST_COUNT = "vllm:request_prompt_tokens_count"


_FINISHED_REASONS = ("stop", "length", "abort", "error", "repetition")


# ---------------------------------------------------------------------------
# Prometheus text parsing.
# ---------------------------------------------------------------------------

_LINE = re.compile(r"^(.+?)\s+([0-9eE.+\-]+|NaN|\+Inf|\-Inf)$")


def parse_prometheus_text(body: str) -> dict[str, float]:
    """Parse `/metrics` text into `{metric_line: value}` where `metric_line`
    is everything up to the trailing whitespace+value — labels included."""
    out: dict[str, float] = {}
    for line in body.splitlines():
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        try:
            out[m.group(1)] = float(m.group(2))
        except ValueError:
            pass
    return out


def metric_value(
    metrics: dict[str, float], name_prefix: str, label_substring: str = ""
) -> float:
    """First metric line that starts with `name_prefix` and (optionally)
    contains `label_substring`. Returns 0.0 if absent."""
    for k, v in metrics.items():
        if not k.startswith(name_prefix):
            continue
        if label_substring and label_substring not in k:
            continue
        return v
    return 0.0


# ---------------------------------------------------------------------------
# Snapshot.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Snapshot:
    request_count: int
    timing_seconds_sum: dict[str, float]  # prefill | decode | queue | tpot | e2e
    prompt_tokens: int
    gen_tokens: int
    prefill_kv_computed: int
    finished_reason_counts: dict[str, int]
    kv_usage_pct: float
    wall_time: float

    @classmethod
    def from_metrics(cls, metrics: dict[str, float]) -> "Snapshot":
        timings = {
            "prefill": metric_value(metrics, _Metric.PREFILL_SUM),
            "decode": metric_value(metrics, _Metric.DECODE_SUM),
            "queue": metric_value(metrics, _Metric.QUEUE_SUM),
            "tpot": metric_value(metrics, _Metric.TPOT_SUM),
            "e2e": metric_value(metrics, _Metric.E2E_SUM),
        }
        finished = {
            reason: int(
                metric_value(
                    metrics, _Metric.FINISH_REASON, f'finished_reason="{reason}"'
                )
            )
            for reason in _FINISHED_REASONS
        }
        return cls(
            request_count=int(metric_value(metrics, _Metric.REQUEST_COUNT)),
            timing_seconds_sum=timings,
            prompt_tokens=int(metric_value(metrics, _Metric.PROMPT_TOKENS_SUM)),
            gen_tokens=int(metric_value(metrics, _Metric.GEN_TOKENS_SUM)),
            prefill_kv_computed=int(
                metric_value(metrics, _Metric.PREFILL_KV_COMPUTED_SUM)
            ),
            finished_reason_counts=finished,
            kv_usage_pct=metric_value(metrics, _Metric.KV_USAGE_PCT),
            wall_time=time.time(),
        )


def _diff_finished_reason(before: Snapshot, after: Snapshot) -> str:
    for reason in _FINISHED_REASONS:
        if (
            after.finished_reason_counts.get(reason, 0)
            - before.finished_reason_counts.get(reason, 0)
        ) >= 1:
            return reason
    return ""


def derive_row(before: Snapshot, after: Snapshot) -> dict:
    """Build one row of raw measurements from two consecutive snapshots
    that bracket exactly one request completion (concurrency=1)."""

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
        "stop_reason": _diff_finished_reason(before, after),
        "kv_cache_usage_pct": round(after.kv_usage_pct * 100, 3),
    }


# ---------------------------------------------------------------------------
# Filesystem helpers.
# ---------------------------------------------------------------------------

_SAFE_FILE_RE = re.compile(r"[^A-Za-z0-9_\-.]")


def safe_filename(s: str) -> str:
    return _SAFE_FILE_RE.sub("_", s)[:200]


class PerProblemJSONL:
    """Append-only writer, lazy per-instance_id file in `out_dir`."""

    def __init__(self, out_dir: Path) -> None:
        self._dir = out_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def write(self, instance_id: str, row: dict) -> None:
        if not instance_id:
            return
        path = self._dir / f"{safe_filename(instance_id)}.vllm.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def read_active_instance(control_file: Path) -> str:
    try:
        return control_file.read_text().strip()
    except FileNotFoundError:
        return ""


# ---------------------------------------------------------------------------
# Watcher loop.
# ---------------------------------------------------------------------------


class MetricsWatcher:
    """Polls vLLM `/metrics` and writes one JSONL row per detected request."""

    def __init__(
        self,
        vllm_url: str,
        control_file: Path,
        out_dir: Path,
        poll_interval_s: float = 0.1,
    ) -> None:
        self._url = vllm_url.rstrip("/")
        self._control_file = control_file
        self._out = PerProblemJSONL(out_dir)
        self._poll_interval_s = poll_interval_s

    async def _scrape(self, client: httpx.AsyncClient) -> Snapshot:
        r = await client.get(f"{self._url}/metrics", timeout=10.0)
        return Snapshot.from_metrics(parse_prometheus_text(r.text))

    async def run(self) -> None:
        """Main polling loop. Returns when the process is killed.

        Transient `/metrics` failures never raise: vLLM is killed and
        restarted between problems by `reset_vllm.sh`. A failed scrape
        clears the baseline; the next successful one re-baselines.
        """
        async with httpx.AsyncClient() as client:
            previous: Snapshot | None = None
            while previous is None:
                try:
                    previous = await self._scrape(client)
                except httpx.HTTPError as exc:
                    print(
                        f"[metrics-watcher] baseline scrape error: {exc!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                    await asyncio.sleep(self._poll_interval_s)
            print(
                f"[metrics-watcher] baseline scrape: "
                f"request_count={previous.request_count}",
                flush=True,
            )

            while True:
                await asyncio.sleep(self._poll_interval_s)
                try:
                    current = await self._scrape(client)
                except httpx.HTTPError as exc:
                    print(
                        f"[metrics-watcher] scrape error: {exc!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                    previous = None
                    continue
                if previous is None:
                    previous = current
                    continue

                completed = current.request_count - previous.request_count
                if completed >= 1:
                    instance_id = read_active_instance(self._control_file)
                    row = derive_row(previous, current)
                    self._out.write(instance_id, row)
                    if completed > 1:
                        print(
                            f"[metrics-watcher] WARNING: {completed} "
                            f"requests completed in one tick — "
                            f"attribution may be lossy",
                            flush=True,
                        )
                previous = current


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vllm-url", required=True, help="vLLM base URL, e.g. http://localhost:8000"
    )
    parser.add_argument(
        "--control-file",
        required=True,
        type=Path,
        help="text file that coding_agent.py updates with the current instance_id",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="directory to write per-problem <instance_id>.vllm.jsonl into",
    )
    parser.add_argument(
        "--poll-interval-s",
        type=float,
        default=0.1,
        help="seconds between /metrics scrapes",
    )
    args = parser.parse_args()

    watcher = MetricsWatcher(
        vllm_url=args.vllm_url,
        control_file=args.control_file,
        out_dir=args.out_dir,
        poll_interval_s=args.poll_interval_s,
    )
    try:
        asyncio.run(watcher.run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
