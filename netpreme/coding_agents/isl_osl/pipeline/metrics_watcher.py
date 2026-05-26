"""Background per-turn metrics watcher for a vLLM server.

This process replaces the request-intercepting proxy with a polling
collector that talks to vLLM's `/metrics` Prometheus endpoint directly.

How it works (concurrency must be 1):
    1. We snapshot `/metrics` every `poll_interval_s` seconds.
    2. Whenever `vllm:e2e_request_latency_seconds_count` increments, one
       request has just completed; the delta of every other counter
       across the two snapshots is that request's contribution.
    3. We attribute the resulting row to whichever `instance_id` the
       caller has written to the control file (`run.sh` updates it before
       launching `claude -p` for each problem).
    4. The row gets appended to `<csv_dir>/<instance_id>.csv` in the same
       schema the analysis pipeline already consumes.

Why a separate watcher instead of an HTTP proxy:
    - Claude-CLI talks straight to vLLM, removing one network hop.
    - We get vLLM-internal timing (queue, prefill, decode, e2e, ITL) for
      free, plus GPU KV-cache utilization.
    - The watcher does no HTTP forwarding so it cannot get in the way of
      the request path; if it crashes mid-run, vLLM keeps serving and we
      just lose telemetry for the affected turns.

What we no longer have, compared to the old proxy:
    - `category` / `num_tool_calls` (required SSE-content parsing).
    - `agent` main-vs-sub classification (required reading the request's
      system prompt). The schema keeps the column but it is always
      "main" — sub-agent calls are indistinguishable from outside vLLM.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from agent_labels import LabelReader, LabelRecord


# ---------------------------------------------------------------------------
# Output schema. Kept in lock-step with analysis/build_data.py.
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "ts", "instance_id", "elapsed_ms",
    "ttft_ms", "prefill_ms", "decode_ms", "itl_ms", "queue_ms",
    "isl", "osl", "isl_new", "isl_cached", "cache_hit_rate",
    "stop_reason", "agent", "num_tool_defs", "num_messages",
    "system_prompt_chars", "kv_cache_usage_pct",
]


# ---------------------------------------------------------------------------
# vLLM Prometheus metric names. See `curl localhost:8000/metrics` on a
# running server for the full set.
# ---------------------------------------------------------------------------

class _Metric:
    """String constants for the metrics we care about, grouped for clarity.

    A note on metric selection: vLLM has two flavors of token-count metrics.
    The plain `vllm:prompt_tokens_total` and `vllm:generation_tokens_total`
    counters are updated *incrementally during* prefill and decode (i.e.,
    they grow as the model burns through tokens). The per-request
    histograms `vllm:request_prompt_tokens_*` and
    `vllm:request_generation_tokens_*`, in contrast, are updated *atomically
    at request completion* alongside the timing histograms.

    We MUST use the per-request histograms here. If we mixed them with a
    completion-event trigger (e.g. `e2e_request_latency_seconds_count`),
    we'd race: by the time the trigger ticks, the incremental counters
    have already eaten the value across an earlier scrape interval and
    the delta we compute would be zero. The per-request histograms tick
    in lockstep with the trigger.
    """

    # Cumulative histogram _sum counters (seconds). Atomic at request
    # completion — diff across two snapshots is the per-request value.
    # We intentionally skip `vllm:time_to_first_token_seconds_sum` and
    # `vllm:e2e_request_latency_seconds_sum`: the former races our
    # completion trigger (vLLM observes TTFT at first-token-time, not at
    # completion), and we reconstruct e2e from local wall-clock deltas.
    PREFILL_SUM      = "vllm:request_prefill_time_seconds_sum"
    DECODE_SUM       = "vllm:request_decode_time_seconds_sum"
    QUEUE_SUM        = "vllm:request_queue_time_seconds_sum"
    TPOT_SUM         = "vllm:request_time_per_output_token_seconds_sum"

    # Per-request token histograms — atomic at completion.
    PROMPT_TOKENS_SUM = "vllm:request_prompt_tokens_sum"
    GEN_TOKENS_SUM    = "vllm:request_generation_tokens_sum"
    # Uncached input tokens that actually went through prefill compute.
    # `isl_cached` is then derived as `isl - prefill_kv_computed`.
    PREFILL_KV_COMPUTED_SUM = "vllm:request_prefill_kv_computed_tokens_sum"

    # Labelled by finished_reason — the label whose counter incremented
    # by 1 between scrapes is the stop_reason for this turn.
    FINISH_REASON    = "vllm:request_success_total"

    # Instantaneous gauge — read directly, no diff.
    KV_USAGE_PCT     = "vllm:kv_cache_usage_perc"

    # The trigger we watch for new request completion. Any `_count` of a
    # per-request histogram would work; we pick the one tied to the same
    # event vLLM uses to publish the rest of the per-request values.
    REQUEST_COUNT    = "vllm:request_prompt_tokens_count"


# Finished-reason labels emitted by vLLM. Order is the priority we pick
# in case multiple incremented within a single scrape interval (rare).
_FINISHED_REASONS = ("stop", "length", "abort", "error", "repetition")


# ---------------------------------------------------------------------------
# Prometheus text parsing.
# ---------------------------------------------------------------------------

_LINE = re.compile(r"^(.+?)\s+([0-9eE.+\-]+|NaN|\+Inf|\-Inf)$")


def parse_prometheus_text(body: str) -> dict[str, float]:
    """Parse the body of a Prometheus `/metrics` response into
    `{metric_line: value}` where `metric_line` is everything up to (but
    not including) the trailing whitespace and value — labels and all.

    This is a tiny purpose-built parser; we don't need the full
    `prometheus_client` library for our handful of reads.
    """
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


def metric_value(metrics: dict[str, float], name_prefix: str,
                 label_substring: str = "") -> float:
    """Return the first metric line whose key starts with `name_prefix`
    and (optionally) contains `label_substring`. Returns 0.0 if absent.

    vLLM's labels include `engine="0"` and `model_name="..."`. We don't
    pin them — we just match on the metric-name prefix and (when we need
    to distinguish labelled siblings) a substring of the label set.
    """
    for k, v in metrics.items():
        if not k.startswith(name_prefix):
            continue
        if label_substring and label_substring not in k:
            continue
        return v
    return 0.0


# ---------------------------------------------------------------------------
# Snapshot data class.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Snapshot:
    """A point-in-time view of the metrics we need for one row.

    Stored unitless / cumulative as Prometheus reports them. Differencing
    happens in `derive_row`.
    """

    request_count:           int
    timing_seconds_sum:      dict[str, float]   # prefill | decode | queue | tpot
    prompt_tokens:           int                # per-request histogram _sum
    gen_tokens:              int
    prefill_kv_computed:     int                # uncached input tokens (= isl_new)
    finished_reason_counts:  dict[str, int]
    kv_usage_pct:            float
    wall_time:               float              # local wall clock when scrape returned

    @classmethod
    def from_metrics(cls, metrics: dict[str, float]) -> "Snapshot":
        timings = {
            "prefill": metric_value(metrics, _Metric.PREFILL_SUM),
            "decode":  metric_value(metrics, _Metric.DECODE_SUM),
            "queue":   metric_value(metrics, _Metric.QUEUE_SUM),
            "tpot":    metric_value(metrics, _Metric.TPOT_SUM),
        }
        finished = {
            reason: int(metric_value(metrics, _Metric.FINISH_REASON,
                                     f'finished_reason="{reason}"'))
            for reason in _FINISHED_REASONS
        }
        return cls(
            request_count=int(metric_value(metrics, _Metric.REQUEST_COUNT)),
            timing_seconds_sum=timings,
            prompt_tokens=int(metric_value(metrics, _Metric.PROMPT_TOKENS_SUM)),
            gen_tokens=int(metric_value(metrics, _Metric.GEN_TOKENS_SUM)),
            prefill_kv_computed=int(metric_value(metrics,
                                                 _Metric.PREFILL_KV_COMPUTED_SUM)),
            finished_reason_counts=finished,
            kv_usage_pct=metric_value(metrics, _Metric.KV_USAGE_PCT),
            wall_time=time.time(),
        )


# ---------------------------------------------------------------------------
# Per-request derivation.
# ---------------------------------------------------------------------------

def _diff_finished_reason(before: Snapshot, after: Snapshot) -> str:
    """Return the label whose counter went up between the two snapshots.
    Returns "" if nothing incremented (shouldn't happen for a successful
    completion)."""
    for reason in _FINISHED_REASONS:
        if (after.finished_reason_counts.get(reason, 0)
                - before.finished_reason_counts.get(reason, 0)) >= 1:
            return reason
    return ""


def derive_row(before: Snapshot, after: Snapshot, instance_id: str,
               label: LabelRecord | None) -> dict[str, Any]:
    """Build one CSV row from two consecutive snapshots that bracket
    exactly one request completion (concurrency=1).

    `label` is the matching record from the agent-labeler queue, or None
    if the labeler isn't running. We default the agent-side fields to
    something sensible so the schema is always populated."""

    def delta_ms(name: str) -> float:
        return (after.timing_seconds_sum[name]
                - before.timing_seconds_sum[name]) * 1000.0

    isl     = after.prompt_tokens       - before.prompt_tokens
    osl     = after.gen_tokens          - before.gen_tokens
    isl_new = after.prefill_kv_computed - before.prefill_kv_computed
    isl_cached = max(isl - isl_new, 0)
    cache_hit  = (isl_cached / isl) if isl > 0 else 0.0

    queue_ms   = delta_ms("queue")
    prefill_ms = delta_ms("prefill")
    decode_ms  = delta_ms("decode")
    # `time_to_first_token_seconds` is observed by vLLM at first-token
    # time (mid-request), so its _sum delta against our completion-
    # triggered scrapes is 0. Reconstruct TTFT from queue + prefill,
    # both of which are atomic at completion. At concurrency=1 the queue
    # is essentially zero, so TTFT ≈ prefill time.
    ttft_ms = queue_ms + prefill_ms

    return {
        "ts":                 before.wall_time,
        "instance_id":        instance_id,
        "elapsed_ms":         int((after.wall_time - before.wall_time) * 1000),
        "ttft_ms":            round(ttft_ms,    2),
        "prefill_ms":         round(prefill_ms, 2),
        "decode_ms":          round(decode_ms,  2),
        "itl_ms":             round(delta_ms("tpot"), 3) if osl > 0 else None,
        "queue_ms":           round(queue_ms,   2),
        "isl":                isl,
        "osl":                osl,
        "isl_new":            isl - isl_cached,
        "isl_cached":         isl_cached,
        "cache_hit_rate":     round(cache_hit, 4),
        "stop_reason":        _diff_finished_reason(before, after),
        "agent":               label.agent               if label else "main",
        "num_tool_defs":       label.num_tool_defs       if label else 0,
        "num_messages":        label.num_messages        if label else 0,
        "system_prompt_chars": label.system_prompt_chars if label else 0,
        "kv_cache_usage_pct": round(after.kv_usage_pct * 100, 3),
    }


# ---------------------------------------------------------------------------
# Filesystem helpers.
# ---------------------------------------------------------------------------

_SAFE_FILE_RE = re.compile(r"[^A-Za-z0-9_\-.]")


def safe_filename(s: str) -> str:
    """Translate an instance_id (which may contain `/`) into a filename."""
    return _SAFE_FILE_RE.sub("_", s)[:200]


class PerProblemCSV:
    """Append-only writer that lazily creates one file per `instance_id`
    inside `csv_dir`. Writes a header on first row."""

    def __init__(self, csv_dir: Path) -> None:
        self._dir = csv_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._seen: set[Path] = set()

    def write(self, row: dict[str, Any]) -> None:
        iid = row.get("instance_id")
        if not iid:
            return
        path = self._dir / f"{safe_filename(str(iid))}.csv"
        new_file = path not in self._seen and (not path.exists()
                                               or path.stat().st_size == 0)
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if new_file:
                w.writeheader()
            w.writerow({k: row.get(k) for k in CSV_COLUMNS})
        self._seen.add(path)


# ---------------------------------------------------------------------------
# Control file: a tiny text file `run.sh` rewrites to tell us which
# problem is currently active. Watcher reads it at every scrape so each
# row gets the right instance_id.
# ---------------------------------------------------------------------------

def read_active_instance(control_file: Path) -> str:
    """Return the current `instance_id`, or "" if the file is empty/missing."""
    try:
        return control_file.read_text().strip()
    except FileNotFoundError:
        return ""


# ---------------------------------------------------------------------------
# Watcher loop.
# ---------------------------------------------------------------------------

class MetricsWatcher:
    """Polls vLLM `/metrics` and writes one CSV row per detected request.

    Designed to be run as a background process for the lifetime of a
    `run.sh` invocation. Polling is asynchronous so we never block the
    GPU; one HTTP call per tick.
    """

    def __init__(self, vllm_url: str, control_file: Path, csv_dir: Path,
                 labels_file: Path | None = None,
                 poll_interval_s: float = 0.1) -> None:
        self._url             = vllm_url.rstrip("/")
        self._control_file    = control_file
        self._csv             = PerProblemCSV(csv_dir)
        self._labels          = LabelReader(labels_file) if labels_file else None
        self._poll_interval_s = poll_interval_s

    async def _scrape(self, client: httpx.AsyncClient) -> Snapshot:
        r = await client.get(f"{self._url}/metrics", timeout=10.0)
        return Snapshot.from_metrics(parse_prometheus_text(r.text))

    async def run(self) -> None:
        """Main polling loop. Returns when the process is killed.

        We never raise on transient `/metrics` failures: vLLM is killed and
        restarted between problems by `reset_vllm.sh`, and our scrape can
        race that. A failed scrape just clears `previous`; the next
        successful one becomes the new baseline. This applies to the very
        first scrape too — so the watcher can be safely launched while
        vLLM is still warming up.
        """
        async with httpx.AsyncClient() as client:
            previous: Snapshot | None = None
            while previous is None:
                try:
                    previous = await self._scrape(client)
                except httpx.HTTPError as exc:
                    print(f"[metrics-watcher] baseline scrape error: {exc!r}",
                          file=sys.stderr, flush=True)
                    await asyncio.sleep(self._poll_interval_s)
            print(f"[metrics-watcher] baseline scrape: "
                  f"request_count={previous.request_count}", flush=True)

            while True:
                await asyncio.sleep(self._poll_interval_s)
                try:
                    current = await self._scrape(client)
                except httpx.HTTPError as exc:
                    # vLLM may be restarting between problems — log and
                    # treat the next scrape as a new baseline.
                    print(f"[metrics-watcher] scrape error: {exc!r}",
                          file=sys.stderr, flush=True)
                    previous = None
                    continue
                if previous is None:
                    previous = current
                    continue

                completed = current.request_count - previous.request_count
                if completed >= 1:
                    instance_id = read_active_instance(self._control_file)
                    label = self._labels.pop() if self._labels else None
                    row = derive_row(previous, current, instance_id, label)
                    self._csv.write(row)
                    if completed > 1:
                        print(f"[metrics-watcher] WARNING: {completed} "
                              f"requests completed in one tick — "
                              f"attribution may be lossy", flush=True)
                previous = current


# ---------------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-url",        required=True,
                        help="vLLM base URL, e.g. http://localhost:8000")
    parser.add_argument("--control-file",    required=True, type=Path,
                        help="text file that run.sh updates with the "
                             "current instance_id")
    parser.add_argument("--per-problem-csv-dir", required=True, type=Path,
                        help="directory to write per-problem CSVs into")
    parser.add_argument("--labels-file",     type=Path, default=None,
                        help="optional file produced by pipeline/"
                             "agent_labeler.py; if set, the watcher pops "
                             "one record per detected completion and "
                             "fills in agent/num_tool_defs/etc.")
    parser.add_argument("--poll-interval-s", type=float, default=0.1,
                        help="seconds between /metrics scrapes")
    args = parser.parse_args()

    watcher = MetricsWatcher(
        vllm_url=args.vllm_url,
        control_file=args.control_file,
        csv_dir=args.per_problem_csv_dir,
        labels_file=args.labels_file,
        poll_interval_s=args.poll_interval_s,
    )
    try:
        asyncio.run(watcher.run())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
