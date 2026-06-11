"""Runtime: scrape vLLM's Prometheus `/metrics` once per turn, in-process.

``MetricsScraper`` (context manager) runs ``Poller`` on a background thread
for the duration of one problem — no subprocess.
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import httpx
from loguru import logger
from pipeline.utils.jsonl import JsonlWriter, instance_dir
from pipeline.vllm_metrics.prometheus import parse_raw_response
from pipeline.vllm_metrics.snapshot import Snapshot, compute_turn_metrics


class MetricsScraper:
    """Scrape vLLM's Prometheus ``/metrics`` once per turn, in-process.

    On enter, starts a background thread that polls ``/metrics`` every
    ``poll_interval_s``; on exit, signals it to stop and joins it. The poll
    loop fires once per completed request: it watches
    ``vllm:request_prompt_tokens_count`` and, each time it increments, records
    the delta of every other counter as that turn's row. Relies on
    **concurrency = 1** so each increment maps to exactly one turn. Rows are
    RAW measurements appended to ``save_dir/telemetry/<instance_id>/vllm_metrics.jsonl``.
    """

    def __init__(
        self,
        url: str,
        save_dir: Path,
        instance_id: str,
        poll_interval_s: float = 0.1,
        *,
        enabled: bool = True,
    ) -> None:
        # enabled=False (Anthropic backend): no local vLLM to scrape, so this
        # is a no-op context manager. Kept in the with-block uniformly.
        self.enabled = enabled
        self.save_dir = save_dir
        self.instance_id = instance_id
        self.out_dir = save_dir / "telemetry"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._poller = Poller(
            url=url,
            instance_id=instance_id,
            out_dir=self.out_dir,
            poll_interval_s=poll_interval_s,
        )
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> MetricsScraper:
        if not self.enabled:
            return self
        # Truncate any prior file for this id (retry-on-resume safety).
        (instance_dir(self.out_dir, self.instance_id) / "vllm_metrics.jsonl").unlink(
            missing_ok=True
        )
        self._thread = threading.Thread(
            target=self._poller.run_blocking,
            args=(self._stop,),
            name="vllm-metrics-scraper",
            daemon=True,
        )
        self._thread.start()
        logger.info("started metrics scraper thread for {}", self.instance_id)
        return self

    def __exit__(self, *exc) -> bool:
        if not self.enabled:
            return False
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None
        logger.info("stopped metrics scraper thread for {}", self.instance_id)
        return False


class Poller:
    """Poll vLLM `/metrics` and write one JSONL row per detected request,
    all attributed to a single `instance_id`."""

    def __init__(
        self,
        url: str,
        instance_id: str,
        out_dir: Path,
        poll_interval_s: float = 0.1,
    ) -> None:
        self._url = url.rstrip("/")
        self._instance_id = instance_id
        self._out = JsonlWriter(out_dir, "vllm_metrics.jsonl")
        self._poll_interval_s = poll_interval_s

    def run_blocking(self, stop_event: threading.Event) -> None:
        """Thread entry point: drive the async loop until `stop_event` is set."""
        asyncio.run(self.run(stop_event))

    async def scrape_prometheus_metrics(self, client: httpx.AsyncClient) -> Snapshot:
        resp = await client.get(f"{self._url}/metrics", timeout=10.0)
        return Snapshot.from_metrics(parse_raw_response(resp.text))

    async def run(self, stop_event: threading.Event) -> None:
        """Polling loop; returns when `stop_event` is set.

        Transient `/metrics` failures never raise: vLLM is killed and
        restarted between problems by `Server`. A failed scrape clears
        the baseline; the next successful one re-baselines.
        """
        async with httpx.AsyncClient() as client:
            previous: Snapshot | None = None
            while previous is None and not stop_event.is_set():
                try:
                    previous = await self.scrape_prometheus_metrics(client)
                except httpx.HTTPError as exc:
                    logger.warning("baseline scrape error: {!r}", exc)
                    await asyncio.sleep(self._poll_interval_s)
            if previous is not None:
                logger.info("baseline scrape: request_count={}", previous.request_count)

            # Max kv_cache_usage gauge seen across this turn's in-flight polls.
            # vLLM frees a request's KV the instant it finishes, so the gauge is
            # ~0 at completion; the peak mid-request is the occupancy we report.
            peak_kv_usage = 0.0
            while not stop_event.is_set():
                await asyncio.sleep(self._poll_interval_s)
                try:
                    current = await self.scrape_prometheus_metrics(client)
                except httpx.HTTPError as exc:
                    logger.warning("scrape error: {!r}", exc)
                    previous = None
                    peak_kv_usage = 0.0
                    continue
                if previous is None:
                    previous = current
                    continue

                peak_kv_usage = max(peak_kv_usage, current.kv_usage_pct)
                completed = current.request_count - previous.request_count
                if completed >= 1:
                    row = compute_turn_metrics(
                        before=previous, after=current, peak_kv_usage=peak_kv_usage
                    )
                    self._out.write(instance_id=self._instance_id, row=row)
                    if completed > 1:
                        logger.warning(
                            "{} requests completed in one tick — "
                            "attribution may be lossy",
                            completed,
                        )
                    # Advance the baseline ONLY on a completion, so each turn's
                    # delta window spans its whole lifecycle. Prefix-cache
                    # counters (prefix_cache_hits/queries, external hits) tick at
                    # PREFILL, not completion; advancing every poll would leave
                    # those jumps outside the window and record them as 0.
                    previous = current
                    peak_kv_usage = 0.0  # start the next turn's peak fresh
