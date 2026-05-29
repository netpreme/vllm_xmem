"""vLLM Prometheus metrics — scrape per turn, parse, derive.

Public API:
    MetricsScraper       context manager that scrapes /metrics per turn
    Snapshot, compute_turn_metrics pure model + per-turn derivation (for analysis/tests)
    parse_raw_response, extract_metric   raw /metrics text parsing
"""

from pipeline.vllm_metrics.prometheus import extract_metric, parse_raw_response
from pipeline.vllm_metrics.scraper import MetricsScraper
from pipeline.vllm_metrics.snapshot import Snapshot, compute_turn_metrics

__all__ = [
    "MetricsScraper",
    "Snapshot",
    "compute_turn_metrics",
    "parse_raw_response",
    "extract_metric",
]
