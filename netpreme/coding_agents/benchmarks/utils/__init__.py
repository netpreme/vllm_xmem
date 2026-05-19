"""Reusable building blocks for the netpreme concurrent-agents benchmark.

The original implementations live in `bench_concurrent_users` (for capture
mode) and `bench_replay` (for replay mode). This package re-exports them
into small, topically-grouped modules so that test code and external callers
can `from utils import benchmark` rather than reaching into the entrypoint
scripts.

Top-level entry: `utils.benchmark.CodingAgents`
"""

from .benchmark import CodingAgents

__all__ = ["CodingAgents"]
