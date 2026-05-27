"""Single source of truth for benchmark configuration.

Today: SETUP_DEFAULTS dict + SetupSpec dataclass.
Future (phase 5): BenchmarkConfig dataclass that the CLI builds once and
the runners consume directly (instead of via sys.argv munging).
"""
from dataclasses import dataclass
from pathlib import Path


# Per-setup defaults: which port and which GPU each setup binds to.
# Used by capture/replay to assign distinct (port, gpus) per backend so
# Prometheus can label them and they don't collide on hardware.
SETUP_DEFAULTS: dict[str, tuple[int, str]] = {
    "hybrid-mtier": (8001, "0"),
    "mtier-only":   (8001, "0"),
    "hybrid-cpu":   (8002, "1"),
    "cpu-only":     (8002, "1"),
}


@dataclass
class SetupSpec:
    """One backend's coordinates."""
    setup: str
    port: int
    gpus: str | None
    base_url: str
    workspace_root: Path | None = None
