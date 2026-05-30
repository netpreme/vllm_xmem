"""SWE-bench dataset (``swebench``) + the per-problem sandbox it runs in
(``sandbox``). Import from here; the submodule split is an implementation
detail.
"""

from __future__ import annotations

from pipeline.datasets.sandbox import Sandbox
from pipeline.datasets.swebench import get_dataset

__all__ = ["Sandbox", "get_dataset"]
