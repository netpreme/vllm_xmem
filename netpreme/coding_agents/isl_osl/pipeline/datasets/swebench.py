"""SWE-bench dataset helper.

``get_dataset`` is responsible for one thing: returning the list of
problems to work on, in dataset order. It fetches from HuggingFace and
drops any ids already solved. No file I/O — the caller infers
`solved_ids` from prior per-problem results, so a run is reproducible
from ``name`` (the split is fixed-order) and resuming is just calling it
again. The caller caps the count with ``--limit``.
"""

from __future__ import annotations

# Absolute import — resolves to the HuggingFace ``datasets`` lib in
# site-packages, NOT this ``pipeline.datasets`` package (which is only
# reachable via its dotted name). Py3 imports are absolute by default.
from datasets import load_dataset


def get_dataset(
    name: str,
    *,
    solved_ids: set[str] | None = None,
) -> list[dict]:
    """Return the SWE-bench problems to run, as a list of dicts, in dataset
    order. Drop every problem whose ``instance_id`` is in ``solved_ids`` —
    which is what makes a resume skip work already done. Caller caps the
    count with ``--limit``."""
    rows = list(load_dataset(name, split="test"))
    if solved_ids:
        rows = [row for row in rows if row["instance_id"] not in solved_ids]
    return rows
