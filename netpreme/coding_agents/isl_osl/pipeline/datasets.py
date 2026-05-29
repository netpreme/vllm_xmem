"""SWE-bench dataset helper.

``get_dataset`` is responsible for one thing: returning the list of
problems to work on. It fetches from HuggingFace, selects a slice (or a
seeded random sample), and drops any ids already solved. No file I/O —
the caller infers `solved_ids` from prior per-problem results, so a run
is reproducible from ``(name, start, end, random, seed)`` and resuming
is just calling it again.
"""

from __future__ import annotations

import random as _random

from datasets import load_dataset


def get_dataset(
    name: str,
    *,
    start: int = 0,
    end: int | None = None,
    random: int = 0,
    seed: int = 0,
    solved_ids: set[str] | None = None,
) -> list[dict]:
    """Return the SWE-bench problems to run, as a list of dicts.

    With ``random > 0``, deterministically shuffle by ``seed`` and take
    that many; otherwise take the ``[start:end]`` slice. Then drop every
    problem whose ``instance_id`` is in ``solved_ids`` — which is what
    makes a resume skip work already done."""
    rows = list(load_dataset(name, split="test"))
    if random:
        _random.Random(seed).shuffle(rows)
        rows = rows[:random]
    else:
        rows = rows[start:end]

    if solved_ids:
        rows = [r for r in rows if r["instance_id"] not in solved_ids]
    return rows
