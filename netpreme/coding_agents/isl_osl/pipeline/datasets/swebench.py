"""SWE-bench dataset helper.

``get_dataset`` is responsible for one thing: returning the list of
problems to work on. It fetches from HuggingFace, optionally takes a
seeded random sample, and drops any ids already solved. No file I/O —
the caller infers `solved_ids` from prior per-problem results, so a run
is reproducible from ``(name, random, seed)`` and resuming is just
calling it again.

Supported datasets (CLI alias → HuggingFace id) live in ``DATASETS``.
Every row is normalized to the fields the pipeline relies on
(``instance_id`` / ``repo`` / ``base_commit`` / ``problem_statement``),
so downstream code never branches on the dataset.
"""

from __future__ import annotations

import json
import random as _random

# Absolute import — resolves to the HuggingFace ``datasets`` lib in
# site-packages, NOT this ``pipeline.datasets`` package (which is only
# reachable via its dotted name). Py3 imports are absolute by default.
from datasets import load_dataset

# CLI alias → HuggingFace dataset id.
DATASETS = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "swe-bench-pro": "ScaleAI/SWE-bench_Pro",
}

_SWE_BENCH_PRO = DATASETS["swe-bench-pro"]

# SWE-bench Pro text columns are JSON-string-encoded (the raw value is
# '"..."' with \n as a two-char escape). Decode them before use.
_PRO_TEXT_FIELDS = ("problem_statement", "requirements", "interface")


def get_dataset(
    name: str,
    *,
    random: int = 0,
    seed: int = 0,
    solved_ids: set[str] | None = None,
) -> list[dict]:
    """Return the SWE-bench problems to run, as a list of dicts.

    ``name`` is a ``DATASETS`` alias or a raw HuggingFace id. With
    ``random > 0``, deterministically shuffle by ``seed`` and take
    that many; otherwise take the whole split. Then drop every problem
    whose ``instance_id`` is in ``solved_ids`` — which is what makes a
    resume skip work already done."""
    name = DATASETS.get(name, name)
    rows = list(load_dataset(name, split="test"))
    if name == _SWE_BENCH_PRO:
        rows = [_normalize_pro(row) for row in rows]
    if random:
        _random.Random(seed).shuffle(rows)
        rows = rows[:random]

    if solved_ids:
        rows = [row for row in rows if row["instance_id"] not in solved_ids]
    return rows


def _normalize_pro(row: dict) -> dict:
    """Normalize one SWE-bench Pro row to the Verified-style fields the
    pipeline expects.

    Decodes the JSON-string-encoded text columns, then folds
    ``requirements`` and ``interface`` into ``problem_statement`` — the
    official Pro harness shows all three to the agent, and a single
    field keeps claude.PROMPT dataset-agnostic."""
    row = dict(row)
    for field in _PRO_TEXT_FIELDS:
        row[field] = _json_unescape(row.get(field) or "")

    parts = [row["problem_statement"]]
    if row["requirements"]:
        parts.append("# Requirements\n" + row["requirements"])
    if row["interface"]:
        parts.append("# Interface\n" + row["interface"])
    row["problem_statement"] = "\n\n".join(parts)
    return row


def _json_unescape(value: str) -> str:
    """Decode a JSON-string-encoded value ('"..."'); pass through raw text."""
    if value.startswith('"') and value.endswith('"'):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    return value
