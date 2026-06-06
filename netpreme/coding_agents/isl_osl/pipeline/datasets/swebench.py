"""SWE-bench dataset helper.

``get_dataset`` returns the problems to work on, in dataset order: it fetches
from HuggingFace, normalizes each row to the fields the pipeline relies on
(``instance_id`` / ``repo`` / ``base_commit`` / ``problem_statement``), and
drops any ids already solved. The caller caps the count with ``--limit``.

Supported datasets (CLI alias → HuggingFace id) live in ``DATASETS``.
"""

from __future__ import annotations

import json

# Absolute import — resolves to the HuggingFace ``datasets`` lib in
# site-packages, NOT this ``pipeline.datasets`` package.
from datasets import load_dataset

# CLI alias → HuggingFace dataset id.
DATASETS = {
    "verified": "princeton-nlp/SWE-bench_Verified",
    "pro": "ScaleAI/SWE-bench_Pro",
}

_SWE_BENCH_PRO = DATASETS["pro"]

# SWE-bench Pro text columns are JSON-string-encoded ('"..."' with \n escaped).
_PRO_TEXT_FIELDS = ("problem_statement", "requirements", "interface")


def get_dataset(
    name: str,
    *,
    solved_ids: set[str] | None = None,
) -> list[dict]:
    """Return the problems to run as a list of dicts, in dataset order. ``name``
    is a ``DATASETS`` alias or a raw HuggingFace id. Drops every problem whose
    ``instance_id`` is in ``solved_ids`` (so a resume skips solved work); the
    caller caps the count with ``--limit``."""
    name = DATASETS.get(name, name)
    rows = list(load_dataset(name, split="test"))
    if name == _SWE_BENCH_PRO:
        rows = [_normalize_pro(row) for row in rows]
    if solved_ids:
        rows = [row for row in rows if row["instance_id"] not in solved_ids]
    return rows


def _normalize_pro(row: dict) -> dict:
    """Normalize one SWE-bench Pro row to the Verified-style fields the pipeline
    expects. Decodes the JSON-string-encoded text columns, then folds
    ``requirements`` and ``interface`` into ``problem_statement`` (the official
    Pro harness shows all three) so claude.PROMPT stays dataset-agnostic."""
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
