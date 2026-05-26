"""Shared label-record format used by `agent_labeler` (producer) and
`metrics_watcher` (consumer).

The labeler classifies each incoming /v1/messages request (main agent vs
Task-tool sub-agent, from the system-prompt size) and appends one record
to `<run_dir>/.agent_labels`. The watcher pops one record per detected
vLLM completion and merges it into the CSV row.

This file owns the on-disk format (one JSON object per line) and the
read/write helpers — nothing else. Keeping it isolated means the wire
format is defined in exactly one place; both processes import from here.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Record schema.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LabelRecord:
    """One label per /v1/messages request, written in request-arrival order.

    Fields are deliberately limited to things the labeler can extract from
    the request body alone (no response inspection). Anything vLLM can
    measure better — token counts, timings, cache stats — lives in the
    Prometheus metrics the watcher already collects.
    """

    agent:               str   # "main" or "sub"
    num_tool_defs:       int   # len(body["tools"])
    num_messages:        int   # len(body["messages"])
    system_prompt_chars: int   # raw character count of the system prompt


# ---------------------------------------------------------------------------
# Writer — used by the labeler.
# ---------------------------------------------------------------------------

class LabelWriter:
    """Append-only writer. One JSON object per line, flushed immediately so
    the watcher can read it within ~100 ms."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: LabelRecord) -> None:
        line = json.dumps(asdict(record), separators=(",", ":")) + "\n"
        with self._path.open("a") as f:
            f.write(line)
            f.flush()


# ---------------------------------------------------------------------------
# Reader — used by the watcher.
# ---------------------------------------------------------------------------

class LabelReader:
    """Streaming reader that pops one record per call in append order.

    Uses a byte-offset cursor rather than a persistent file handle so it
    naturally handles run.sh truncating the file between problems: if the
    file shrinks below the cursor, we reset to the start.

    Returns `None` if no new record is available (file missing, empty, or
    no new line since the last pop). The watcher treats `None` as
    "labeler not running" and falls back to a default.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._offset = 0

    def pop(self) -> LabelRecord | None:
        if not self._path.exists():
            return None
        size = self._path.stat().st_size
        if size < self._offset:
            # The file was truncated (e.g. start of a new problem).
            self._offset = 0
        if size == self._offset:
            return None
        with self._path.open("r") as f:
            f.seek(self._offset)
            line = f.readline()
            # Ignore a partial trailing line — the writer hasn't flushed
            # the newline yet. We'll pick it up on the next pop.
            if not line.endswith("\n"):
                return None
            self._offset = f.tell()
        try:
            data = json.loads(line)
            return LabelRecord(**data)
        except (json.JSONDecodeError, TypeError):
            # Corrupt line — skip it and continue.
            return None
