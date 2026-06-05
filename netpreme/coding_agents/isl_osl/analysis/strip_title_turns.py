"""Strip Claude Code's session-title requests from captured telemetry.

claude-cli fires a one-shot, toolless "generate a sentence-case title" request
at session start to label the session in its sidebar. vLLM short-circuits it
before the engine (vllm/entrypoints/anthropic/serving.py), so it never lands in
``vllm.jsonl`` — but the proxy tees it into ``proxy.jsonl``/``raw.jsonl`` as a
phantom leading turn. ``pipeline/proxy/app.py`` now skips it at capture time, so
*new* runs are clean; this tool retrofits already-captured runs.

Two independent per-file signals (a row need not appear in both):
  proxy.jsonl  — ``num_tool_defs == 0``       (every real turn carries tools)
  raw.jsonl    — ``"sentence-case title"`` in isl_text / isl_new_text

Only **complete** problems are touched: a problem is done once its ``meta.json``
exists (written last, with ``ended_at``) AND none of its files were modified in
the last ``--min-age`` seconds. The problem the run is currently solving has no
``meta.json`` yet, so it is never touched — safe to run against a live run.

Idempotent: a file with no title rows is left alone; a ``.bak`` is written only
when a file is actually modified, and an existing ``.bak`` is never overwritten.

Usage:
    python strip_title_turns.py --save-dir <run_dir> [--dry-run] [--min-age 60]
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

TITLE_MARKER = "sentence-case title"


def _read_rows(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def _is_title_proxy(row: dict) -> bool:
    # Toolless == title request; guard rows from older proxy.jsonl that never
    # captured num_tool_defs (keep them rather than treat missing as 0).
    return "num_tool_defs" in row and int(row.get("num_tool_defs") or 0) == 0


def _is_title_raw(row: dict) -> bool:
    blob = (row.get("isl_text") or "") + (row.get("isl_new_text") or "")
    return TITLE_MARKER in blob


def _recently_touched(problem_dir: Path, min_age: float) -> bool:
    """True if any jsonl was modified within `min_age` seconds (still being
    written). Belt-and-suspenders on top of the meta.json check."""
    now = time.time()
    for f in problem_dir.glob("*.jsonl"):
        if now - f.stat().st_mtime < min_age:
            return True
    return False


def _strip_file(path: Path, is_title, *, dry_run: bool) -> int:
    """Drop title rows from one jsonl. Returns rows removed (0 = untouched)."""
    if not path.exists():
        return 0
    rows = _read_rows(path)
    kept = [r for r in rows if not is_title(r)]
    removed = len(rows) - len(kept)
    if removed == 0 or dry_run:
        return removed
    bak = path.with_suffix(path.suffix + ".bak")
    if not bak.exists():  # preserve the very first original
        shutil.copy(path, bak)
    path.write_text("".join(json.dumps(r) + "\n" for r in kept))
    return removed


def run(save_dir: Path, *, dry_run: bool, min_age: float) -> None:
    telemetry = save_dir / "telemetry"
    done = skipped_running = touched = 0
    for meta_path in sorted(telemetry.glob("*/meta.json")):
        d = meta_path.parent
        iid = d.name
        meta = json.loads(meta_path.read_text())
        if meta.get("ended_at") is None or _recently_touched(d, min_age):
            skipped_running += 1
            continue
        done += 1
        p = _strip_file(path=d / "proxy.jsonl", is_title=_is_title_proxy, dry_run=dry_run)
        r = _strip_file(path=d / "raw.jsonl", is_title=_is_title_raw, dry_run=dry_run)
        if p or r:
            touched += 1
            verb = "would strip" if dry_run else "stripped"
            print(f"  {iid}: {verb} proxy={p} raw={r}")

    # A problem mid-flight has no meta.json yet — count those too.
    in_flight = sum(1 for d in telemetry.glob("*/") if not (d / "meta.json").exists())
    print(
        f"\n{'DRY RUN — ' if dry_run else ''}complete problems scanned={done}, "
        f"modified={touched}, skipped(running/recent)={skipped_running + in_flight}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-dir", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    ap.add_argument(
        "--min-age",
        type=float,
        default=60.0,
        help="skip problems whose files changed within this many seconds "
        "(guards against the in-flight problem)",
    )
    args = ap.parse_args()
    run(args.save_dir, dry_run=args.dry_run, min_age=args.min_age)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
