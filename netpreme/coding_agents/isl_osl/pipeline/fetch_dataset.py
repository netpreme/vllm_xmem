"""Download a SWE-bench split to a local JSONL file.

Supports an optional difficulty-stratified sample (`--per-bucket N`) so you
can build a balanced 30-per-difficulty mini-set for quick experiments.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Mirror the remap in analysis/data.py: ">4 hours" has too few problems
# (~3) to stand alone, so fold it into "1+ hours" with "1-4 hours".
DIFFICULTY_REMAP = {">4 hours": "1+ hours", "1-4 hours": "1+ hours"}


def stratify(rows: list[dict], field: str, per_bucket: int) -> list[dict]:
    """Take the first `per_bucket` rows from each value of `field`."""
    buckets: dict[str, list[dict]] = {}
    for r in rows:
        b = r.get(field)
        if isinstance(b, list):
            b = ",".join(map(str, b))
        if field == "difficulty":
            b = DIFFICULTY_REMAP.get(b, b)
        buckets.setdefault(b, []).append(r)
    out: list[dict] = []
    for bucket_rows in buckets.values():
        out.extend(bucket_rows[:per_bucket])
    return out


def main() -> int:
    import random
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset",     default="princeton-nlp/SWE-bench_Verified")
    ap.add_argument("--split",       default="test")
    ap.add_argument("--limit",       type=int, default=0,
                    help="take the first N rows in dataset order (0 = no limit)")
    ap.add_argument("--random",      type=int, default=0,
                    help="take a uniformly random sample of N rows")
    ap.add_argument("--seed",        type=int, default=0,
                    help="random seed for --random (default 0 for reproducibility)")
    ap.add_argument("--per-bucket",  type=int, default=0,
                    help="if >0, take N rows per --bucket-by value (stratified)")
    ap.add_argument("--bucket-by",   default="difficulty")
    ap.add_argument("--out",         required=True, type=Path)
    args = ap.parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("install `datasets`: uv pip install datasets", file=sys.stderr)
        return 2

    rows = list(load_dataset(args.dataset, split=args.split))
    if args.per_bucket > 0:
        rows = stratify(rows, args.bucket_by, args.per_bucket)
    elif args.random > 0:
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.random]
    elif args.limit > 0:
        rows = rows[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[fetch] wrote {len(rows)} problems to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
