"""Download SWE-bench Verified to a local JSONL file."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0, help="0 = no limit")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    try:
        from datasets import load_dataset  # type: ignore
    except ImportError:
        print("install `datasets`: uv pip install datasets", file=sys.stderr)
        return 2

    ds = load_dataset(args.dataset, split=args.split)
    rows = list(ds)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[fetch] wrote {len(rows)} problems to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
