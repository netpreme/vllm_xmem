"""Build the canonical analytic dataset for a run.

Walks runs/<stamp>/per_problem/*.csv + problems.jsonl, joins difficulty
per problem, and writes one structured numpy array to <run-dir>/data.npz.
Every plot script reads from this file rather than re-parsing CSVs.

  import numpy as np
  t = np.load("runs/<stamp>/data.npz")["turns"]
  t.dtype.names    # see schema
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from data import load_per_problem_rows, load_problem_field, num

DTYPE = np.dtype([
    ("instance_id",    "U64"),
    ("difficulty",     "U24"),
    ("turn",           "i4"),
    ("isl",            "i4"),
    ("osl",            "i4"),
    ("isl_new",        "i4"),
    ("isl_cached",     "i4"),
    ("cache_hit_rate", "f4"),
    ("category",       "U10"),
    ("num_tool_calls", "i4"),
    ("ttft_ms",        "i4"),
    ("decode_ms",      "i4"),
    ("itl_ms",         "f4"),
    ("elapsed_ms",     "i4"),
])


def _int(v: float) -> int:
    """Coerce NaN/inf to 0; keep finite ints intact."""
    return int(v) if math.isfinite(v) else 0


def build_records(run_dir: Path) -> np.ndarray:
    """Walk per-problem CSVs in order; emit one structured record per turn."""
    difficulty = load_problem_field(run_dir, "difficulty")
    rows: list[tuple] = []
    for iid, turns in load_per_problem_rows(run_dir).items():
        for i, r in enumerate(turns, start=1):
            rows.append((
                iid,
                difficulty.get(iid) or "",
                i,
                _int(num(r, "isl")),
                _int(num(r, "osl")),
                _int(num(r, "isl_new")),
                _int(num(r, "isl_cached")),
                num(r, "cache_hit_rate"),
                r.get("category") or "",
                _int(num(r, "num_tool_calls")),
                _int(num(r, "ttft_ms")),
                _int(num(r, "decode_ms")),
                num(r, "itl_ms"),
                _int(num(r, "elapsed_ms")),
            ))
    return np.array(rows, dtype=DTYPE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, type=Path)
    args = ap.parse_args()

    turns = build_records(args.run_dir)
    out = args.run_dir / "data.npz"
    np.savez_compressed(out, turns=turns)
    print(f"wrote {out}  ({len(turns):,} turns)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
