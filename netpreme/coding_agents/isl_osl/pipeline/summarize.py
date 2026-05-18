"""
Aggregate per-turn ISL/OSL records into a summary report grouped by the
four categories (text_only, tool_only, mixed, empty).

Inputs : usage.jsonl produced by run_one.py
Outputs: summary.json (machine-readable) + plain-text report on stdout.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


CATEGORIES = ("text_only", "tool_only", "mixed", "empty")


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return float(s[lo])
    return float(s[lo] + (s[hi] - s[lo]) * (k - lo))


def stat_block(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    fv = [float(v) for v in values]
    return {
        "n": len(fv),
        "mean": round(statistics.fmean(fv), 1),
        "stdev": round(statistics.pstdev(fv), 1) if len(fv) > 1 else 0.0,
        "min": int(min(fv)),
        "p50": int(percentile(fv, 0.5)),
        "p90": int(percentile(fv, 0.9)),
        "p99": int(percentile(fv, 0.99)),
        "max": int(max(fv)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--usage", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    by_cat: dict[str, dict[str, list[int]]] = {
        c: {"isl": [], "osl": [], "isl_new": [], "isl_cached": []}
        for c in CATEGORIES
    }
    instances: set[str] = set()
    total_rows = 0
    sum_isl = 0
    sum_isl_new = 0
    sum_isl_cached = 0

    with args.usage.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            cat = row.get("category", "empty")
            if cat not in CATEGORIES:
                cat = "empty"
            isl = int(row.get("isl") or 0)
            osl = int(row.get("osl") or 0)
            isl_new = int(row.get("isl_new") or 0)
            isl_cached = int(row.get("isl_cached") or 0)
            by_cat[cat]["isl"].append(isl)
            by_cat[cat]["osl"].append(osl)
            by_cat[cat]["isl_new"].append(isl_new)
            by_cat[cat]["isl_cached"].append(isl_cached)
            sum_isl += isl
            sum_isl_new += isl_new
            sum_isl_cached += isl_cached
            if row.get("instance_id"):
                instances.add(row["instance_id"])
            total_rows += 1

    summary = {
        "totals": {
            "rows": total_rows,
            "instances": len(instances),
            "isl_total": sum_isl,
            "isl_new_total": sum_isl_new,
            "isl_cached_total": sum_isl_cached,
            "overall_cache_hit_rate": round(sum_isl_cached / sum_isl, 4) if sum_isl else 0.0,
        },
        "by_category": {},
        "all": {
            "isl": stat_block([v for c in CATEGORIES for v in by_cat[c]["isl"]]),
            "osl": stat_block([v for c in CATEGORIES for v in by_cat[c]["osl"]]),
            "isl_new": stat_block([v for c in CATEGORIES for v in by_cat[c]["isl_new"]]),
            "isl_cached": stat_block([v for c in CATEGORIES for v in by_cat[c]["isl_cached"]]),
        },
    }
    for cat in CATEGORIES:
        b = by_cat[cat]
        s_isl = sum(b["isl"]) or 0
        s_cached = sum(b["isl_cached"]) or 0
        summary["by_category"][cat] = {
            "count": len(b["isl"]),
            "isl": stat_block(b["isl"]),
            "osl": stat_block(b["osl"]),
            "isl_new": stat_block(b["isl_new"]),
            "isl_cached": stat_block(b["isl_cached"]),
            "cache_hit_rate": round(s_cached / s_isl, 4) if s_isl else 0.0,
        }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2))

    # Plain-text report
    t = summary["totals"]
    print(f"problems   : {t['instances']}")
    print(f"turns      : {t['rows']}")
    print(f"isl total  : {t['isl_total']}  (new: {t['isl_new_total']}  cached: {t['isl_cached_total']})")
    print(f"overall cache hit rate : {t['overall_cache_hit_rate']:.2%}")
    print()
    hdr = f"{'category':<12}{'n':>6}{'isl p50':>9}{'isl p90':>9}{'isl p99':>9}{'osl p50':>9}{'osl p90':>9}{'osl p99':>9}{'new p50':>9}{'cached p50':>11}{'hit rate':>10}"
    print(hdr)
    for cat in CATEGORIES:
        b = summary["by_category"][cat]
        print(
            f"{cat:<12}{b['count']:>6}"
            f"{b['isl'].get('p50', 0):>9}{b['isl'].get('p90', 0):>9}{b['isl'].get('p99', 0):>9}"
            f"{b['osl'].get('p50', 0):>9}{b['osl'].get('p90', 0):>9}{b['osl'].get('p99', 0):>9}"
            f"{b['isl_new'].get('p50', 0):>9}{b['isl_cached'].get('p50', 0):>11}"
            f"{b['cache_hit_rate']*100:>9.1f}%"
        )
    print()
    print(f"summary written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
