#!/usr/bin/env python3
"""
Analyze TTFT benchmark results and compare CPU tier vs chip tier.

Usage:
    python3 analyze_results.py ttft_log.jsonl              # single run summary
    python3 analyze_results.py --compare cpu.jsonl chip.jsonl
    python3 analyze_results.py --log ttft_log.jsonl --tasks results/cpu/tasks.jsonl
"""

import argparse
import json
import statistics
import sys
from pathlib import Path


def load_log(path: str) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def classify(rec: dict) -> str:
    """
    Classify each LLM call by cache tier:
      cold       — no CPU offload activity, first time this context is seen
      gpu_hit    — no CPU load (c2g_bytes == 0), blocks served from GPU
      cpu_load   — c2g_bytes > 0 during this call (CPU→GPU transfer happened)
    We can't distinguish cold from gpu_hit purely from proxy data,
    but GPU→CPU activity helps confirm eviction is happening.
    """
    c2g = rec.get('c2g_bytes', 0) or 0
    g2c = rec.get('g2c_bytes', 0) or 0
    if c2g > 0:
        return 'cpu_load'
    if g2c > 0:
        return 'gpu_evict'   # blocks moving out — GPU under pressure
    return 'gpu_hit'


def stats(values: list[float]) -> dict:
    if not values:
        return {'n': 0}
    sv = sorted(values)
    n  = len(sv)
    return {
        'n':    n,
        'mean': round(statistics.mean(sv), 1),
        'p50':  round(sv[n // 2], 1),
        'p75':  round(sv[int(n * 0.75)], 1),
        'p95':  round(sv[int(n * 0.95)], 1),
        'min':  round(sv[0], 1),
        'max':  round(sv[-1], 1),
    }


def fmt_stats(s: dict) -> str:
    if s.get('n', 0) == 0:
        return 'no data'
    return (f"mean={s['mean']}ms  "
            f"p50={s['p50']}ms  "
            f"p75={s['p75']}ms  "
            f"p95={s['p95']}ms  "
            f"n={s['n']}")


def summarize(records: list[dict], label: str):
    print(f"\n{'═'*60}")
    print(f"  {label}")
    print(f"{'═'*60}")

    # Filter to actual LLM calls (have ttft_ms)
    calls = [r for r in records if r.get('ttft_ms') is not None]
    print(f"  Total LLM calls : {len(calls)}")
    print(f"  Tasks           : {len({r.get('task_id','') for r in calls})}")

    # Total CPU transfer
    total_c2g = sum(r.get('c2g_bytes', 0) or 0 for r in calls)
    total_g2c = sum(r.get('g2c_bytes', 0) or 0 for r in calls)
    if total_g2c > 0 or total_c2g > 0:
        print(f"  GPU→CPU total   : {total_g2c/1e9:.3f} GB")
        print(f"  CPU→GPU total   : {total_c2g/1e9:.3f} GB")

    # Classify calls
    by_class: dict[str, list[float]] = {}
    for r in calls:
        cls = classify(r)
        by_class.setdefault(cls, []).append(r['ttft_ms'])

    print()
    for cls in ('gpu_hit', 'cpu_load', 'gpu_evict'):
        vals = by_class.get(cls, [])
        if vals:
            print(f"  [{cls:10s}]  {fmt_stats(stats(vals))}")

    # Overall
    all_ttft = [r['ttft_ms'] for r in calls]
    print(f"\n  [overall   ]  {fmt_stats(stats(all_ttft))}")

    # Per-task breakdown
    tasks: dict[str, list[float]] = {}
    for r in calls:
        tid = r.get('task_id', 'unknown')
        tasks.setdefault(tid, []).append(r['ttft_ms'])
    if len(tasks) > 1:
        print(f"\n  Per-task mean TTFT (ms):")
        for tid, vals in sorted(tasks.items()):
            print(f"    {tid:50s}  {statistics.mean(vals):.0f}ms  ({len(vals)} calls)")

    return by_class, all_ttft


def compare(log_a: str, log_b: str, label_a: str = 'A', label_b: str = 'B'):
    rec_a = load_log(log_a)
    rec_b = load_log(log_b)

    cls_a, all_a = summarize(rec_a, f"Run {label_a}  ({log_a})")
    cls_b, all_b = summarize(rec_b, f"Run {label_b}  ({log_b})")

    print(f"\n{'═'*60}")
    print(f"  COMPARISON  {label_a} vs {label_b}")
    print(f"{'═'*60}")

    def delta(name: str):
        va = cls_a.get(name, [])
        vb = cls_b.get(name, [])
        if not va or not vb:
            return
        ma = statistics.mean(va)
        mb = statistics.mean(vb)
        diff = mb - ma
        pct  = 100 * diff / ma if ma else 0
        sign = '+' if diff > 0 else ''
        print(f"  [{name:10s}]  "
              f"{label_a} mean={ma:.0f}ms  "
              f"{label_b} mean={mb:.0f}ms  "
              f"delta={sign}{diff:.0f}ms ({sign}{pct:.1f}%)")

    delta('gpu_hit')
    delta('cpu_load')
    delta('gpu_evict')

    if all_a and all_b:
        ma = statistics.mean(all_a)
        mb = statistics.mean(all_b)
        diff = mb - ma
        sign = '+' if diff > 0 else ''
        print(f"\n  [overall   ]  "
              f"{label_a} mean={ma:.0f}ms  "
              f"{label_b} mean={mb:.0f}ms  "
              f"delta={sign}{diff:.0f}ms")

    # The number that matters:
    # CPU→GPU load delta = chip_cpu_load_mean - cpu_cpu_load_mean
    va = cls_a.get('cpu_load', [])
    vb = cls_b.get('cpu_load', [])
    if va and vb:
        diff = statistics.mean(vb) - statistics.mean(va)
        sign = '+' if diff > 0 else ''
        print(f"\n  ★ Tier load overhead delta (cpu_load calls only):")
        print(f"    {label_b} vs {label_a}: {sign}{diff:.0f}ms mean TTFT change")
        if diff < 0:
            print(f"    → {label_b} is {abs(diff):.0f}ms FASTER on cached-prefix loads")
        else:
            print(f"    → {label_b} is {abs(diff):.0f}ms SLOWER on cached-prefix loads")

    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('log',          nargs='?', help='Single TTFT log file')
    parser.add_argument('--compare',    nargs=2, metavar=('LOG_A', 'LOG_B'))
    parser.add_argument('--labels',     nargs=2, default=['cpu', 'chip'],
                        metavar=('LABEL_A', 'LABEL_B'))
    parser.add_argument('--tasks',      help='tasks.jsonl for patch/resolution stats')
    args = parser.parse_args()

    if args.compare:
        compare(args.compare[0], args.compare[1],
                args.labels[0], args.labels[1])
    elif args.log:
        records = load_log(args.log)
        summarize(records, args.log)
        if args.tasks and Path(args.tasks).exists():
            tasks = load_log(args.tasks)
            patched  = sum(1 for t in tasks if t.get('patched'))
            total    = sum(1 for t in tasks if t.get('status') == 'completed')
            print(f"\n  Task outcomes: {patched}/{total} tasks produced a patch")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
