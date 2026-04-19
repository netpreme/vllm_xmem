#!/usr/bin/env python3
"""
Compare benchmark results: hybrid-cpu vs hybrid-mtier at the same concurrency.

Usage:
    python3 benchmark/compare_results.py                        # latest pair
    python3 benchmark/compare_results.py --concurrency 12
    python3 benchmark/compare_results.py --cpu results_benchmarks/bench_hybrid-cpu_... \
                                         --mtier results_benchmarks/bench_hybrid-mtier_...
"""
import argparse
import csv
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results_benchmarks"

BLUE   = "\033[94m"
ORANGE = "\033[38;5;208m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"


def load_summary(path: Path) -> dict:
    csv_path = path / "summary.csv"
    if not csv_path.exists():
        return {}
    rows = list(csv.DictReader(open(csv_path)))
    return {int(r["concurrency"]): r for r in rows}


def latest_dirs(setup: str) -> list[Path]:
    dirs = sorted(RESULTS_DIR.glob(f"bench_{setup}_*"), key=lambda p: p.name)
    return [d for d in dirs if (d / "summary.csv").exists()]


def fmt(val, suffix="ms", decimals=0):
    if val is None or val == "" or val == "None":
        return "    n/a"
    try:
        f = float(val)
        if decimals == 0:
            return f"{f:>6.0f}{suffix}"
        return f"{f:>7.{decimals}f}{suffix}"
    except (ValueError, TypeError):
        return "    n/a"


def speedup_color(x: float) -> str:
    if x >= 2.0:
        return GREEN
    if x >= 1.2:
        return YELLOW
    return RESET


def compare_row(label: str, cpu_val, mt_val, suffix="ms", higher_is_better=False):
    cpu_str = fmt(cpu_val, suffix)
    mt_str  = fmt(mt_val, suffix)
    try:
        c = float(cpu_val)
        m = float(mt_val)
        if higher_is_better:
            ratio = m / c if c > 0 else None
            arrow = "↑" if ratio and ratio > 1.05 else ("↓" if ratio and ratio < 0.95 else "~")
        else:
            ratio = c / m if m > 0 else None
            arrow = "↑" if ratio and ratio > 1.05 else ("↓" if ratio and ratio < 0.95 else "~")
        if ratio:
            color = speedup_color(ratio) if arrow == "↑" else (YELLOW if arrow == "~" else RESET)
            ratio_str = f"{color}{ratio:.2f}×{RESET}"
        else:
            ratio_str = "  n/a"
    except (TypeError, ValueError):
        ratio_str = "  n/a"
    print(f"  {label:<22}  {BLUE}{cpu_str}{RESET}   {ORANGE}{mt_str}{RESET}   {ratio_str}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cpu",          type=Path, default=None)
    p.add_argument("--mtier",        type=Path, default=None)
    p.add_argument("--concurrency",  type=int,  default=12)
    args = p.parse_args()

    if args.cpu and args.mtier:
        cpu_data   = load_summary(args.cpu).get(args.concurrency)
        mtier_data = load_summary(args.mtier).get(args.concurrency)
        cpu_dir    = args.cpu
        mt_dir     = args.mtier
    else:
        cpu_dirs   = latest_dirs("hybrid-cpu")
        mt_dirs    = latest_dirs("hybrid-mtier")
        if not cpu_dirs or not mt_dirs:
            print("ERROR: no benchmark results found in", RESULTS_DIR, file=sys.stderr)
            sys.exit(1)
        cpu_dir    = cpu_dirs[-1]
        mt_dir     = mt_dirs[-1]
        cpu_data   = load_summary(cpu_dir).get(args.concurrency)
        mtier_data = load_summary(mt_dir).get(args.concurrency)

    if not cpu_data:
        print(f"ERROR: no c={args.concurrency} row in {cpu_dir}/summary.csv", file=sys.stderr)
        sys.exit(1)
    if not mtier_data:
        print(f"ERROR: no c={args.concurrency} row in {mt_dir}/summary.csv", file=sys.stderr)
        sys.exit(1)

    c = args.concurrency
    print(f"\n{BOLD}Benchmark comparison — concurrency={c}{RESET}")
    print(f"  {BLUE}CPU  : {cpu_dir.name}{RESET}")
    print(f"  {ORANGE}MTier: {mt_dir.name}{RESET}\n")

    print(f"  {'Metric':<22}  {'hybrid-cpu':>9}   {'hybrid-mtier':>9}   MTier speedup")
    print(f"  {'──────':<22}  {'─────────':>9}   {'────────────':>9}   ─────────────")

    print(f"\n  {BOLD}TTFT (Prometheus histogram){RESET}")
    compare_row("TTFT p50",  cpu_data.get("ttft_p50"),  mtier_data.get("ttft_p50"))
    compare_row("TTFT p90",  cpu_data.get("ttft_p90"),  mtier_data.get("ttft_p90"))
    compare_row("TTFT p95",  cpu_data.get("ttft_p95"),  mtier_data.get("ttft_p95"))
    compare_row("TTFT p99",  cpu_data.get("ttft_p99"),  mtier_data.get("ttft_p99"))

    print(f"\n  {BOLD}E2E latency (Prometheus histogram){RESET}")
    compare_row("E2E p50",   cpu_data.get("e2e_p50"),   mtier_data.get("e2e_p50"))
    compare_row("E2E p90",   cpu_data.get("e2e_p90"),   mtier_data.get("e2e_p90"))

    print(f"\n  {BOLD}Turn E2E (benchmark timing){RESET}")
    compare_row("Turn E2E p50", cpu_data.get("turn_e2e_p50"), mtier_data.get("turn_e2e_p50"))
    compare_row("Turn E2E p90", cpu_data.get("turn_e2e_p90"), mtier_data.get("turn_e2e_p90"))

    print(f"\n  {BOLD}Cache hit rates{RESET}")
    compare_row("GPU hit %", cpu_data.get("gpu_hit_pct"),  mtier_data.get("gpu_hit_pct"),
                suffix="%", higher_is_better=True)
    compare_row("CPU hit %", cpu_data.get("cpu_hit_pct"),  mtier_data.get("cpu_hit_pct"),
                suffix="%", higher_is_better=True)
    compare_row("KV used %", cpu_data.get("kv_used_pct"),  mtier_data.get("kv_used_pct"),
                suffix="%", higher_is_better=True)

    print(f"\n  {BOLD}Throughput{RESET}")
    compare_row("Output tok/s", cpu_data.get("output_tok_per_sec"),
                mtier_data.get("output_tok_per_sec"), suffix="", higher_is_better=True)
    compare_row("Tasks ok",     cpu_data.get("n_ok"),     mtier_data.get("n_ok"),
                suffix="", higher_is_better=True)
    compare_row("Turns total",  cpu_data.get("n_turns_total"), mtier_data.get("n_turns_total"),
                suffix="", higher_is_better=True)
    print()


if __name__ == "__main__":
    main()
