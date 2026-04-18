#!/usr/bin/env python3
"""
Run SWE-bench Verified instances against the local vLLM server.

Measures per-turn context growth and token metrics. Correctness is NOT evaluated.

Usage:
    python3 run_swebench.py                                   # first instance (index 0)
    python3 run_swebench.py --index 5                         # 5th instance
    python3 run_swebench.py --id django__django-11099         # by instance_id
    python3 run_swebench.py --list                            # list first 20 instances
    python3 run_swebench.py --all                             # run every instance sequentially
    python3 run_swebench.py --all --start 10 --end 50         # slice of the dataset
    python3 run_swebench.py --all --skip-done                 # skip already-completed instances
    python3 run_swebench.py --all --start 5 --skip-done       # resume from index 5, skip done
    python3 run_swebench.py --no-clone                        # skip repo setup, run in cwd

Results saved per run to:
    tasks/results/<instance_id>_<timestamp>/
        metrics.json   — instance info, per-turn ISL/OSL/tools, summary, final usage
        patch.diff     — git diff of changes Claude made in the workspace

Run watch_vllm.py in a separate terminal for TTFT/ITL/cache/offload metrics.

Notes:
    - Requires the vLLM server to be running (./start_vllm_server.sh)
    - OSL per turn is scraped from vLLM /metrics (same method as watch_vllm.py)
    - claude --verbose emits 2 assistant events per request (text + tool_use);
      events with the same input_tokens are merged into one turn row
"""

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR      = Path(__file__).resolve().parent
BENCHMARKS_DIR  = SCRIPT_DIR.parent
ENV_FILE        = BENCHMARKS_DIR.parent / ".env"
WORKSPACE_ROOT  = Path("/tmp/swe_workspaces")
TOOL_CALLS_FILE = Path("/tmp/vllm_tool_calls.json")   # shared with watch_vllm.py
RESULTS_DIR     = SCRIPT_DIR / "results"

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL = "qwen/qwen3-coder-30b-a3b-instruct-fp8"
DEFAULT_PORT  = "8000"


# ── .env loader ───────────────────────────────────────────────────────────────
def load_env(path: Path) -> None:
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()


# ── vLLM metrics scraper (exact copy of watch_vllm.py) ───────────────────────
_LINE_RE = re.compile(
    r'^([a-zA-Z_:][a-zA-Z0-9_:]*)'
    r'(?:\{([^}]*)\})?'
    r'\s+([+-]?(?:[0-9]*\.)?[0-9]+(?:[eE][+-]?[0-9]+)?|NaN|[+-]?Inf)'
)
_LABEL_RE = re.compile(r'(\w+)="([^"]*)"')
Scraped = dict[str, list[tuple[dict[str, str], float]]]


def scrape(url: str) -> Scraped:
    result: Scraped = {}
    try:
        r = requests.get(f"{url}/metrics", timeout=2)
        for line in r.text.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            m = _LINE_RE.match(line)
            if not m:
                continue
            name, labels_str, value_str = m.groups()
            if value_str in ("NaN", "Inf", "+Inf", "-Inf"):
                continue
            try:
                value = float(value_str)
            except ValueError:
                continue
            labels: dict[str, str] = {}
            if labels_str:
                for kv in _LABEL_RE.finditer(labels_str):
                    labels[kv.group(1)] = kv.group(2)
            result.setdefault(name, []).append((labels, value))
    except Exception:
        pass
    return result


def sum_metric(s: Scraped, name: str) -> Optional[float]:
    series = s.get(name)
    if not series:
        return None
    return sum(v for _, v in series)


def _delta_avg(new_s, new_c, old_s, old_c) -> Optional[float]:
    if None in (new_s, new_c, old_s, old_c):
        return None
    d = new_c - old_c
    return (new_s - old_s) / d if d > 0 else None


# ── dataset ───────────────────────────────────────────────────────────────────
def load_dataset():
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed.  pip install datasets", file=sys.stderr)
        sys.exit(1)
    print("Loading SWE-bench Verified dataset...", flush=True)
    return load_dataset("princeton-nlp/SWE-bench_Verified", split="test")


def get_instance(ds, *, index=None, instance_id=None):
    if instance_id:
        matches = [i for i, row in enumerate(ds) if row["instance_id"] == instance_id]
        if not matches:
            print(f"ERROR: instance_id '{instance_id}' not found.", file=sys.stderr)
            sys.exit(1)
        return ds[matches[0]]
    return ds[index if index is not None else 0]


# ── workspace ─────────────────────────────────────────────────────────────────
def setup_workspace(instance: dict) -> Path:
    """Clone repo at base_commit into a cached workspace. Returns workspace path."""
    instance_id = instance["instance_id"]
    repo        = instance["repo"]
    base_commit = instance["base_commit"]
    workspace   = WORKSPACE_ROOT / instance_id

    if workspace.exists():
        print(f"Workspace exists, resetting to {base_commit[:8]}...")
        subprocess.run(["git", "checkout", "-f", base_commit],
                       cwd=workspace, check=True, capture_output=True)
        subprocess.run(["git", "clean", "-fd"],
                       cwd=workspace, check=True, capture_output=True)
    else:
        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        print(f"Cloning {repo} ...")
        subprocess.run(["git", "clone", "--depth=50",
                        f"https://github.com/{repo}.git", str(workspace)], check=True)
        subprocess.run(["git", "fetch", "--depth=50", "origin", base_commit],
                       cwd=workspace, check=True, capture_output=True)
        subprocess.run(["git", "checkout", base_commit],
                       cwd=workspace, check=True, capture_output=True)

    print(f"Workspace: {workspace}  (commit {base_commit[:8]})")
    return workspace


# ── stream parser ─────────────────────────────────────────────────────────────
def iter_stream(proc):
    for raw in proc.stdout:
        line = raw.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue


# ── table helpers ─────────────────────────────────────────────────────────────
COL = "{:>5}  {:>9}  {:>9}  {:>10}  {:>10}  {}"
HDR = COL.format("Turn", "ISL(in)", "OSL(out)", "Δ Context", "Tools", "First text")
SEP = "─" * len(HDR)


def fmt_tok(n) -> str:
    return f"{int(n):,}" if n is not None else "—"


def fmt_delta(d) -> str:
    if d is None:
        return "—"
    return f"+{int(d):,}" if d >= 0 else f"{int(d):,}"


def first_text(content: list) -> str:
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            t = block.get("text", "").replace("\n", " ").strip()
            if t:
                return t[:72]
    return ""


# ── save results ──────────────────────────────────────────────────────────────
def save_results(instance: dict, workdir: Path, model: str, base_url: str,
                 turns: list, final_usage: dict, started_at: str) -> Path:
    instance_id = instance["instance_id"]
    ts          = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir     = RESULTS_DIR / f"{instance_id}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    patch = ""
    try:
        r = subprocess.run(["git", "diff"], cwd=str(workdir), capture_output=True, text=True)
        patch = r.stdout
        if patch:
            (out_dir / "patch.diff").write_text(patch)
    except Exception:
        pass

    valid_isl   = [t["isl"] for t in turns if t.get("isl") is not None]
    total_tools = sum(len(t.get("tool_names", [])) for t in turns)
    peak_isl    = max(valid_isl) if valid_isl else 0
    valid_osl   = [t["osl"] for t in turns if t.get("osl") is not None]

    data = {
        "instance_id":       instance_id,
        "repo":              instance["repo"],
        "base_commit":       instance["base_commit"],
        "difficulty":        instance.get("difficulty"),
        "problem_statement": instance.get("problem_statement", ""),
        "model":             model,
        "server":       base_url,
        "workdir":      str(workdir),
        "started_at":   started_at,
        "finished_at":  datetime.now().isoformat(),
        "turns":        turns,
        "summary": {
            "total_turns":         len(turns),
            "peak_isl":            peak_isl,
            "total_osl":           sum(valid_osl),
            "total_output_tokens": final_usage.get("output_tokens"),
            "total_tool_calls":    total_tools,
            "avg_tools_per_turn":  round(total_tools / len(turns), 2) if turns else 0,
            "patch_lines":         len(patch.splitlines()) if patch else 0,
        },
        "final_usage": final_usage,
    }

    (out_dir / "metrics.json").write_text(json.dumps(data, indent=2))
    print(f"  Results → {out_dir}/")
    return out_dir


# ── run one instance ──────────────────────────────────────────────────────────
def run_instance(instance: dict, workdir: Path, model: str, base_url: str) -> dict:
    problem     = instance["problem_statement"]
    instance_id = instance["instance_id"]
    started_at  = datetime.now().isoformat()

    print()
    print("═" * 72)
    print(f"  Instance : {instance_id}")
    print(f"  Repo     : {instance['repo']}  @{instance['base_commit'][:8]}")
    print(f"  Model    : {model}")
    print(f"  Server   : {base_url}")
    print(f"  Workdir  : {workdir}")
    print("═" * 72)
    print()
    print("Problem statement:")
    print("─" * 72)
    print(problem[:600] + ("..." if len(problem) > 600 else ""))
    print("─" * 72)
    print()
    print(HDR)
    print(SEP)

    env = {
        **os.environ,
        "ANTHROPIC_BASE_URL":             base_url,
        "ANTHROPIC_API_KEY":              "dummy",
        "ANTHROPIC_AUTH_TOKEN":           "dummy",
        "ANTHROPIC_DEFAULT_OPUS_MODEL":   model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL":  model,
    }

    proc = subprocess.Popen(
        [
            "claude",
            "--model", model,
            "--dangerously-skip-permissions",
            "--verbose",
            "--output-format", "stream-json",
            "--max-turns", "100",
            "-p", problem,
        ],
        cwd=str(workdir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=sys.stderr,
        text=True,
        bufsize=1,
    )

    turns       = []
    final_usage = {}

    # ── Background OSL poller ─────────────────────────────────────────────────
    # With --verbose, output_tokens=0 in individual assistant events.
    # A background thread polls /metrics every 100 ms (same as watch_vllm.py),
    # detects each new completed request via delta(sum)/delta(count), and pushes
    # the OSL into a queue.  flush_pending() pops values in order.
    osl_queue = queue.Queue()
    _stop     = threading.Event()

    def _poll_osl():
        snap0    = scrape(base_url)
        prev_sum = sum_metric(snap0, "vllm:request_generation_tokens_sum")
        prev_cnt = sum_metric(snap0, "vllm:request_generation_tokens_count")
        while not _stop.is_set():
            time.sleep(0.1)
            snap    = scrape(base_url)
            new_sum = sum_metric(snap, "vllm:request_generation_tokens_sum")
            new_cnt = sum_metric(snap, "vllm:request_generation_tokens_count")
            osl     = _delta_avg(new_sum, new_cnt, prev_sum, prev_cnt)
            if osl is not None:
                osl_queue.put(round(osl))
                prev_sum = new_sum
                prev_cnt = new_cnt

    threading.Thread(target=_poll_osl, daemon=True).start()

    # ── Buffered turn state ───────────────────────────────────────────────────
    # --verbose emits 2 assistant events per vLLM request when the response has
    # both a text block and a tool_use block, both carrying the same input_tokens.
    # Buffer until ISL changes, then flush one canonical row per unique ISL.
    pending_isl        = None
    pending_turn_num   = 0
    pending_tool_calls = 0
    pending_tool_names: list = []
    pending_snippet    = ""

    TOOL_CALLS_FILE.write_text("[]")

    def _write_sidecar():
        """Write/update the sidecar entry for the current pending turn."""
        try:
            existing = json.loads(TOOL_CALLS_FILE.read_text())
            # Update if entry already exists, otherwise append
            idx = next((i for i, e in enumerate(existing)
                        if e.get("turn") == pending_turn_num), None)
            entry = {"turn": pending_turn_num, "tool_calls": pending_tool_calls,
                     "tool_names": pending_tool_names[:]}
            if idx is not None:
                existing[idx] = entry
            else:
                existing.append(entry)
            TOOL_CALLS_FILE.write_text(json.dumps(existing))
        except Exception:
            pass

    def flush_pending():
        nonlocal pending_isl, pending_turn_num, pending_tool_calls
        nonlocal pending_tool_names, pending_snippet
        if pending_isl is None:
            return

        prev_isl = turns[-1]["isl"] if turns else None
        delta    = (pending_isl - prev_isl) if prev_isl is not None else pending_isl

        types: list[str] = []
        if pending_snippet:
            types.append("text")
        types.extend(["tool"] * pending_tool_calls)

        try:
            osl = osl_queue.get(timeout=2.0)
        except queue.Empty:
            osl = None

        turns.append({
            "turn":       pending_turn_num,
            "isl":        pending_isl,
            "osl":        osl,
            "delta":      delta,
            "types":      types,
            "tool_names": pending_tool_names[:],
            "snippet":    pending_snippet,
        })

        print(COL.format(pending_turn_num, fmt_tok(pending_isl), fmt_tok(osl),
                         fmt_delta(delta), pending_tool_calls, pending_snippet))

        pending_isl        = None
        pending_turn_num   = 0
        pending_tool_calls = 0
        pending_tool_names = []
        pending_snippet    = ""

    # ── Event loop ────────────────────────────────────────────────────────────
    try:
        for event in iter_stream(proc):
            etype = event.get("type")

            if etype == "assistant":
                msg       = event.get("message", {})
                isl       = msg.get("usage", {}).get("input_tokens")
                content   = msg.get("content", [])
                tool_uses = [b for b in content
                             if isinstance(b, dict) and b.get("type") == "tool_use"]

                if isl != pending_isl:
                    flush_pending()
                    pending_isl      = isl
                    pending_turn_num = len(turns) + 1

                pending_tool_calls += len(tool_uses)
                pending_tool_names.extend(b.get("name", "") for b in tool_uses)
                if tool_uses:
                    _write_sidecar()   # write immediately so watch_vllm.py sees it in time
                if not pending_snippet:
                    pending_snippet = first_text(content)

            elif etype == "result":
                flush_pending()
                final_usage = event.get("usage", {})
                if final_usage:
                    print()
                    print(f"  Final usage  "
                          f"input={fmt_tok(final_usage.get('input_tokens'))}  "
                          f"output={fmt_tok(final_usage.get('output_tokens'))}  "
                          f"cache_read={fmt_tok(final_usage.get('cache_read_input_tokens'))}")

    except KeyboardInterrupt:
        flush_pending()
        proc.terminate()
        raise
    finally:
        _stop.set()

    proc.wait()

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("SUMMARY")
    print(SEP)
    if turns:
        peak_isl    = max(t["isl"] for t in turns if t["isl"] is not None)
        total_osl   = sum(t["osl"] for t in turns if t["osl"] is not None)
        total_tools = sum(len(t.get("tool_names", [])) for t in turns)
        print(f"  Total turns      : {len(turns)}")
        print(f"  Peak ISL (input) : {fmt_tok(peak_isl)} tokens")
        print(f"  Total OSL        : {fmt_tok(total_osl)} tokens")
        print(f"  Total tool calls : {total_tools}")
        print(f"  Avg tools/turn   : {total_tools / len(turns):.1f}")
        print()
        print("  Per-turn ISL growth:")
        for t in turns:
            bar_len = max(1, int((t["isl"] or 0) / (peak_isl or 1) * 40))
            print(f"    Turn {t['turn']:>3}  {fmt_tok(t['isl']):>9} tokens  {'█' * bar_len}")
    else:
        print("  No turns recorded.")
    print(SEP)

    save_results(instance, workdir, model, base_url, turns, final_usage, started_at)

    return {
        "instance_id": instance_id,
        "total_turns": len(turns),
        "peak_isl":    max((t["isl"] for t in turns if t["isl"] is not None), default=0),
        "total_tools": total_tools,
        "status":      "ok",
    }


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--index", type=int,          help="Dataset row index (default: 0)")
    g.add_argument("--id",    type=str,          help="instance_id to run")
    g.add_argument("--all",   action="store_true", help="Run all instances sequentially")
    g.add_argument("--list",  action="store_true", help="List first 20 instances and exit")
    p.add_argument("--start",     type=int, default=0,    help="Start index for --all (default: 0)")
    p.add_argument("--end",       type=int, default=None, help="End index (exclusive) for --all")
    p.add_argument("--skip-done",  action="store_true",    help="Skip instances that already have results")
    p.add_argument("--no-hard-first", action="store_true",  help="Disable default hardest-first ordering")
    p.add_argument("--difficulty", type=str, nargs="+",
                   help="Filter by difficulty level(s): easy, medium, hard, vhard  "
                        "(maps to <15 min fix, 15 min - 1 hour, 1-4 hours, >4 hours)")
    p.add_argument("--no-clone",   action="store_true",    help="Skip repo setup; run claude in cwd")
    p.add_argument("--workdir",   type=str, default=None, help="Override working directory for claude")
    p.add_argument("--port",      type=str, default=None, help="vLLM port (default from .env or 8000)")
    p.add_argument("--model",     type=str, default=None, help="Model name override")
    args = p.parse_args()

    load_env(ENV_FILE)

    model    = args.model or os.environ.get("MODEL", DEFAULT_MODEL)
    port     = args.port  or os.environ.get("PORT",  DEFAULT_PORT)
    base_url = f"http://localhost:{port}"

    ds = load_dataset()

    if args.list:
        print(f"{'#':>4}  {'instance_id':<45}  {'repo'}")
        print("─" * 80)
        for i, row in enumerate(ds):
            if i >= 20:
                break
            print(f"{i:>4}  {row['instance_id']:<45}  {row['repo']}")
        return

    def _workdir(instance):
        if args.no_clone or args.workdir:
            return Path(args.workdir) if args.workdir else Path.cwd()
        return setup_workspace(instance)

    DIFFICULTY_MAP = {
        "easy":   "<15 min fix",
        "medium": "15 min - 1 hour",
        "hard":   "1-4 hours",
        "vhard":  ">4 hours",
    }

    if args.all:
        DIFFICULTY_ORDER = [">4 hours", "1-4 hours", "15 min - 1 hour", "<15 min fix", "unknown"]

        total   = len(ds)
        end     = args.end if args.end is not None else total
        indices = list(range(args.start, min(end, total)))

        if args.difficulty:
            allowed = {DIFFICULTY_MAP.get(d, d) for d in args.difficulty}
            indices = [i for i in indices if (ds[i].get("difficulty") or "unknown") in allowed]
            print(f"Filtering to difficulty levels: {allowed}  ({len(indices)} instances)")

        if not args.no_hard_first:
            # Primary: hardest difficulty first; secondary: longest problem statement first
            indices.sort(key=lambda i: (
                DIFFICULTY_ORDER.index(
                    ds[i].get("difficulty") or "unknown"
                    if (ds[i].get("difficulty") or "unknown") in DIFFICULTY_ORDER
                    else len(DIFFICULTY_ORDER)
                ),
                -len(ds[i].get("problem_statement") or ""),
            ))

        n       = len(indices)
        results = []

        order_note = "" if args.no_hard_first else " (hardest first)"
        print(f"\nRunning {n} instances{order_note} against {base_url}")
        print(f"Results will be saved to {RESULTS_DIR}/\n")

        for pos, idx in enumerate(indices, 1):
            instance = ds[idx]
            iid      = instance["instance_id"]

            if args.skip_done and list(RESULTS_DIR.glob(f"{iid}_*/metrics.json")):
                print(f"[{pos}/{n}] Skipping {iid} (results already exist)")
                continue

            print(f"\n[{pos}/{n}] {iid}")
            try:
                summary = run_instance(instance, _workdir(instance), model, base_url)
                results.append(summary)
            except KeyboardInterrupt:
                print("\nInterrupted.")
                break
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({"instance_id": iid, "status": "error", "error": str(e)})

        ok  = [r for r in results if r.get("status") == "ok"]
        err = [r for r in results if r.get("status") == "error"]
        print("\n" + "═" * 72)
        print(f"  Completed {len(results)}/{n} instances  "
              f"({len(ok)} ok, {len(err)} errors)")
        if ok:
            print(f"  Avg turns    : {sum(r['total_turns'] for r in ok) / len(ok):.1f}")
            print(f"  Avg peak ISL : {sum(r['peak_isl'] for r in ok) / len(ok):,.0f} tokens")
        if err:
            print("  Errors:")
            for r in err:
                print(f"    {r['instance_id']}: {r.get('error', '?')}")
        print("═" * 72)

    else:
        instance = get_instance(ds, index=args.index, instance_id=args.id)
        run_instance(instance, _workdir(instance), model, base_url)


if __name__ == "__main__":
    main()
