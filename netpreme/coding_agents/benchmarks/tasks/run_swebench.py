#!/usr/bin/env python3
"""
Run a SWE-bench Verified instance against the local vLLM server.

Measures per-turn context growth and token metrics. Correctness is NOT evaluated.

Usage:
    python3 run_swebench.py                            # first instance
    python3 run_swebench.py --index 5                  # 5th instance
    python3 run_swebench.py --id django__django-11099  # by instance_id
    python3 run_swebench.py --list                     # list first 20 instances
    python3 run_swebench.py --no-clone                 # skip repo setup, run in cwd

Output:
    Per-turn table printed live as claude runs:
      Turn  ISL(in)  OSL(out)  Δ Context  Tool Calls  Text
         1    4,521       312     +4,521           2  Read file...
         2    8,103       287     +3,582           3  Found issue...
        ...

    Final summary: total turns, peak ISL, total tool calls, output tokens.

    The vLLM /metrics endpoint (watch_vllm.py) gives TTFT/ITL/cache metrics
    alongside this — run it in a separate terminal.

Requirements:
    pip install datasets  (for dataset loading)
    claude CLI in PATH
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = SCRIPT_DIR.parent
ENV_FILE = BENCHMARKS_DIR.parent / ".env"
WORKSPACE_ROOT = Path("/tmp/swe_workspaces")
TOOL_CALLS_FILE = Path("/tmp/vllm_tool_calls.json")  # shared with watch_vllm.py

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
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()


# ── dataset ───────────────────────────────────────────────────────────────────
def load_dataset_instances():
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed.", file=sys.stderr)
        print("       pip install datasets", file=sys.stderr)
        sys.exit(1)
    print("Loading SWE-bench Verified dataset...", flush=True)
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    return ds


def get_instance(ds, *, index=None, instance_id=None):
    if instance_id:
        matches = [i for i, row in enumerate(ds) if row["instance_id"] == instance_id]
        if not matches:
            print(f"ERROR: instance_id '{instance_id}' not found.", file=sys.stderr)
            sys.exit(1)
        return ds[matches[0]]
    idx = index if index is not None else 0
    return ds[idx]


# ── workspace ─────────────────────────────────────────────────────────────────
def setup_workspace(instance: dict) -> Path:
    """Clone repo at base_commit into a cached workspace. Returns workspace path."""
    instance_id = instance["instance_id"]
    repo        = instance["repo"]          # e.g. "django/django"
    base_commit = instance["base_commit"]

    workspace = WORKSPACE_ROOT / instance_id
    if workspace.exists():
        print(f"Workspace already exists, resetting to {base_commit[:8]}...")
        subprocess.run(
            ["git", "checkout", "-f", base_commit],
            cwd=workspace, check=True, capture_output=True
        )
    else:
        WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
        clone_url = f"https://github.com/{repo}.git"
        print(f"Cloning {repo} ...")
        subprocess.run(
            ["git", "clone", "--depth=50", clone_url, str(workspace)],
            check=True
        )
        subprocess.run(
            ["git", "fetch", "--depth=50", "origin", base_commit],
            cwd=workspace, check=True, capture_output=True
        )
        subprocess.run(
            ["git", "checkout", base_commit],
            cwd=workspace, check=True, capture_output=True
        )

    print(f"Workspace: {workspace}  (commit {base_commit[:8]})")
    return workspace


# ── stream parser ─────────────────────────────────────────────────────────────
def iter_stream(proc) -> dict:
    """
    Yield parsed JSON objects from claude --output-format stream-json.
    Each line is one JSON event.
    """
    for raw in proc.stdout:
        line = raw.strip()
        if not line:
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            # non-JSON lines (e.g. debug output with --verbose) — skip
            continue


# ── table helpers ─────────────────────────────────────────────────────────────
COL = "{:>5}  {:>9}  {:>9}  {:>10}  {:>10}  {}"
HDR = COL.format("Turn", "ISL(in)", "OSL(out)", "Δ Context", "Tool Calls", "First text")
SEP = "-" * len(HDR)


def fmt_tok(n) -> str:
    return f"{int(n):,}" if n is not None else "—"


def fmt_delta(d) -> str:
    if d is None:
        return "—"
    sign = "+" if d >= 0 else ""
    return f"{sign}{int(d):,}"


def first_text(content: list) -> str:
    """Return the first text snippet from a content array (truncated)."""
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            t = block.get("text", "").replace("\n", " ").strip()
            if t:
                return t[:72]
    return ""


# ── main run ──────────────────────────────────────────────────────────────────
def run_instance(instance: dict, workdir: Path, model: str, base_url: str) -> None:
    problem = instance["problem_statement"]
    instance_id = instance["instance_id"]

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
    print("-" * 72)
    print(problem[:600] + ("..." if len(problem) > 600 else ""))
    print("-" * 72)
    print()
    print(HDR)
    print(SEP)

    env = {
        **os.environ,
        "ANTHROPIC_BASE_URL":            base_url,
        "ANTHROPIC_API_KEY":             "dummy",
        "ANTHROPIC_AUTH_TOKEN":          "dummy",
        "ANTHROPIC_DEFAULT_OPUS_MODEL":  model,
        "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
        "ANTHROPIC_DEFAULT_HAIKU_MODEL": model,
    }

    cmd = [
        "claude",
        "--model", model,
        "--dangerously-skip-permissions",
        "--output-format", "stream-json",
        "-p", problem,
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(workdir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,  # suppress verbose noise; watch_vllm.py has metrics
        text=True,
        bufsize=1,
    )

    turns         = []
    prev_input    = None
    total_tools   = 0

    # Reset sidecar file so watch_vllm.py starts clean for this run
    TOOL_CALLS_FILE.write_text("[]")

    try:
        for event in iter_stream(proc):
            etype = event.get("type")

            if etype == "assistant":
                msg     = event.get("message", {})
                usage   = msg.get("usage", {})
                content = msg.get("content", [])

                isl        = usage.get("input_tokens")
                osl        = usage.get("output_tokens")
                tool_calls = sum(1 for b in content if isinstance(b, dict) and b.get("type") == "tool_use")
                snippet    = first_text(content)

                delta = (isl - prev_input) if (isl is not None and prev_input is not None) else isl
                prev_input = isl
                total_tools += tool_calls

                turn_num = len(turns) + 1
                tool_names = [b.get("name", "") for b in content
                              if isinstance(b, dict) and b.get("type") == "tool_use"]
                turns.append(dict(turn=turn_num, isl=isl, osl=osl, delta=delta, tools=tool_calls))

                # Write sidecar for watch_vllm.py
                try:
                    existing = json.loads(TOOL_CALLS_FILE.read_text())
                    existing.append({"turn": turn_num, "tool_calls": tool_calls, "tool_names": tool_names})
                    TOOL_CALLS_FILE.write_text(json.dumps(existing))
                except Exception:
                    pass

                print(COL.format(
                    turn_num,
                    fmt_tok(isl),
                    fmt_tok(osl),
                    fmt_delta(delta),
                    tool_calls,
                    snippet,
                ))

            elif etype == "result":
                # Final event — may have authoritative usage totals
                final_usage = event.get("usage", {})
                if final_usage:
                    print()
                    print(f"  Final usage  input={fmt_tok(final_usage.get('input_tokens'))}  "
                          f"output={fmt_tok(final_usage.get('output_tokens'))}  "
                          f"cache_read={fmt_tok(final_usage.get('cache_read_input_tokens'))}")

    except KeyboardInterrupt:
        proc.terminate()

    proc.wait()

    # ── summary ──────────────────────────────────────────────────────────────
    print()
    print(SEP)
    print("SUMMARY")
    print(SEP)
    if turns:
        peak_isl   = max(t["isl"] for t in turns if t["isl"] is not None)
        total_osl  = sum(t["osl"] for t in turns if t["osl"] is not None)
        print(f"  Total turns      : {len(turns)}")
        print(f"  Peak ISL (input) : {fmt_tok(peak_isl)} tokens")
        print(f"  Total output     : {fmt_tok(total_osl)} tokens")
        print(f"  Total tool calls : {total_tools}")
        print(f"  Avg tools/turn   : {total_tools/len(turns):.1f}")
        print()
        print("  Per-turn ISL growth:")
        for t in turns:
            bar_len = max(1, int((t["isl"] or 0) / (peak_isl or 1) * 40))
            bar = "█" * bar_len
            print(f"    Turn {t['turn']:>3}  {fmt_tok(t['isl']):>9} tokens  {bar}")
    else:
        print("  No turns recorded.")
    print(SEP)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--index",    type=int,  default=None, help="Dataset row index (default: 0)")
    p.add_argument("--id",       type=str,  default=None, help="instance_id to run")
    p.add_argument("--list",     action="store_true",     help="List first 20 instances and exit")
    p.add_argument("--no-clone", action="store_true",     help="Skip repo setup; run claude in cwd")
    p.add_argument("--workdir",  type=str,  default=None, help="Override working directory for claude")
    p.add_argument("--port",     type=str,  default=None, help="vLLM port (default from .env or 8000)")
    p.add_argument("--model",    type=str,  default=None, help="Model name override")
    args = p.parse_args()

    load_env(ENV_FILE)

    model    = args.model or os.environ.get("MODEL", DEFAULT_MODEL)
    port     = args.port  or os.environ.get("PORT",  DEFAULT_PORT)
    base_url = f"http://localhost:{port}"

    ds       = load_dataset_instances()

    if args.list:
        print(f"{'#':>4}  {'instance_id':<45}  {'repo'}")
        print("-" * 80)
        for i, row in enumerate(ds):
            if i >= 20:
                break
            print(f"{i:>4}  {row['instance_id']:<45}  {row['repo']}")
        return

    instance = get_instance(ds, index=args.index, instance_id=args.id)

    if args.no_clone or args.workdir:
        workdir = Path(args.workdir) if args.workdir else Path.cwd()
    else:
        workdir = setup_workspace(instance)

    run_instance(instance, workdir=workdir, model=model, base_url=base_url)


if __name__ == "__main__":
    main()
