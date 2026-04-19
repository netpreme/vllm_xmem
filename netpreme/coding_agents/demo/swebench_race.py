#!/usr/bin/env python3
"""
SWE-bench race: run one task simultaneously on hybrid-cpu vs hybrid-mtier.

Runs the same SWE-bench instance on both servers in parallel and measures
wall time per turn, showing which server is faster in real time.

Note: outputs are NOT deterministic (agent paths diverge after turn 1).
The comparison is about SPEED, not solution quality.

Usage:
    python3 demo/swebench_race.py                     # first instance
    python3 demo/swebench_race.py --index 5           # 5th instance
    python3 demo/swebench_race.py --id sympy__sympy-1234
    python3 demo/swebench_race.py --list              # list first 20
    python3 demo/swebench_race.py --cpu-port 8001 --mtier-port 8002
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from datetime import datetime

# ── Colors ───────────────────────────────────────────────────────────────────
BLUE   = "\033[94m"
ORANGE = "\033[38;5;208m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"

MODEL          = "qwen/qwen3-coder-30b-a3b-instruct-fp8"
WORKSPACE_ROOT = Path("/tmp/swe_race")


# ── Dataset ───────────────────────────────────────────────────────────────────
def load_instances():
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets", file=sys.stderr)
        sys.exit(1)
    print(f"{DIM}Loading SWE-bench Verified...{RESET}", end="", flush=True)
    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test",
                      trust_remote_code=True)
    print(f"\r{DIM}Loaded {len(ds)} instances.       {RESET}", flush=True)
    return ds


def pick_instance(ds, instance_id=None, index=0):
    if instance_id:
        for item in ds:
            if item["instance_id"] == instance_id:
                return item
        print(f"ERROR: {instance_id} not found", file=sys.stderr)
        sys.exit(1)
    return ds[index]


# ── Workspace ─────────────────────────────────────────────────────────────────
def setup_workspace(instance: dict, tag: str) -> Path:
    repo        = instance["repo"]
    base_commit = instance["base_commit"]
    iid         = instance["instance_id"]
    workdir     = WORKSPACE_ROOT / tag / iid
    workdir.mkdir(parents=True, exist_ok=True)
    repo_url = f"https://github.com/{repo}.git"

    if not (workdir / ".git").exists():
        print(f"  [{tag}] Cloning {repo} ...", flush=True)
        subprocess.run(
            ["git", "clone", "--depth=50", repo_url, str(workdir)],
            capture_output=True, check=False,
        )
    subprocess.run(
        ["git", "-C", str(workdir), "checkout", "-f", base_commit],
        capture_output=True, check=False,
    )
    subprocess.run(
        ["git", "-C", str(workdir), "clean", "-fd"],
        capture_output=True, check=False,
    )
    return workdir


# ── Runner ────────────────────────────────────────────────────────────────────
class Runner(threading.Thread):
    def __init__(self, label: str, color: str, port: int,
                 instance: dict, workdir: Path):
        super().__init__(daemon=True)
        self.label   = label
        self.color   = color
        self.port    = port
        self.instance = instance
        self.workdir  = workdir
        self.turns: list[dict] = []
        self.wall_time: float | None = None
        self.error: str | None = None
        self._lock   = threading.Lock()

    # Called from main thread for display
    def snapshot(self):
        with self._lock:
            return list(self.turns), self.wall_time, self.error

    def run(self):
        base_url = f"http://localhost:{self.port}"
        env = {
            **os.environ,
            "ANTHROPIC_BASE_URL":             base_url,
            "ANTHROPIC_API_KEY":              "dummy",
            "ANTHROPIC_AUTH_TOKEN":           "dummy",
            "ANTHROPIC_DEFAULT_OPUS_MODEL":   MODEL,
            "ANTHROPIC_DEFAULT_SONNET_MODEL": MODEL,
            "ANTHROPIC_DEFAULT_HAIKU_MODEL":  MODEL,
        }
        t_start = time.perf_counter()
        try:
            proc = subprocess.Popen(
                [
                    "claude",
                    "--model", MODEL,
                    "--dangerously-skip-permissions",
                    "--output-format", "stream-json",
                    "--max-turns", "30",
                    "-p", self.instance["problem_statement"],
                ],
                cwd=str(self.workdir),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                bufsize=1,
            )

            # stream-json event sequence per turn:
            #   assistant (tool_use) → user (tool_result) → assistant (next response)
            #
            # Timing breakdown:
            #   tool_ms      = user_event_time - prev_assistant_event_time  (shell execution)
            #   infer_ms     = next_assistant_event_time - user_event_time  (vLLM TTFT+decode)
            #   e2e_ms       = tool_ms + infer_ms
            #
            # infer_ms is a proxy for TTFT+decode: smaller → faster model/cache recall.

            turn_start      = time.perf_counter()
            t_tools_done    = None   # time tool_result user event received
            pending_isl     = None
            pending_turn    = 0
            pending_osl     = 0
            pending_tools:  list[str] = []

            for raw in proc.stdout:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    ev = json.loads(raw)
                except Exception:
                    continue

                etype = ev.get("type", "")

                if etype == "user":
                    # tool_result returned — marks end of tool execution
                    t_tools_done = time.perf_counter()

                elif etype == "assistant":
                    msg  = ev.get("message", {})
                    isl  = msg.get("usage", {}).get("input_tokens")
                    osl  = msg.get("usage", {}).get("output_tokens", 0)
                    cont = msg.get("content", [])

                    if isl != pending_isl:
                        # New turn: flush previous
                        if pending_isl is not None:
                            self._flush(pending_turn, pending_isl, pending_osl,
                                        pending_tools, turn_start, t_tools_done)
                            turn_start   = time.perf_counter()
                            t_tools_done = None
                        pending_isl   = isl
                        pending_turn  = len(self.turns) + 1
                        pending_osl   = osl
                        pending_tools = []
                    else:
                        pending_osl = osl

                    tools = [b["name"] for b in cont
                             if isinstance(b, dict) and b.get("type") == "tool_use"]
                    pending_tools.extend(tools)

                elif etype == "result":
                    if pending_isl is not None:
                        self._flush(pending_turn, pending_isl, pending_osl,
                                    pending_tools, turn_start, t_tools_done)
                    break

            proc.wait()

        except Exception as exc:
            with self._lock:
                self.error = str(exc)

        with self._lock:
            self.wall_time = time.perf_counter() - t_start

    def _flush(self, turn_num, isl, osl, tools, turn_start, t_tools_done):
        now     = time.perf_counter()
        e2e_ms  = (now - turn_start) * 1000
        # infer_ms = time from tool_results-sent to assistant-received ≈ TTFT + decode
        infer_ms = (now - t_tools_done) * 1000 if t_tools_done is not None else None
        tool_ms  = (t_tools_done - turn_start) * 1000 if t_tools_done is not None else None
        with self._lock:
            self.turns.append({
                "turn": turn_num, "isl": isl, "osl": osl,
                "tools": tools, "e2e_ms": e2e_ms,
                "infer_ms": infer_ms, "tool_ms": tool_ms,
            })


# ── Display ───────────────────────────────────────────────────────────────────
def _fmt_ms(ms: float | None) -> str:
    if ms is None:
        return "      ?"
    if ms < 1000:
        return f"{ms:6.0f}ms"
    return f"{ms / 1000:5.1f}s  "


def print_scoreboard(cpu: Runner, mt: Runner, elapsed: float):
    cpu_turns, cpu_wall, cpu_err = cpu.snapshot()
    mt_turns,  mt_wall,  mt_err  = mt.snapshot()

    print(f"\r\033[K", end="")
    print(f"\n{BOLD}━━━ Elapsed {elapsed:.0f}s │ CPU turns={len(cpu_turns)} │ MTier turns={len(mt_turns)} ━━━{RESET}")
    print(f"  {'Turn':>4}  {'Setup':<14}  {'Infer':>7}  {'E2E':>7}  ISL    Tools")
    print(f"  {'────':>4}  {'──────────────':<14}  {'───────':>7}  {'───────':>7}  ─────  ─────")

    n = max(len(cpu_turns), len(mt_turns))
    for i in range(n):
        for runner, color in [(cpu, BLUE), (mt, ORANGE)]:
            turns = runner.turns
            if i < len(turns):
                t      = turns[i]
                infer  = _fmt_ms(t.get("infer_ms"))
                e2e    = _fmt_ms(t["e2e_ms"])
                tools  = ",".join(t["tools"][:3]) if t["tools"] else "-"
                isl    = f"{t['isl']//1000}K"
                print(f"  {t['turn']:>4}  {color}{runner.label:<14}{RESET}  "
                      f"{infer}  {e2e}  {isl:>5}  {tools}")
            else:
                print(f"       {color}{runner.label:<14}{RESET}  {'...':>7}  {'...':>7}")
        if i < n - 1:
            print()


def print_summary(cpu: Runner, mt: Runner):
    cpu_turns, cpu_wall, cpu_err = cpu.snapshot()
    mt_turns,  mt_wall,  mt_err  = mt.snapshot()

    print(f"\n{BOLD}━━━━━━━━━━━━━━ Race Complete ━━━━━━━━━━━━━━{RESET}")

    for runner, color in [(cpu, BLUE), (mt, ORANGE)]:
        turns     = runner.turns
        wall      = runner.wall_time
        n_tools   = sum(len(t["tools"]) for t in turns)
        avg_e2e   = sum(t["e2e_ms"] for t in turns) / max(1, len(turns))
        infer_vals = [t["infer_ms"] for t in turns if t.get("infer_ms") is not None]
        avg_infer = sum(infer_vals) / len(infer_vals) if infer_vals else None
        wall_str  = f"{wall:.1f}s" if wall else "?"
        infer_str = f"avg infer {avg_infer:.0f}ms  " if avg_infer else ""
        print(f"  {color}{runner.label:<14}{RESET}  "
              f"{len(turns):>3} turns  {n_tools:>3} tool calls  "
              f"{infer_str}avg E2E {avg_e2e:.0f}ms  wall {wall_str}")
        if runner.error:
            print(f"    {YELLOW}error: {runner.error}{RESET}")

    if cpu_wall and mt_wall:
        speedup = cpu_wall / mt_wall
        color   = GREEN if speedup >= 1.2 else (YELLOW if speedup >= 1.05 else RESET)
        print(f"\n  {BOLD}MTier wall-time speedup: {color}{speedup:.2f}×{RESET}")

    if cpu_turns and mt_turns:
        n = min(len(cpu_turns), len(mt_turns))
        # E2E speedup
        cpu_e2e = sum(t["e2e_ms"] for t in cpu_turns[:n]) / max(1, n)
        mt_e2e  = sum(t["e2e_ms"] for t in mt_turns[:n])  / max(1, n)
        if mt_e2e > 0:
            e2e_speedup = cpu_e2e / mt_e2e
            color = GREEN if e2e_speedup >= 1.2 else (YELLOW if e2e_speedup >= 1.05 else RESET)
            print(f"  {BOLD}MTier turn E2E speedup:  {color}{e2e_speedup:.2f}×{RESET}  "
                  f"{DIM}(avg {n} turns, includes tool time){RESET}")
        # Inference speedup (TTFT+decode proxy, excludes tool execution)
        cpu_infer_vals = [t["infer_ms"] for t in cpu_turns[:n] if t.get("infer_ms") is not None]
        mt_infer_vals  = [t["infer_ms"] for t in mt_turns[:n]  if t.get("infer_ms") is not None]
        if cpu_infer_vals and mt_infer_vals:
            cpu_infer = sum(cpu_infer_vals) / len(cpu_infer_vals)
            mt_infer  = sum(mt_infer_vals)  / len(mt_infer_vals)
            if mt_infer > 0:
                infer_speedup = cpu_infer / mt_infer
                color = GREEN if infer_speedup >= 1.2 else (YELLOW if infer_speedup >= 1.05 else RESET)
                print(f"  {BOLD}MTier infer speedup:     {color}{infer_speedup:.2f}×{RESET}  "
                      f"{DIM}(avg {min(len(cpu_infer_vals),len(mt_infer_vals))} turns, "
                      f"tool-result→response time){RESET}")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="SWE-bench race: CPU vs MTier")
    p.add_argument("--index",      type=int, default=0, help="Dataset index (default 0)")
    p.add_argument("--id",         type=str, default=None, help="Instance ID")
    p.add_argument("--list",       action="store_true", help="List first 20 instances")
    p.add_argument("--cpu-port",   type=int, default=8001)
    p.add_argument("--mtier-port", type=int, default=8002)
    p.add_argument("--no-clone",   action="store_true",
                   help="Skip repo clone (use existing workspace)")
    args = p.parse_args()

    ds = load_instances()

    if args.list:
        print(f"\n{'#':>3}  {'Instance ID':<40}  {'Difficulty'}")
        print(f"{'───':>3}  {'─'*40}  {'──────────'}")
        for i, item in enumerate(ds):
            if i >= 20:
                break
            print(f"{i:>3}  {item['instance_id']:<40}  {item.get('difficulty','?')}")
        return

    instance = pick_instance(ds, args.id, args.index)

    print(f"\n{BOLD}SWE-bench Race: CPU vs MTier{RESET}")
    print(f"  {BLUE}blue   = hybrid-cpu   (port {args.cpu_port}){RESET}")
    print(f"  {ORANGE}orange = hybrid-mtier (port {args.mtier_port}){RESET}")
    print(f"\n  Instance : {instance['instance_id']}")
    print(f"  Repo     : {instance['repo']}  @{instance['base_commit'][:8]}")
    print(f"  Difficulty: {instance.get('difficulty','?')}")
    print(f"\n  Problem: {instance['problem_statement'][:200]}...")
    print()

    if not args.no_clone:
        print("Setting up workspaces...")
        cpu_workdir   = setup_workspace(instance, "cpu")
        mtier_workdir = setup_workspace(instance, "mtier")
    else:
        cpu_workdir   = WORKSPACE_ROOT / "cpu"   / instance["instance_id"]
        mtier_workdir = WORKSPACE_ROOT / "mtier" / instance["instance_id"]

    cpu_runner   = Runner("hybrid-cpu",   BLUE,   args.cpu_port,   instance, cpu_workdir)
    mtier_runner = Runner("hybrid-mtier", ORANGE, args.mtier_port, instance, mtier_workdir)

    print(f"Starting both agents simultaneously...\n")
    t0 = time.perf_counter()
    cpu_runner.start()
    mtier_runner.start()

    last_scoreboard = 0
    try:
        while cpu_runner.is_alive() or mtier_runner.is_alive():
            time.sleep(2)
            elapsed = time.perf_counter() - t0
            if elapsed - last_scoreboard >= 15:
                print_scoreboard(cpu_runner, mtier_runner, elapsed)
                last_scoreboard = elapsed
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Interrupted.{RESET}")

    cpu_runner.join(timeout=5)
    mtier_runner.join(timeout=5)

    print_summary(cpu_runner, mtier_runner)
    print()


if __name__ == "__main__":
    main()
