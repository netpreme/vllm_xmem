#!/usr/bin/env python3
"""End-to-end orchestrator for one ISL/OSL benchmarking run.

Flow per problem:
    1. Cold-restart vLLM so each problem starts with an empty prefix cache.
    2. Truncate `.agent_labels` so leftover labels don't cross problems.
    3. Update `.active_instance` so completions get attributed correctly.
    4. Invoke claude-cli (via solve_problem.py) pointed at the agent_labeler
       proxy. The labeler classifies each /v1/messages call and forwards to
       vLLM; the metrics_watcher polls /metrics, pops one label per detected
       completion, and writes one CSV row per turn.
    5. After every problem, analyze.sh consolidates the CSVs into data.npz
       and renders the figures.

Usage examples:
    ./run.py                                                       # all 500 problems
    ./run.py --limit 50                                            # first 50
    ./run.py --random 100 --seed 0                                 # random sample
    ./run.py --model openai/gpt-oss-120b --tool-call-parser gpt_oss

vLLM-server flags (--model, --tool-call-parser, --tensor-parallel-size,
--max-model-len, --gpu-memory-utilization) are exported as env vars before
reset_vllm.sh relaunches the server between problems. Unset overrides fall
back to the values in ../.env, then to the hardcoded defaults in server.sh.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Paths and constants.
# ---------------------------------------------------------------------------

HERE  = Path(__file__).resolve().parent
ROOT  = HERE.parent

PIPELINE     = HERE / "pipeline"
ANALYZE_SH   = HERE / "analyze.sh"
RESET_VLLM_SH = PIPELINE / "reset_vllm.sh"

PY = Path("/root/vllm_xmem/.venv/bin/python3")
if not PY.exists():
    PY = Path(sys.executable)

DATASET = "princeton-nlp/SWE-bench_Verified"

# Ports are env-overridable so multiple TP1 shards can run side by side
# on the same host (one shard per GPU, distinct port pairs).
VLLM_PORT    = int(os.environ.get("VLLM_PORT", "8000"))
LABELER_PORT = int(os.environ.get("LABELER_PORT", "8001"))
VLLM_URL     = f"http://localhost:{VLLM_PORT}"
LABELER_URL  = f"http://127.0.0.1:{LABELER_PORT}"

# Watcher must see at least one post-restart scrape before the first claude
# request lands (and at least one post-claude scrape before we change the
# instance_id again). 0.3 s gives 3 scrapes at the 100 ms cadence.
WATCHER_QUIESCE_S = 0.3


# ---------------------------------------------------------------------------
# Configuration.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunConfig:
    """Everything the run is parameterised on. Populated from CLI flags
    with fallbacks to env vars (which `.env` has already populated)."""

    # Benchmark selection.
    limit:    int
    random:   int
    seed:     int

    # vLLM server config — forwarded to reset_vllm.sh → server.sh via env.
    model_name:             str
    tool_call_parser:       str | None
    tensor_parallel_size:   int | None
    max_model_len:          int | None
    gpu_memory_utilization: float | None

    # Per-problem agent caps.
    claude_max_turns:    int
    claude_timeout_secs: int

    @property
    def effective_limit(self) -> int:
        return self.random if self.random > 0 else self.limit


def load_env_file(path: Path) -> None:
    """Populate os.environ from a KEY=VALUE file, but only for vars that
    aren't already set. Mirrors server.sh's loader so the same precedence
    rule (caller > .env > hardcoded default) holds end to end."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.split("#", 1)[0].strip()
        os.environ.setdefault(key, val)


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    return int(raw) if raw else None


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    return float(raw) if raw else None


def parse_args() -> RunConfig:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Benchmark.
    p.add_argument("--limit",  type=int, default=500,
                   help="use the first N problems (default 500)")
    p.add_argument("--random", type=int, default=0,
                   help="uniformly random sample of N problems (overrides --limit)")
    p.add_argument("--seed",   type=int, default=0)
    # Server.
    p.add_argument("--model",
                   default=os.environ.get("MODEL_NAME",
                                          "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"))
    p.add_argument("--tool-call-parser",      default=os.environ.get("TOOL_CALL_PARSER"))
    p.add_argument("--tensor-parallel-size", "--tp",
                   type=int, default=_env_int("TENSOR_PARALLEL_SIZE"))
    p.add_argument("--max-model-len",         type=int,   default=_env_int("MAX_MODEL_LEN"))
    p.add_argument("--gpu-memory-utilization", type=float, default=_env_float("GPU_MEMORY_UTILIZATION"))
    args = p.parse_args()
    return RunConfig(
        limit=args.limit,
        random=args.random,
        seed=args.seed,
        model_name=args.model,
        tool_call_parser=args.tool_call_parser,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        claude_max_turns=int(os.environ.get("CLAUDE_MAX_TURNS", "999")),
        claude_timeout_secs=int(os.environ.get("CLAUDE_TIMEOUT_SECS", "86400")),
    )


def export_server_env(cfg: RunConfig) -> None:
    """Make the CLI-resolved values visible to reset_vllm.sh → server.sh."""
    if cfg.model_name:                   os.environ["MODEL_NAME"] = cfg.model_name
    if cfg.tool_call_parser:             os.environ["TOOL_CALL_PARSER"] = cfg.tool_call_parser
    if cfg.tensor_parallel_size:         os.environ["TENSOR_PARALLEL_SIZE"] = str(cfg.tensor_parallel_size)
    if cfg.max_model_len:                os.environ["MAX_MODEL_LEN"] = str(cfg.max_model_len)
    if cfg.gpu_memory_utilization:       os.environ["GPU_MEMORY_UTILIZATION"] = str(cfg.gpu_memory_utilization)


# ---------------------------------------------------------------------------
# Run-directory layout.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunPaths:
    """The set of files / directories that belong to one run. Centralised
    here so the rest of the script never builds paths by string concat."""

    root:        Path
    csv_dir:     Path
    workdirs:    Path
    problems:    Path
    solved:      Path
    control:     Path
    labels:      Path
    config_json: Path
    labeler_log: Path
    watcher_log: Path

    @classmethod
    def for_stamp(cls, stamp: str) -> "RunPaths":
        root = HERE / "runs" / stamp
        return cls(
            root=root,
            csv_dir=root / "per_problem",
            workdirs=Path(f"/tmp/swe_workdirs/{stamp}"),
            problems=root / "problems.jsonl",
            solved=root / "solved.txt",
            control=root / ".active_instance",
            labels=root / ".agent_labels",
            config_json=root / "config.json",
            labeler_log=root / ".labeler.log",
            watcher_log=root / ".watcher.log",
        )

    def initialise(self) -> None:
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        self.workdirs.mkdir(parents=True, exist_ok=True)
        if not self.solved.exists():
            self.solved.write_text("")
        self.control.write_text("")


# ---------------------------------------------------------------------------
# config.json — a one-shot snapshot of how this run was parameterised.
# ---------------------------------------------------------------------------

def _gpu_info() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5, check=False,
        ).stdout.strip().splitlines()
    except FileNotFoundError:
        return None
    if not out:
        return None
    name, mem = (x.strip() for x in out[0].split(",", 1))
    return {"name": name, "count": len(out), "memory_per_gpu": mem}


def write_config_json(path: Path, cfg: RunConfig, stamp: str) -> dict:
    payload = {
        "run_id":  stamp,
        "agent":   "claude",
        "backend": "vllm",
        "machine": {
            "hostname": socket.gethostname(),
            "platform": f"{platform.system()} {platform.release()}",
            "gpu":      _gpu_info(),
        },
        "model": cfg.model_name,
        "server": {
            "tool_call_parser":       cfg.tool_call_parser,
            "tensor_parallel_size":   cfg.tensor_parallel_size,
            "max_model_len":          cfg.max_model_len,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
        },
        "dataset": {"name": DATASET, "limit": cfg.effective_limit},
        "agent_settings": {
            "claude_max_turns":    cfg.claude_max_turns,
            "claude_timeout_secs": cfg.claude_timeout_secs,
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def print_config_summary(cfg: dict) -> None:
    print("[run] resolved config:")
    print(f"  run_id         {cfg['run_id']}")
    print(f"  model          {cfg['model']}")
    gpu = cfg["machine"]["gpu"]
    if gpu:
        print(f"  gpu            {gpu['name']} × {gpu['count']} ({gpu['memory_per_gpu']})")
    else:
        print(f"  gpu            n/a")
    print(f"  dataset        {cfg['dataset']['name']} (limit={cfg['dataset']['limit']})")
    print(f"  upstream       {VLLM_URL}")


# ---------------------------------------------------------------------------
# Sidecar processes (agent_labeler + metrics_watcher).
# ---------------------------------------------------------------------------

class Sidecar:
    """A background subprocess managed for the duration of the run.

    `start()` spawns and `stop()` is safe to call multiple times. We send
    SIGTERM (not SIGKILL) so the process can flush whatever it's holding
    in memory before exiting.
    """

    def __init__(self, name: str, argv: list[str], log_path: Path) -> None:
        self.name      = name
        self._argv     = argv
        self._log_path = log_path
        self._proc:  subprocess.Popen | None = None
        self._logfh: any = None

    def start(self) -> None:
        self._logfh = self._log_path.open("w")
        self._proc = subprocess.Popen(
            self._argv,
            stdout=self._logfh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        print(f"[run] started {self.name} (pid {self._proc.pid})")

    def stop(self) -> None:
        if self._proc and self._proc.poll() is None:
            print(f"[run] stopping {self.name} (pid {self._proc.pid})")
            self._proc.terminate()
            try:
                self._proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait()
        if self._logfh:
            self._logfh.close()
            self._logfh = None


def wait_for_http(url: str, timeout_s: float, interval_s: float = 0.1) -> bool:
    """Poll `url` until it returns 2xx, or until `timeout_s` elapses."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if 200 <= r.status < 300:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(interval_s)
    return False


# ---------------------------------------------------------------------------
# Problem dispatch.
# ---------------------------------------------------------------------------

def iter_problems(path: Path) -> Iterator[dict]:
    """Yield one problem record per non-blank line of problems.jsonl."""
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line:
            yield json.loads(line)


def fetch_problems(cfg: RunConfig, paths: RunPaths) -> int:
    argv = [str(PY), str(PIPELINE / "fetch_dataset.py"),
            "--dataset", DATASET, "--out", str(paths.problems)]
    if cfg.random > 0:
        argv += ["--random", str(cfg.random), "--seed", str(cfg.seed)]
    else:
        argv += ["--limit", str(cfg.limit)]
    subprocess.run(argv, check=True)
    return sum(1 for _ in iter_problems(paths.problems))


def solve_one(paths: RunPaths, cfg: RunConfig, problem: dict) -> int:
    """Run claude-cli against one SWE-bench problem. Returns its exit code.

    The sequence (cold-restart → reset label queue → set control file →
    quiesce → claude → quiesce → clear control file) is what keeps the
    per-turn watcher's row attribution aligned with the right instance_id.
    """
    if os.environ.get("SKIP_VLLM_RESET", "0") != "1":
        subprocess.run(["bash", str(RESET_VLLM_SH)], check=False)
    paths.labels.write_text("")
    paths.control.write_text(problem["instance_id"])
    time.sleep(WATCHER_QUIESCE_S)

    try:
        result = subprocess.run(
            [
                str(PY), str(PIPELINE / "solve_problem.py"),
                "--instance-id",       problem["instance_id"],
                "--repo",              problem["repo"],
                "--base-commit",       problem["base_commit"],
                "--problem-statement", problem["problem_statement"],
                "--model",             cfg.model_name,
                "--workdir-root",      str(paths.workdirs),
                "--max-turns",         str(cfg.claude_max_turns),
                "--timeout-secs",      str(cfg.claude_timeout_secs),
                "--vllm-url",          LABELER_URL,
                "--per-problem-dir",   str(paths.csv_dir),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return result.returncode
    finally:
        time.sleep(WATCHER_QUIESCE_S)
        paths.control.write_text("")


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------

def main() -> int:
    load_env_file(ROOT / ".env")
    cfg = parse_args()
    export_server_env(cfg)

    if not wait_for_http(f"{VLLM_URL}/v1/models", timeout_s=2.0):
        print(f"[run] vllm not reachable at {VLLM_URL} — start it first:",
              file=sys.stderr)
        print(f"      bash {ROOT}/server.sh", file=sys.stderr)
        return 1

    stamp = os.environ.get("RUN_STAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = RunPaths.for_stamp(stamp)
    paths.initialise()
    print(f"[run] writing to {paths.root}")

    config = write_config_json(paths.config_json, cfg, stamp)
    print_config_summary(config)

    sidecars = [
        Sidecar("agent_labeler",
                argv=[str(PY), str(PIPELINE / "agent_labeler.py"),
                      "--upstream",     VLLM_URL,
                      "--labels-file",  str(paths.labels),
                      "--listen-port",  str(LABELER_PORT)],
                log_path=paths.labeler_log),
        Sidecar("metrics_watcher",
                argv=[str(PY), str(PIPELINE / "metrics_watcher.py"),
                      "--vllm-url",            VLLM_URL,
                      "--control-file",        str(paths.control),
                      "--per-problem-csv-dir", str(paths.csv_dir),
                      "--labels-file",         str(paths.labels)],
                log_path=paths.watcher_log),
    ]

    try:
        sidecars[0].start()
        if not wait_for_http(f"{LABELER_URL}/v1/models", timeout_s=10.0):
            print(f"[run] labeler failed to start; see {paths.labeler_log}",
                  file=sys.stderr)
            return 1
        sidecars[1].start()

        if paths.problems.exists():
            total = sum(1 for _ in iter_problems(paths.problems))
            print(f"[run] reusing existing problems.jsonl ({total} problems)")
        else:
            total = fetch_problems(cfg, paths)
        print(f"[run] {total} problems queued")

        for i, problem in enumerate(iter_problems(paths.problems), start=1):
            iid    = problem["instance_id"]
            commit = problem["base_commit"][:8]
            print(f"[run] [{i}/{total}] {iid} ({problem['repo']} @ {commit})")

            if solve_one(paths, cfg, problem) == 0:
                with paths.solved.open("a") as f:
                    f.write(iid + "\n")
            else:
                print("[run]   ! failed, continuing")
    finally:
        for s in reversed(sidecars):
            s.stop()

    n_solved = sum(1 for _ in paths.solved.read_text().splitlines() if _.strip())
    n_csvs   = len(list(paths.csv_dir.glob("*.csv")))
    print(f"\n[run] done. results at {paths.root}")
    print(f"[run]   solved.txt:       {n_solved} of {total}")
    print(f"[run]   per-problem CSVs: {n_csvs}\n")

    subprocess.run(["bash", str(ANALYZE_SH), str(paths.root)], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
