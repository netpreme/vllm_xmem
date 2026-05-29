#!/usr/bin/env python3
"""Coding-agent benchmark runner.

dataset = datasets.get_dataset(args, run_dir)
agent   = CodingAgent(model=..., run_dir=..., capture=True)
try:
    for data in dataset:
        agent.init()         # cold-restart vllm
        agent.run(data)      # clone + claude-cli
        agent.save(run_dir)  # per-problem meta.json + solved.txt
        agent.reset()
finally:
    agent.close()             # stop sidecars + write run_meta.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Make `from pipeline import ...` work when this script is invoked directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pipeline import claude, datasets, git_repo, http_utils, sidecar

HERE = Path(__file__).resolve().parent
PIPELINE = HERE / "pipeline"
ANALYZE_SH = HERE / "analyze.sh"

# Ports are env-overridable so multiple TP1 shards can run side by side
# on the same host (one shard per GPU, distinct port pairs).
VLLM_PORT = int(os.environ.get("VLLM_PORT", "8000"))
LABELER_PORT = int(os.environ.get("LABELER_PORT", "8001"))
VLLM_URL = f"http://localhost:{VLLM_PORT}"
QUIESCE_S = 0.3
DATASET = "princeton-nlp/SWE-bench_Verified"


# ---------------------------------------------------------------------------
# Coding agent. Composes helpers from pipeline/; no implementation details
# beyond per-problem orchestration.
# ---------------------------------------------------------------------------


class CodingAgent:
    """Solves SWE-bench problems one at a time. Owns the metrics_watcher
    sidecar (always) and, with capture=True, the agent_labeler proxy.

    The caller drives the per-problem lifecycle and must call close()
    when done (use try/finally) to terminate the sidecars and flush
    run_meta.json:
        init()   cold-restart vLLM
        run(p)   clone the target repo, drive claude-cli
        save(d)  write per_problem/<iid>.meta.json + append solved.txt
        reset()  per-problem teardown
        close()  stop the sidecars, write run_meta.json
    """

    def __init__(self, model, run_dir, *, capture: bool = False):
        self.model = model
        self.run_dir = run_dir
        self.capture = capture
        self.per_problem_dir = run_dir / "per_problem"
        self.workdirs = Path(f"/tmp/swe_workdirs/{run_dir.name}")
        self.control = run_dir / ".active_instance"
        self.per_problem_dir.mkdir(parents=True, exist_ok=True)
        self.workdirs.mkdir(parents=True, exist_ok=True)
        self.control.write_text("")
        self._started_at = time.time()
        self._problem = self._workdir = None
        self._exit_code: int | None = None
        self._t_start = self._t_end = None
        self._labeler = None

        if capture:
            self._labeler = sidecar.start(
                "agent_labeler",
                [
                    sys.executable,
                    str(PIPELINE / "agent_labeler.py"),
                    "--upstream",
                    VLLM_URL,
                    "--control-file",
                    str(self.control),
                    "--out-dir",
                    str(self.per_problem_dir),
                    "--listen-port",
                    str(LABELER_PORT),
                ],
                run_dir / ".agent_labeler.log",
            )
            if not http_utils.check_initialized(
                f"http://127.0.0.1:{LABELER_PORT}/v1/models", 10.0
            ):
                raise RuntimeError(
                    f"labeler did not start; see {run_dir}/.agent_labeler.log"
                )
            self.base_url = f"http://127.0.0.1:{LABELER_PORT}"
        else:
            self.base_url = VLLM_URL

        self._watcher = sidecar.start(
            "metrics_watcher",
            [
                sys.executable,
                str(PIPELINE / "metrics_watcher.py"),
                "--vllm-url",
                VLLM_URL,
                "--control-file",
                str(self.control),
                "--out-dir",
                str(self.per_problem_dir),
            ],
            run_dir / ".metrics_watcher.log",
        )

    def init(self):
        subprocess.run(["bash", str(PIPELINE / "reset_vllm.sh")], check=False)

    def run(self, problem):
        self._problem = problem
        self._exit_code = None
        iid = problem["instance_id"]
        # Truncate any prior per-problem files (retry-on-resume safety).
        for suffix in (".vllm.jsonl", ".proxy.jsonl"):
            (self.per_problem_dir / f"{iid}{suffix}").unlink(missing_ok=True)
        self.control.write_text(iid)
        time.sleep(QUIESCE_S)
        self._workdir = Path(
            tempfile.mkdtemp(prefix=f"{iid}.", dir=self.workdirs)
        )
        self._t_start = time.time()
        try:
            repo = git_repo.clone(problem, self._workdir)
        except subprocess.SubprocessError as exc:
            print(f"[agent]   clone failed: {exc!r}", file=sys.stderr)
            self._exit_code = 1
            self._t_end = time.time()
            return
        self._exit_code = claude.solve(
            problem, repo, model=self.model, base_url=self.base_url
        )
        self._t_end = time.time()

    def save(self, run_dir):
        if not self._problem:
            return
        iid = self._problem["instance_id"]
        meta = {
            "instance_id": iid,
            "difficulty": self._problem.get("difficulty"),
            "repo": self._problem.get("repo"),
            "base_commit": self._problem.get("base_commit"),
            "started_at": round(self._t_start, 3) if self._t_start else None,
            "ended_at": round(self._t_end, 3) if self._t_end else None,
            "exit_code": self._exit_code,
        }
        (self.per_problem_dir / f"{iid}.meta.json").write_text(
            json.dumps(meta, indent=2) + "\n"
        )
        if self._exit_code == 0:
            with (run_dir / "solved.txt").open("a") as f:
                f.write(iid + "\n")

    def reset(self):
        if self._workdir:
            shutil.rmtree(self._workdir, ignore_errors=True)
        time.sleep(QUIESCE_S)
        self.control.write_text("")
        self._problem = self._workdir = None
        self._exit_code = None
        self._t_start = self._t_end = None

    def close(self):
        sidecar.stop("metrics_watcher", self._watcher)
        if self._labeler is not None:
            sidecar.stop("agent_labeler", self._labeler)
        (self.run_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "model": self.model,
                    "base_url": self.base_url,
                    "vllm_url": VLLM_URL,
                    "dataset": DATASET,
                    "capture": self.capture,
                    "started_at": round(self._started_at, 3),
                    "ended_at": round(time.time(), 3),
                },
                indent=2,
            )
            + "\n"
        )


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-problems", type=int, default=500)
    p.add_argument("--random", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--capture",
        action="store_true",
        help="run the agent_labeler proxy to capture per-turn "
        "agentic metadata (system_prompt_chars, tool calls, stop reason)",
    )
    p.add_argument(
        "--resume",
        type=Path,
        help="reuse <RUN_DIR>/problems.jsonl, skip already-solved ids",
    )
    args = p.parse_args()

    if not http_utils.check_initialized(f"{VLLM_URL}/v1/models", 2.0):
        print(f"[run] vllm not reachable at {VLLM_URL}", file=sys.stderr)
        return 1
    model = http_utils.get_model_name(VLLM_URL)
    print(f"[run] vllm is serving: {model}")

    run_dir = datasets.setup_run_dir(args.resume.name if args.resume else None)
    dataset = datasets.get_dataset(
        name=DATASET,
        out_path=run_dir / "problems.jsonl",
        num_problems=args.num_problems,
        random=args.random,
        seed=args.seed,
    )
    pending = datasets.pending_problems(dataset, run_dir / "solved.txt")
    print(f"[run] {len(pending)} pending of {len(dataset)} → {run_dir}")

    agent = CodingAgent(model=model, run_dir=run_dir, capture=args.capture)
    try:
        offset = len(dataset) - len(pending) + 1
        for i, data in enumerate(pending, start=offset):
            print(
                f"[run] [{i}/{len(dataset)}] {data['instance_id']} "
                f"({data['repo']} @ {data['base_commit'][:8]})"
            )
            agent.init()
            agent.run(data)
            agent.save(run_dir)
            agent.reset()
    finally:
        agent.close()

    subprocess.run(["bash", str(ANALYZE_SH), str(run_dir)], check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
