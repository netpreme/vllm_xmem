"""Run metadata writers.

Two JSON snapshots of a run:

- ``write_config`` — once per run, ``config.json``: the overall config — full
  CLI args, the running server's resolved serving knobs (arg → os env → .env,
  incl. the actually-served model) and that model as a top-level label, the raw
  .env, the dataset selection (name + counts), ports, versions, gpu.
- ``save_session_metadata`` — once per problem,
  ``telemetry/<iid>/session_config.json``: the problem's id/repo/commit, the
  server + resolved model, timing and exit code. Doubles as the resume ledger
  (main.py reads ``exit_code == 0`` to skip solved problems).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from loguru import logger

from pipeline.claude import claude_version
from pipeline.utils.jsonl import instance_dir
from pipeline.vllm_server import Server, _read_env_file, get_package_version, gpu_info


def write_config(
    *,
    save_dir: Path,
    args: argparse.Namespace,
    server: Server,
    dataset: list,
    dataset_name: str,
    solved_ids: set,
    proxy_port: int,
    started_at: float,
) -> None:
    """Snapshot the overall config for this run to ``config.json``."""
    serving_config = server.serving_config()
    config = {
        "stamp": save_dir.name,
        "started_at": round(started_at, 3),
        "command": " ".join(sys.argv),
        "backend": args.backend,
        # served model resolved from args/env/.env (no running server needed);
        # the analysis model label.
        "model": serving_config["model"],
        "capture": args.capture,
        "args": vars(args),
        "serving_config": serving_config,
        "dotenv": _read_env_file(),
        "dataset": {
            "name": dataset_name,
            "pending": len(dataset),
            "skipped_solved": len(solved_ids),
        },
        "ports": {"vllm": server.port, "proxy": proxy_port, "vllm_url": server.url},
        "versions": {
            "claude": claude_version(),
            "vllm": get_package_version("vllm"),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "gpu": gpu_info(),
    }
    # default=str so Path args (e.g. --resume) serialize cleanly.
    (save_dir / "config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n"
    )
    logger.info("wrote config → {}", save_dir / "config.json")


def save_session_metadata(
    save_dir: Path,
    task: dict,
    server: Server,
    started_at: float,
    ended_at: float,
    exit_code: int,
) -> None:
    """Record one problem's session config under telemetry/<iid>/session_config.json."""
    iid = task["instance_id"]
    problem_dir = instance_dir(save_dir / "telemetry", iid)
    problem_dir.mkdir(parents=True, exist_ok=True)
    session = {
        "instance_id": iid,
        "repo": task.get("repo"),
        "base_commit": task.get("base_commit"),
        "model": server.model,
        "serving_config": server.serving_config(),
        "started_at": round(started_at, 3),
        "ended_at": round(ended_at, 3),
        "exit_code": exit_code,
    }
    (problem_dir / "session_config.json").write_text(
        json.dumps(session, indent=2) + "\n"
    )
