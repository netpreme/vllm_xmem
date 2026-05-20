"""Write runs/<stamp>/config.json from environment variables.

Called by run.sh as a one-shot subprocess. Keeps the bash side free of an
inline python heredoc.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from pathlib import Path


def gpu() -> dict | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()
        if not out:
            return None
        name, mem = (x.strip() for x in out[0].split(",", 1))
        return {"name": name, "count": len(out), "memory_per_gpu": mem}
    except Exception:
        return None


def main() -> int:
    backend = os.environ["BACKEND"]
    cfg = {
        "run_id":  os.environ["STAMP"],
        "agent":   "claude",
        "backend": backend,
        "machine": {
            "hostname": socket.gethostname(),
            "platform": f"{platform.system()} {platform.release()}",
            "gpu":      gpu() if backend == "vllm" else None,
        },
        "model":   os.environ["MODEL_NAME"],
        "dataset": {
            "name":  os.environ["SWE_DATASET"],
            "limit": int(os.environ["SWE_LIMIT"]),
        },
        "agent_settings": {
            "claude_max_turns":    int(os.environ["CLAUDE_MAX_TURNS"]),
            "claude_timeout_secs": int(os.environ["CLAUDE_TIMEOUT_SECS"]),
            "max_tokens_cap":      int(os.environ["MAX_TOKENS_CAP"]),
        },
    }
    Path(sys.argv[1]).write_text(json.dumps(cfg, indent=2) + "\n")

    print("[run] resolved config:")
    for k in ("run_id", "agent", "backend", "model"):
        print(f"  {k:<14} {cfg[k]}")
    g = cfg["machine"]["gpu"]
    print(f"  gpu            "
          f"{g['name']} × {g['count']} ({g['memory_per_gpu']})"
          if g else "  gpu            n/a")
    print(f"  dataset        "
          f"{cfg['dataset']['name']} (limit={cfg['dataset']['limit']})")
    print(f"  upstream       {os.environ['UPSTREAM_URL']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
