"""Wrapper around the standalone analyze_snapshot.py script.

Invoked once per level after the Prometheus snapshot is saved. Best-effort:
failures are logged but never abort the sweep — the bench's primary output
(prom_snapshot/ + config.json) is independent of the analyzer.
"""
import subprocess
import sys
from pathlib import Path


_SCRIPT_DIR     = Path(__file__).resolve().parent
_BENCH_ROOT     = _SCRIPT_DIR.parent
_AGENT_ROOT     = _BENCH_ROOT.parent
ANALYZER_SCRIPT = _AGENT_ROOT / "analysis" / "analyze_snapshot.py"
VENV_PYTHON     = Path("/home/ubuntu/vllm_xmem/.venv/bin/python")


def run_analyzer(level_dir: Path, port: int = 9099) -> None:
    """Run analyze_snapshot.py on the given level dir. Best-effort."""
    if not ANALYZER_SCRIPT.exists():
        return
    py = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable
    print(f"  [analyze] running on {level_dir.name} ...", flush=True)
    try:
        r = subprocess.run(
            [py, str(ANALYZER_SCRIPT), str(level_dir), "--port", str(port)],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode == 0:
            print(f"  [analyze] → {level_dir}/analysis/  (fig1, fig2)", flush=True)
        else:
            tail = "\n".join((r.stdout + r.stderr).splitlines()[-5:])
            print(f"  [analyze] FAILED rc={r.returncode}:\n{tail}", flush=True)
    except subprocess.TimeoutExpired:
        print(f"  [analyze] timed out after 120s", flush=True)
    except Exception as e:
        print(f"  [analyze] error: {e}", flush=True)
