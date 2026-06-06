"""Re-derive vllm.jsonl + raw.jsonl from a run's saved claude transcripts.

The Anthropic/OAuth backend copies claude-cli's full session transcript to
``telemetry/<iid>/transcript.jsonl`` — the authoritative record, with FINAL
per-turn usage (correct osl) and full tool outputs. The per-turn
vllm.jsonl/raw.jsonl are derived from it, so if that derivation changes,
replay the saved transcripts instead of re-running Opus:

    python analysis/rederive_raw.py <run_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import claude


def rederive(save_dir: Path) -> None:
    telemetry = save_dir / "telemetry"
    n = 0
    for transcript in sorted(telemetry.glob("*/transcript.jsonl")):
        with open(transcript) as fh:
            claude._capture_usage(fh, telemetry, transcript.parent.name, raw=True)
        n += 1
    print(f"re-derived vllm.jsonl + raw.jsonl for {n} problems in {save_dir}")


if __name__ == "__main__":
    rederive(Path(sys.argv[1]))
