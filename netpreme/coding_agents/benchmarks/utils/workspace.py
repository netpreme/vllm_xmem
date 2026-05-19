"""SWE-bench dataset loading + per-task workspace cloning + env-file loader."""
from . import bench_concurrent_users as _bcu


load_env         = _bcu.load_env
setup_workspace  = _bcu.setup_workspace

# Default dataset ordering ranks (re-exported for callers)
_DIFF_ALIASES = {
    "easy":   "<15 min fix",
    "medium": "15 min - 1 hour",
    "hard":   "1-4 hours",
    "vhard":  ">4 hours",
}
_DIFF_RANK = {
    ">4 hours":        0,
    "1-4 hours":       1,
    "15 min - 1 hour": 2,
    "<15 min fix":     3,
}


def load_swe_bench(difficulty: list[str] | None = None,
                   start: int = 0, end: int | None = None) -> list[dict]:
    """Load SWE-bench Verified, filtered + ordered v-hard → hard → medium → easy.

    Returns the list of instance dicts in canonical order (longest problem
    statement first within each difficulty tier).
    """
    from datasets import load_dataset

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    if difficulty:
        wanted = {_DIFF_ALIASES.get(d, d) for d in difficulty}
        ds = ds.filter(lambda row: row["difficulty"] in wanted)
    rows = sorted(
        ds,
        key=lambda r: (_DIFF_RANK.get(r["difficulty"], 9),
                       -len(r["problem_statement"])),
    )
    if end is None:
        end = len(rows)
    return rows[start:end]


__all__ = ["load_env", "setup_workspace", "load_swe_bench"]
