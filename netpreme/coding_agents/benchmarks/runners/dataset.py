"""SWE-bench Verified loader: filter by difficulty, order (shuffle or
deterministic v-hard → easy), and slice. Used by capture mode.
"""
import random
import time
from dataclasses import dataclass


DIFFICULTY_ALIASES: dict[str, str] = {
    "easy":   "<15 min fix",
    "medium": "15 min - 1 hour",
    "hard":   "1-4 hours",
    "vhard":  ">4 hours",
}
DIFFICULTY_RANK: dict[str, int] = {
    ">4 hours":        0,   # v-hard first
    "1-4 hours":       1,
    "15 min - 1 hour": 2,
    "<15 min fix":     3,
}


@dataclass
class SWEBenchDataset:
    """Filtered + ordered + sliced SWE-bench Verified task list."""
    instances: list[dict]
    shuffle_seed: int | None  # None = deterministic order

    @classmethod
    def load(
        cls,
        *,
        difficulty: list[str] | None = None,
        no_shuffle: bool = False,
        shuffle_seed: int | None = None,
        start: int = 0,
        end: int | None = None,
    ) -> "SWEBenchDataset":
        from datasets import load_dataset
        print("Loading SWE-bench Verified ...", flush=True)
        ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")

        if difficulty:
            wanted = {DIFFICULTY_ALIASES.get(d, d) for d in difficulty}
            ds = ds.filter(lambda row: row["difficulty"] in wanted)
            print(f"  Difficulty filter: {wanted}  →  {len(ds)} tasks", flush=True)

        rows = list(ds)
        if no_shuffle:
            rows = sorted(
                rows,
                key=lambda r: (DIFFICULTY_RANK.get(r["difficulty"], 9),
                               -len(r["problem_statement"])),
            )
            seed = None
            print("  Task order: v-hard → hard → medium → easy (--no-shuffle)",
                  flush=True)
        else:
            seed = shuffle_seed if shuffle_seed is not None else int(time.time_ns() % (2**32))
            random.Random(seed).shuffle(rows)
            print(f"  Task order: shuffled (seed={seed})", flush=True)

        end = end if end is not None else len(rows)
        return cls(instances=rows[start:end], shuffle_seed=seed)

    def __len__(self) -> int:
        return len(self.instances)
