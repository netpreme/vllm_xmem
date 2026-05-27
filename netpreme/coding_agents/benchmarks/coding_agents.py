"""High-level benchmark API: CodingAgents.

Three modes, one method:

    # Mode 1 — plain run with N agents per backend (no traces)
    agents = CodingAgents(
        concurrency=[16],
        setups=["hybrid-mtier", "hybrid-cpu"],
        sustained_mins=20,
    )
    sweep_dir = agents.benchmark()

    # Mode 2 — same, but record every claude → vLLM exchange to JSONL traces
    sweep_dir = agents.benchmark(capture_to="enabled")

    # Mode 3 — replay a previously-captured trace dir
    replay_dir = agents.benchmark(
        from_traces="/path/to/capture_dir/",
        force_osl=True,
    )

    # Analyse (writes figs + CSV under <sweep_dir>/c0NN/analysis/)
    agents.analyze()

``force_osl=True`` only applies with ``from_traces``: rewrites each replayed
turn to pin generation length to the captured output_tokens via the vLLM
``ignore_eos``+``min_tokens`` extension.
"""
import argparse
import os
from pathlib import Path
from typing import Sequence

from modes.capture import CaptureBenchmark, RESULTS_DIR
from modes.replay import ReplayBenchmark
from utils.analyzer import run_analyzer
from utils.config import SETUP_DEFAULTS


class CodingAgents:
    """Programmatic driver for the netpreme concurrent-claude-agents benchmark.

    Parameters
    ----------
    concurrency : int | list[int]
        One or more concurrency levels to sweep.
    setups : list[str]
        vLLM setup labels (e.g. ``["hybrid-mtier", "hybrid-cpu"]``).
    sustained_mins : float
        Wall-clock cap per level in capture mode. Default 20 min.
    duration_cap_mins : float | None
        Hard duration cap in replay mode. Default 20 min.
    model, tp, gpu_util, max_num_seqs : forwarded to VLLMServer.
    difficulty : list[str] | None
        SWE-bench difficulty filter (capture mode only).
    swe_range : tuple[int, int | None]
        ``(start, end)`` into the SWE-bench dataset (capture mode only).
    workspace_root : where to clone task workspaces.
    no_clone : skip workspace setup; run claude in cwd (debug only).
    """

    def __init__(
        self,
        concurrency: int | Sequence[int] = (16,),
        setups: Sequence[str] = ("hybrid-mtier", "hybrid-cpu"),
        sustained_mins: float = 20.0,
        duration_cap_mins: float | None = None,
        model: str | None = None,
        tp: int | None = None,
        gpu_util: float | None = None,
        gpus: str | None = None,
        max_num_seqs: int | None = None,
        difficulty: list[str] | None = None,
        swe_range: tuple[int, int | None] = (0, None),
        workspace_root: "str | Path | None" = None,
        no_clone: bool = False,
        port: int | None = None,
    ):
        self.concurrency = (
            [concurrency] if isinstance(concurrency, int) else list(concurrency)
        )
        self.setups            = list(setups)
        self.sustained_mins    = sustained_mins
        self.duration_cap_mins = duration_cap_mins
        self.model             = model
        self.tp                = tp
        self.gpu_util          = gpu_util
        self.gpus              = gpus
        self.max_num_seqs      = max_num_seqs
        self.difficulty        = difficulty
        self.swe_range         = swe_range
        self.workspace_root    = (
            str(Path(workspace_root).resolve()) if workspace_root else None
        )
        self.no_clone          = no_clone
        self.port_override     = port

        self.last_run_dir:    "Path | None" = None
        self.last_level_dirs: list[Path]    = []

    # ────────────────────────────────────────────────────────────────────
    def benchmark(
        self,
        from_traces: "str | Path | None" = None,
        capture_to:  "str | Path | None" = None,
        force_osl:   bool = False,
    ) -> Path:
        """Execute the benchmark. Returns the top-level run directory.

        Mode dispatch:
            * ``from_traces=<path>``  → replay from captured traces.
            * ``capture_to=<path>``   → run + record traces.
            * neither                 → plain run.

        ``force_osl=True`` only applies with ``from_traces``.
        """
        if from_traces is not None and capture_to is not None:
            raise ValueError("from_traces and capture_to are mutually exclusive")
        if from_traces is not None:
            if force_osl:
                os.environ["REPLAY_FORCE_OSL"] = "1"
            return self._run_replay(from_traces)
        if force_osl:
            raise ValueError("force_osl only applies in replay (from_traces) mode")
        return self._run_capture(capture_to)

    def analyze(self, level_dirs: "Sequence[Path] | None" = None) -> list[Path]:
        """Run the analyzer over the given level dirs (or the most recent run)."""
        dirs = list(level_dirs) if level_dirs is not None else list(self.last_level_dirs)
        for d in dirs:
            run_analyzer(d)
        return dirs

    # ────────────────────────────────────────────────────────────────────
    #  internal: capture / plain (modes 1 + 2)
    # ────────────────────────────────────────────────────────────────────
    def _run_capture(self, capture_to: "str | Path | None") -> Path:
        args = argparse.Namespace(
            concurrency=self.concurrency,
            setup=self.setups,
            sustained_mins=self.sustained_mins,
            start=self.swe_range[0] or 0,
            end=self.swe_range[1],
            difficulty=self.difficulty,
            no_shuffle=False,
            shuffle_seed=None,
            port=self.port_override,
            model=self.model,
            tp=self.tp,
            gpu_util=self.gpu_util,
            gpus=self.gpus,
            max_num_seqs=self.max_num_seqs,
            workspace_root=self.workspace_root,
            no_clone=self.no_clone,
            capture_traces=(str(capture_to) if capture_to else None),
        )
        run_dir = CaptureBenchmark(args).run()
        self._record_run(run_dir, prefix="bench_sweep_")
        return self.last_run_dir

    # ────────────────────────────────────────────────────────────────────
    #  internal: replay-from-trace (mode 3)
    # ────────────────────────────────────────────────────────────────────
    def _run_replay(self, from_traces: "str | Path") -> Path:
        capture_dir = Path(from_traces).expanduser().resolve()
        if not (capture_dir / "capture_meta.json").exists():
            raise FileNotFoundError(f"{capture_dir}/capture_meta.json not found")

        single_setup = len(self.setups) == 1
        args = argparse.Namespace(
            capture_dir=str(capture_dir),
            setup=self.setups,
            duration_cap_mins=self.duration_cap_mins,
            concurrency=self.concurrency[0] if len(self.concurrency) == 1 else None,
            max_num_seqs=self.max_num_seqs,
            tp=self.tp,
            gpu_util=self.gpu_util,
            gpus=self.gpus if single_setup else None,
            port=self.port_override if single_setup else None,
        )
        run_dir = ReplayBenchmark(args).run()
        self._record_run(run_dir, prefix="from_trace_")
        return self.last_run_dir

    # ────────────────────────────────────────────────────────────────────
    def _record_run(self, run_dir: Path, prefix: str) -> None:
        """Cache the most recent run dir + per-level dirs for analyze()."""
        # Prefer the dir we got back; fall back to globbing if it's missing.
        if run_dir and run_dir.exists():
            self.last_run_dir = run_dir
        else:
            runs = sorted(RESULTS_DIR.glob(f"{prefix}*"),
                          key=lambda p: p.stat().st_mtime)
            self.last_run_dir = runs[-1] if runs else None
        if self.last_run_dir:
            self.last_level_dirs = sorted(
                p for p in self.last_run_dir.iterdir()
                if p.is_dir() and p.name.startswith("c")
            )


__all__ = ["CodingAgents", "SETUP_DEFAULTS"]
