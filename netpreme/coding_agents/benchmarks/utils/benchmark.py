"""High-level benchmark API: CodingAgents.

Three modes, all exposed via one method:

    # Mode 1 — Run real claude agents at a given concurrency.
    agents = CodingAgents(
        concurrency=[16],
        setups=["hybrid-mtier", "hybrid-cpu"],
        sustained_mins=20,
    )
    sweep_dir = agents.benchmark()

    # Mode 2 — Same, but record every claude → vLLM exchange to trace files.
    sweep_dir = agents.benchmark(capture_to="/path/to/capture_<ts>/")

    # Mode 3 — Replay a previously-captured trace against fresh vLLM(s).
    replay_dir = agents.benchmark(from_traces="/path/to/capture_dir/",
                                  force_osl=True)

    # Analyse (writes figs + CSV under <sweep_dir>/c0NN/analysis/)
    agents.analyze()

`force_osl=True` only applies in trace-replay mode: rewrites each replayed
turn to pin generation length to the captured output_tokens via the vLLM
``ignore_eos`` + ``min_tokens`` extension.
"""
import os
import sys
from pathlib import Path
from typing import Sequence

from . import bench_concurrent_users as _bcu
from .lifecycle import run_analyzer


# Per-setup defaults: which port and which GPU each setup binds to.
SETUP_DEFAULTS = {
    "hybrid-mtier": (8001, "0"),
    "mtier-only":   (8001, "0"),
    "hybrid-cpu":   (8002, "1"),
    "cpu-only":     (8002, "1"),
}


class CodingAgents:
    """Driver for the netpreme concurrent-claude-agents benchmark.

    Parameters
    ----------
    concurrency : int | list[int]
        One or more concurrency levels to sweep.
    setups : list[str]
        vLLM setup labels (e.g. ``["hybrid-mtier", "hybrid-cpu"]``). Defaults
        run on the ports/GPUs in :data:`SETUP_DEFAULTS`.
    sustained_mins : float
        Wall-clock cap per level in run / capture mode. Default 20 min.
    duration_cap_mins : float | None
        Hard duration cap in replay mode. Default ``1.5 * captured duration``.
    model : str | None
        Override the model name (defaults to ``$MODEL`` env or built-in).
    tp, gpu_util, max_num_seqs : numeric overrides forwarded to ``start_vllm``.
    difficulty : list[str] | None
        SWE-bench difficulty filter (run / capture mode only).
    swe_range : tuple[int, int | None]
        ``(start, end)`` into the SWE-bench dataset (run / capture mode only).
    workspace_root : str | Path | None
        Where to clone task workspaces (default ``/tmp/swe_workspaces_p<port>``).
    no_clone : bool
        Skip workspace setup; run claude in the current directory.
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
            Path(workspace_root).resolve() if workspace_root else None
        )
        self.no_clone          = no_clone
        self.port_override     = port

        # Set by benchmark() — most recent run directory + per-level dirs.
        self.last_run_dir:    "Path | None"        = None
        self.last_level_dirs: list[Path]            = []

    # ────────────────────────────────────────────────────────────────────
    #  benchmark()  — single entry, dispatches by argument
    # ────────────────────────────────────────────────────────────────────
    def benchmark(self,
                  from_traces: "str | Path | None" = None,
                  capture_to:  "str | Path | None" = None,
                  force_osl:   bool = False) -> Path:
        """Execute the benchmark. Returns the top-level run directory.

        Mode dispatch:
            * ``from_traces=<path>``           → replay from captured traces.
            * ``capture_to=<path>``            → run + record traces.
            * neither                          → plain run.

        ``force_osl=True`` only applies with ``from_traces``: rewrites each
        replayed request to pin generation length to the captured OSL via
        the vLLM ``ignore_eos``+``min_tokens`` extension.
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

    # ────────────────────────────────────────────────────────────────────
    #  analyze()
    # ────────────────────────────────────────────────────────────────────
    def analyze(self, level_dirs: "Sequence[Path] | None" = None) -> list[Path]:
        """Run the analyzer over the given level dirs (or the most recent run)."""
        dirs = list(level_dirs) if level_dirs is not None else list(self.last_level_dirs)
        for d in dirs:
            run_analyzer(d)
        return dirs

    # ────────────────────────────────────────────────────────────────────
    #  internal: capture-run (Mode 1 and 2)
    # ────────────────────────────────────────────────────────────────────
    def _run_capture(self, capture_to: "str | Path | None") -> Path:
        argv = [
            "--concurrency", *[str(c) for c in self.concurrency],
            "--setup",       *self.setups,
            "--sustained-mins", str(self.sustained_mins),
        ]
        if self.tp           is not None: argv += ["--tp",       str(self.tp)]
        if self.gpu_util     is not None: argv += ["--gpu-util", str(self.gpu_util)]
        if self.gpus         is not None: argv += ["--gpus",     str(self.gpus)]
        if self.max_num_seqs is not None: argv += ["--max-num-seqs", str(self.max_num_seqs)]
        if self.port_override is not None: argv += ["--port",    str(self.port_override)]
        if self.difficulty:               argv += ["--difficulty", *self.difficulty]
        if self.swe_range[0]:             argv += ["--start", str(self.swe_range[0])]
        if self.swe_range[1] is not None: argv += ["--end",   str(self.swe_range[1])]
        if self.workspace_root:           argv += ["--workspace-root", str(self.workspace_root)]
        if self.no_clone:                 argv += ["--no-clone"]
        if self.model:                    argv += ["--model", self.model]
        if capture_to:                    argv += ["--capture-traces", str(capture_to)]

        old_argv = sys.argv
        sys.argv = ["bench_concurrent_users.py", *argv]
        try:
            _bcu.main()
        finally:
            sys.argv = old_argv

        runs = sorted(Path(_bcu.RESULTS_DIR).glob("bench_sweep_*"),
                      key=lambda p: p.stat().st_mtime)
        if runs:
            self.last_run_dir    = runs[-1]
            self.last_level_dirs = sorted(
                [p for p in runs[-1].iterdir() if p.is_dir() and p.name.startswith("c")]
            )
        return self.last_run_dir

    # ────────────────────────────────────────────────────────────────────
    #  internal: replay-from-trace (Mode 3) — delegates to from_trace
    # ────────────────────────────────────────────────────────────────────
    def _run_replay(self, from_traces: "str | Path") -> Path:
        capture_dir = Path(from_traces).expanduser().resolve()
        if not (capture_dir / "capture_meta.json").exists():
            raise FileNotFoundError(f"{capture_dir}/capture_meta.json not found")

        from . import from_trace as _br
        argv = [
            "--capture-dir", str(capture_dir),
            "--setup",       *self.setups,
        ]
        if self.duration_cap_mins is not None:
            argv += ["--duration-cap-mins", str(self.duration_cap_mins)]
        if len(self.concurrency) == 1:
            argv += ["--concurrency", str(self.concurrency[0])]
        if self.tp           is not None: argv += ["--tp", str(self.tp)]
        if self.gpu_util     is not None: argv += ["--gpu-util", str(self.gpu_util)]
        if self.max_num_seqs is not None: argv += ["--max-num-seqs", str(self.max_num_seqs)]
        if len(self.setups) == 1:
            if self.gpus           is not None: argv += ["--gpus", self.gpus]
            if self.port_override  is not None: argv += ["--port", str(self.port_override)]

        old_argv = sys.argv
        sys.argv = ["from_trace.py", *argv]
        try:
            _br.main()
        finally:
            sys.argv = old_argv

        runs = sorted(Path(_bcu.RESULTS_DIR).glob("from_trace_*"),
                      key=lambda p: p.stat().st_mtime)
        if runs:
            self.last_run_dir    = runs[-1]
            self.last_level_dirs = sorted(
                [p for p in runs[-1].iterdir() if p.is_dir() and p.name.startswith("c")]
            )
        return self.last_run_dir


__all__ = ["CodingAgents", "SETUP_DEFAULTS"]
