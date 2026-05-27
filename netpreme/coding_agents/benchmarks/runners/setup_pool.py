"""One setup's Claude pool: N concurrent agents, kept full for `duration_s`.

SetupPool clones the initial N workspaces in parallel, then runs N concurrent
ClaudeTasks on them. As each task finishes, the next instance's workspace is
already cloned in the background and the next task is submitted — keeping
exactly N tasks active at all times.

Designed to be run from a thread so multiple SetupPools can run their pools
in parallel for a fair dual-backend comparison (mtier + cpu).
"""
import concurrent.futures
import queue as _queue
import threading
import time
from pathlib import Path

from runners.claude_task import (
    CaptureConfig,
    ClaudeTask,
    kill_all_claudes,
    kill_all_proxies,
)
from runners.workspace import setup_workspace


CLONE_WORKERS = 16


class SetupPool:
    """Drive one backend's claude pool at a fixed concurrency.

    Parameters
    ----------
    spec : dict
        Setup spec: {setup, base_url, workspace_root, ...}.
    concurrency : int
        Target number of in-flight claude tasks.
    instances : list[dict]
        SWE-bench task dicts. Indexed in order; first N go in immediately,
        the rest stream in as tasks finish.
    duration_s : float
        Wall-clock cap for this pool. Stops submitting new tasks past this.
    model : str
        Model name passed to claude.
    no_clone : bool
        If True, skip workspace setup and run claude in the current cwd
        (debug-only — agents from different tasks share one workdir).
    capture_config : CaptureConfig | None
        If set, each ClaudeTask spawns a CaptureProxy to tee its traffic.
    """

    def __init__(
        self,
        spec: dict,
        concurrency: int,
        instances: list[dict],
        duration_s: float,
        model: str,
        no_clone: bool,
        capture_config: CaptureConfig | None = None,
    ):
        self.spec           = spec
        self.concurrency    = concurrency
        self.instances      = instances
        self.duration_s     = duration_s
        self.model          = model
        self.no_clone       = no_clone
        self.capture_config = capture_config

    def run(self, t_start_mono: float, shared_state: dict | None = None) -> dict:
        """Run the pool. Returns per-setup counters.

        `shared_state[setup_label]` (if provided) is updated in-place every
        time a task completes so a status thread can render live counts.
        """
        setup    = self.spec["setup"]
        base_url = self.spec["base_url"]

        initial_specs, work_q, stop_clone = self._stage_workspaces()

        n_done = n_ok = n_fail = 0
        ex = concurrent.futures.ThreadPoolExecutor(max_workers=self.concurrency)
        active: dict = {}
        spec_iter = iter(initial_specs)

        def _next_spec() -> "dict | None":
            s = next(spec_iter, None)
            if s is not None:
                return s
            if time.monotonic() - t_start_mono >= self.duration_s:
                return None
            try:
                return work_q.get(timeout=1.0)
            except _queue.Empty:
                return None

        def _submit(s: dict):
            task = ClaudeTask(s["instance"], s["workdir"], self.model, base_url,
                              capture_config=self.capture_config)
            f = ex.submit(task.run)
            active[f] = s

        for _ in range(self.concurrency):
            s = _next_spec()
            if s is None:
                break
            _submit(s)

        def _update_shared():
            if shared_state is not None:
                shared_state[setup] = {
                    "n_done":   n_done,
                    "n_ok":     n_ok,
                    "n_fail":   n_fail,
                    "n_active": len(active),
                }

        _update_shared()
        try:
            while active and time.monotonic() - t_start_mono < self.duration_s:
                done, _ = concurrent.futures.wait(
                    list(active.keys()), timeout=1.0,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for fut in done:
                    try:
                        _, ok = fut.result()
                    except Exception:
                        ok = False
                    del active[fut]
                    n_done += 1
                    if ok: n_ok  += 1
                    else:  n_fail += 1
                    if time.monotonic() - t_start_mono < self.duration_s:
                        s = _next_spec()
                        if s is not None:
                            _submit(s)
                _update_shared()
        finally:
            if not self.no_clone:
                stop_clone.set()
            for f in list(active.keys()):
                f.cancel()
            # Kill in-flight claude subprocesses BEFORE shutdown(wait=...).
            # Otherwise worker threads stuck in proc.wait() prevent Python exit.
            kill_all_claudes()
            kill_all_proxies()
            ex.shutdown(wait=True, cancel_futures=True)
            _update_shared()

        return {
            "setup":             setup,
            "n_tasks_started":   n_done,
            "n_tasks_completed": n_ok,
            "n_tasks_failed":    n_fail,
        }

    # ─── workspace staging ────────────────────────────────────────────────
    def _stage_workspaces(self):
        """Returns (initial_specs, work_queue, stop_clone_event).

        initial_specs is the first batch of N specs ready to submit.
        work_queue is fed by a background thread that clones the remaining
        instances on demand.
        """
        workspace_root = self.spec["workspace_root"]
        initial = self.instances[:self.concurrency]
        rest    = self.instances[self.concurrency:]
        work_q: "_queue.Queue[dict | None]" = _queue.Queue()
        stop_clone = threading.Event()

        if self.no_clone:
            initial_specs = [
                {"instance": inst, "workdir": Path.cwd()} for inst in initial
            ]
            for inst in rest:
                work_q.put({"instance": inst, "workdir": Path.cwd()})
            return initial_specs, work_q, stop_clone

        # Clone the initial batch in parallel.
        initial_specs = []
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(CLONE_WORKERS, max(1, len(initial)))
        ) as cex:
            fmap = {cex.submit(setup_workspace, inst, workspace_root): inst
                    for inst in initial}
            for f in concurrent.futures.as_completed(fmap):
                inst = fmap[f]
                try:
                    wd = f.result()
                except Exception:
                    wd = Path.cwd()
                initial_specs.append({"instance": inst, "workdir": wd})

        # Background-clone the rest on demand and feed them into work_q.
        def _bg_clone(all_inst=rest or self.instances, q=work_q, stop=stop_clone):
            idx = 0
            with concurrent.futures.ThreadPoolExecutor(max_workers=CLONE_WORKERS) as cex:
                while not stop.is_set():
                    if not all_inst:
                        break
                    inst = all_inst[idx % len(all_inst)]
                    idx += 1
                    try:
                        wd = cex.submit(setup_workspace, inst, workspace_root).result()
                    except Exception:
                        wd = Path.cwd()
                    if stop.is_set():
                        break
                    q.put({"instance": inst, "workdir": wd})

        threading.Thread(target=_bg_clone, daemon=True).start()
        return initial_specs, work_q, stop_clone
