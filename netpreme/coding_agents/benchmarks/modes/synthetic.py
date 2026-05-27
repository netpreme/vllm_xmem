"""Synthetic trace capture — write a fabricated /v1/messages trace with
controlled (ISL, ISL_new, OSL) per turn. No agents, no GPU. The on-disk
format matches a real capture so ``benchmark.sh --from-trace`` works.

Used when ``benchmark.sh --save-trace --isl N --isl-new K [--osl M] [--n-turns T] [--n-sessions S]``.

Pattern (growing-ISL conversation):
  * Turn 0: messages = [user_0]            → ISL ≈ isl (the initial target)
  * Turn n>0: messages = [..., user_{n-1}, asst_{n-1}, user_n]
              ISL grows by ≈ isl_new + osl tokens per turn.
              ISL_new per turn ≈ isl_new (only the new user message is uncached
              when the conversation prefix is HBM-cache-hit).

The model's actual response is irrelevant — we record a placeholder
assistant message that just provides isl_new + osl tokens of "history"
for the next turn's prefix. ``--deterministic`` at replay time pins OSL
exactly via ``min_tokens``+``ignore_eos``.
"""
from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path

from transformers import AutoTokenizer

MODEL = "qwen/qwen3-coder-30b-a3b-instruct-fp8"
# Save under the standard results dir so the chain/find-latest convention works.
DEFAULT_OUT_ROOT = (Path(__file__).resolve().parent.parent / "results_benchmarks")


def _random_text(rng: random.Random, n_tokens: int, tok) -> str:
    """Generate text that tokenizes to ~n_tokens tokens (±~5% drift)."""
    SAFE_LO, SAFE_HI = 1024, 100000
    ids = [rng.randrange(SAFE_LO, SAFE_HI) for _ in range(max(n_tokens, 1))]
    return tok.decode(ids, skip_special_tokens=True)


def _gen_session(sid_idx: int, isl: int, isl_new: int, osl: int,
                 n_turns: int, master_seed: int, tok) -> tuple[str, list[dict]]:
    rng = random.Random(master_seed * 100_000 + sid_idx)
    instance_id = f"synth_isln{isl_new}_seed{master_seed:05d}_s{sid_idx:03d}"

    # First turn: a small system block + a bulky user message totalling ~isl tokens.
    sys_tokens = 200
    init_user_tokens = max(isl - sys_tokens, 100)
    system_text = _random_text(rng, sys_tokens, tok)
    initial_user_text = _random_text(rng, init_user_tokens, tok)

    messages: list[dict] = []
    turns: list[dict] = []
    cumulative_isl = sys_tokens + init_user_tokens

    for turn_idx in range(n_turns):
        if turn_idx == 0:
            user_content = initial_user_text
        else:
            user_content = _random_text(rng, isl_new, tok)
            cumulative_isl += isl_new + osl   # prior assistant response also in history

        messages.append({"role": "user", "content": user_content})

        request_body = {
            "model": MODEL,
            "system": system_text,
            "messages": [dict(m) for m in messages],   # shallow copy
            "max_tokens": osl,
            "stream": True,
        }
        turns.append({
            "t_request": round(turn_idx * 5.0, 4),      # 5-second nominal spacing
            "request_body": request_body,
            "expected_isl":     cumulative_isl,
            "expected_isl_new": init_user_tokens if turn_idx == 0 else (isl_new + osl),
            "expected_osl":     osl,
        })

        # Append fake assistant turn (osl tokens) into history for next turn.
        messages.append({"role": "assistant", "content": _random_text(rng, osl, tok)})

    return instance_id, turns


def gen_synthetic_capture(*, isl: int, isl_new: int, osl: int,
                          n_turns: int, n_sessions: int, seed: int = 42,
                          out_root: Path | None = None) -> Path:
    """Generate a synthetic trace capture and return the output directory.

    Layout:
      <out_root>/bench_sweep_synth_<ts>_isln<K>_osl<M>/c<n_sessions:03d>/
          capture_meta.json
          sessions.jsonl
          traces/<instance>.jsonl
    """
    if out_root is None:
        out_root = DEFAULT_OUT_ROOT
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_root) / f"bench_sweep_synth_{ts}_isln{isl_new}_osl{osl}"
    level_dir = run_dir / f"c{n_sessions:03d}"
    level_dir.mkdir(parents=True, exist_ok=True)
    (level_dir / "traces").mkdir(exist_ok=True)

    print(f"[synthetic] loading tokenizer {MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL)
    print(f"[synthetic] generating {n_sessions} sessions × {n_turns} turns "
          f"(ISL={isl}, ISL_new={isl_new}, OSL={osl})", flush=True)

    t_wall_start_base = time.time()
    sessions_jsonl_rows: list[dict] = []

    for sid_idx in range(n_sessions):
        instance_id, turns = _gen_session(sid_idx, isl, isl_new, osl,
                                          n_turns, seed, tok)
        trace_file = f"{instance_id}.jsonl"
        trace_path = level_dir / "traces" / trace_file
        t_sess_wall = t_wall_start_base + sid_idx * 0.001
        sessions_jsonl_rows.append({
            "instance_id":     instance_id,
            "trace_file":      trace_file,
            "t_session_start": round(sid_idx * 0.001, 4),
            "workdir":         instance_id,
            "t_wall_start":    t_sess_wall,
        })
        with open(trace_path, "w") as f:
            f.write(json.dumps({
                "kind": "meta", "session_id": instance_id,
                "upstream": "http://localhost:8001", "t_wall_start": t_sess_wall,
            }) + "\n")
            for t in turns:
                f.write(json.dumps({
                    "kind": "turn",
                    "t_request": t["t_request"],
                    "t_response_end": round(t["t_request"] + 5.0, 4),
                    "method": "POST", "path": "/v1/messages", "status": 200,
                    "request": t["request_body"], "response_sse": "",
                    "input_tokens":  t["expected_isl"],
                    "output_tokens": t["expected_osl"],
                }) + "\n")
        if (sid_idx + 1) % 10 == 0:
            print(f"  [{sid_idx + 1}/{n_sessions}] sessions written", flush=True)

    with open(level_dir / "sessions.jsonl", "w") as f:
        for r in sessions_jsonl_rows:
            f.write(json.dumps(r) + "\n")

    (level_dir / "capture_meta.json").write_text(json.dumps({
        "kind": "capture_level",
        "concurrency": n_sessions,
        "model": MODEL,
        "setup": "synthetic",
        "vllm_port": 8001,
        "gpus": "0",
        "sustained_mins": 0.0,
        "n_sessions": n_sessions,
        "t_level_start_unix": t_wall_start_base,
        "t_level_end_unix":   t_wall_start_base + n_turns * 5.0,
        "duration_s":         n_turns * 5.0,
        "determinism": {"VLLM_BATCH_INVARIANT": 0, "seed": seed, "temperature": 0},
        "sweep_run_dir": str(level_dir),
        "synthetic_params": {
            "isl": isl, "isl_new": isl_new, "osl": osl,
            "n_turns": n_turns, "n_sessions": n_sessions, "seed": seed,
            "pattern": "growing-ISL conversation",
        },
    }, indent=2))

    return level_dir
