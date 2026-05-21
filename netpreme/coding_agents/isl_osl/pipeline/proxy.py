"""Logging proxy for Anthropic /v1/messages.

Sits between `claude -p` and the upstream model server (local vLLM or
api.anthropic.com). Forwards SSE bytes verbatim while in parallel parsing
each response to extract per-turn ISL/OSL/timing. One CSV row + one JSONL
body line per /v1/messages call, keyed by the X-Instance-Id header.

  ┌─────────┐   POST /v1/messages   ┌────────┐   forwarded   ┌──────────┐
  │ claude  │ ───────────────────▶  │ proxy  │ ────────────▶ │ upstream │
  │         │ ◀─── SSE bytes ─────  │        │ ◀─── SSE ──── │          │
  └─────────┘   (unchanged)         └────┬───┘               └──────────┘
                                         │
                                         ├▶ <csv_dir>/<id>.csv     (per-turn metrics)
                                         └▶ <transcripts_dir>/<id>.jsonl (full request+response)

Claude always streams; the unary path isn't implemented.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse


# ---------------------------------------------------------------- metrics --

CSV_COLUMNS = [
    "ts", "instance_id", "elapsed_ms",
    "ttft_ms", "decode_ms", "itl_ms",
    "isl", "osl", "isl_new", "isl_cached", "cache_hit_rate",
    "stop_reason", "category", "num_tool_calls",
]


def categorize(content: list[dict[str, Any]]) -> str:
    """Classify an assistant turn by which kinds of blocks it emitted."""
    has_text = any(b.get("type") == "text" and (b.get("text") or "").strip()
                   for b in content)
    has_tool = any(b.get("type") == "tool_use" for b in content)
    if has_text and has_tool: return "mixed"
    if has_tool:              return "tool_only"
    if has_text:              return "text_only"
    return "empty"


def derive_metrics(usage: dict[str, Any]) -> dict[str, Any]:
    """Turn raw Anthropic `usage` into ISL / OSL / cache breakdown."""
    inp    = int(usage.get("input_tokens") or 0)
    create = int(usage.get("cache_creation_input_tokens") or 0)
    read   = int(usage.get("cache_read_input_tokens") or 0)
    out    = int(usage.get("output_tokens") or 0)
    isl    = inp + create + read
    return {
        "isl":            isl,
        "osl":            out,
        "isl_new":        inp + create,
        "isl_cached":     read,
        "cache_hit_rate": round(read / isl, 4) if isl > 0 else 0.0,
    }


# --------------------------------------------------------------- SSE parse --

class SSEParser:
    """Buffer an Anthropic SSE response, then walk events to extract the
    final `usage` and the assistant `content` blocks.

    Anthropic emits, in order:
      message_start          — initial usage estimate
      content_block_start / delta / stop  — one set per output block
                                            (text_delta or input_json_delta)
      message_delta          — final usage corrections (cache_read populated)
      message_stop
    """

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> None:
        self._buf.extend(chunk)

    def parse(self) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        usage: dict[str, Any] = {}
        content: list[dict[str, Any]] = []
        for ev in self._events():
            self._apply(ev, usage, content)
        self._finalize_blocks(content)
        return usage, content

    def _events(self):
        for raw in self._buf.split(b"\n\n"):
            for ln in raw.splitlines():
                if ln.startswith(b"data:"):
                    try:
                        yield json.loads(ln[5:].strip())
                    except json.JSONDecodeError:
                        pass
                    break

    def _apply(self, ev, usage, content):
        t = ev.get("type")
        if t == "message_start":
            u = (ev.get("message") or {}).get("usage") or {}
            for k in ("input_tokens", "cache_creation_input_tokens",
                      "cache_read_input_tokens", "output_tokens"):
                usage[k] = int(u.get(k) or 0)
        elif t == "content_block_start":
            i = ev.get("index", 0)
            while len(content) <= i:
                content.append({})
            content[i] = {**(ev.get("content_block") or {}),
                          "_text": "", "_json": ""}
        elif t == "content_block_delta":
            i = ev.get("index", 0)
            if i >= len(content): return
            d = ev.get("delta") or {}
            if d.get("type") == "text_delta":
                content[i]["_text"] += d.get("text") or ""
            elif d.get("type") == "input_json_delta":
                content[i]["_json"] += d.get("partial_json") or ""
        elif t == "message_delta":
            u = ev.get("usage") or {}
            for k in ("input_tokens", "cache_creation_input_tokens",
                      "cache_read_input_tokens", "output_tokens"):
                if k in u: usage[k] = int(u.get(k) or 0)
            sr = (ev.get("delta") or {}).get("stop_reason")
            if sr: usage["stop_reason"] = sr

    @staticmethod
    def _finalize_blocks(content):
        for blk in content:
            text = blk.pop("_text", None)
            raw_json = blk.pop("_json", None)
            if blk.get("type") == "text" and text is not None:
                blk["text"] = text
            elif blk.get("type") == "tool_use" and raw_json is not None:
                try:
                    blk["input"] = json.loads(raw_json) if raw_json else {}
                except json.JSONDecodeError:
                    blk["input"] = {"__raw": raw_json}


# ------------------------------------------------------------ turn logger --

@dataclass
class TurnRecord:
    ts: float
    instance_id: str | None
    elapsed_ms: int
    ttft_ms: int
    decode_ms: int
    itl_ms: float | None
    usage: dict[str, Any]
    content: list[dict[str, Any]]
    request: dict[str, Any] | None


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in s)[:200]


class TurnLogger:
    """Append per-turn rows to <csv_dir>/<id>.csv and full text to
    <transcripts_dir>/<id>.jsonl. Async-locked so concurrent requests don't
    interleave bytes inside a single file."""

    def __init__(self, csv_dir: Path | None, transcripts_dir: Path | None) -> None:
        self._csv_dir = csv_dir
        self._transcripts_dir = transcripts_dir
        self._lock = asyncio.Lock()

    async def write(self, rec: TurnRecord) -> None:
        if not rec.instance_id: return
        async with self._lock:
            self._write_csv(rec)
            self._write_body(rec)

    def _write_csv(self, rec: TurnRecord) -> None:
        if self._csv_dir is None: return
        row = {
            "ts":             rec.ts,
            "instance_id":    rec.instance_id,
            "elapsed_ms":     rec.elapsed_ms,
            "ttft_ms":        rec.ttft_ms,
            "decode_ms":      rec.decode_ms,
            "itl_ms":         rec.itl_ms,
            "stop_reason":    rec.usage.get("stop_reason") or "",
            "category":       categorize(rec.content),
            "num_tool_calls": sum(1 for b in rec.content if b.get("type") == "tool_use"),
            **derive_metrics(rec.usage),
        }
        path = self._csv_dir / f"{_safe(rec.instance_id)}.csv"
        new = not path.exists() or path.stat().st_size == 0
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
            if new: w.writeheader()
            w.writerow({k: row.get(k) for k in CSV_COLUMNS})

    def _write_body(self, rec: TurnRecord) -> None:
        if self._transcripts_dir is None or rec.request is None: return
        messages = rec.request.get("messages") or []
        payload = {
            "ts": rec.ts,
            "instance_id": rec.instance_id,
            "usage": {
                **derive_metrics(rec.usage),
                "category":       categorize(rec.content),
                "stop_reason":    rec.usage.get("stop_reason"),
                "num_tool_calls": sum(1 for b in rec.content
                                       if b.get("type") == "tool_use"),
                "ttft_ms":   rec.ttft_ms,
                "decode_ms": rec.decode_ms,
                "itl_ms":    rec.itl_ms,
            },
            "request": {
                "system":   rec.request.get("system"),
                "tools":    rec.request.get("tools"),
                "messages": messages,
            },
            "isl_new_text": messages[-1] if messages else None,
            "response": {"content": rec.content,
                         "stop_reason": rec.usage.get("stop_reason")},
        }
        path = self._transcripts_dir / f"{_safe(rec.instance_id)}.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


# -------------------------------------------------------------- proxy app --

@dataclass
class Config:
    upstream: str
    port: int = 9001
    csv_dir: Path | None = None
    transcripts_dir: Path | None = None
    max_tokens_cap: int = 0          # 0 disables the clamp
    passthrough_auth: bool = False   # forward client Authorization/x-api-key


def _forward_headers(req: Request, cfg: Config) -> dict[str, str]:
    """Strip hop-by-hop headers (and auth, unless passthrough_auth=True).
    Force identity encoding so we can read SSE bytes verbatim."""
    strip = {"host", "content-length", "accept-encoding"}
    if not cfg.passthrough_auth:
        strip |= {"x-api-key", "authorization"}
    out = {k: v for k, v in req.headers.items() if k.lower() not in strip}
    out["accept-encoding"] = "identity"
    return out


def _clamp_max_tokens(body: bytes, cap: int) -> tuple[bytes, dict[str, Any] | None]:
    """Decode the JSON body; if max_tokens exceeds cap, rewrite the body.
    Returns (possibly-rewritten body, parsed-request-dict-or-None)."""
    if not body: return body, None
    try:
        req = json.loads(body)
    except json.JSONDecodeError:
        return body, None
    if cap and int(req.get("max_tokens") or 0) > cap:
        req["max_tokens"] = cap
        body = json.dumps(req).encode()
    return body, req


def build_app(cfg: Config) -> FastAPI:
    logger = TurnLogger(cfg.csv_dir, cfg.transcripts_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.client = httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0))
        try: yield
        finally: await app.state.client.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"ok": True, "upstream": cfg.upstream}

    @app.post("/v1/messages")
    async def messages(request: Request):
        body, req_obj = _clamp_max_tokens(await request.body(),
                                          cfg.max_tokens_cap)
        headers = _forward_headers(request, cfg)
        headers["content-length"] = str(len(body))

        instance_id = request.headers.get("x-instance-id")
        started = time.time()
        client: httpx.AsyncClient = request.app.state.client

        upstream_req = client.build_request(
            "POST", f"{cfg.upstream}/v1/messages",
            content=body, headers=headers,
        )
        upstream_resp = await client.send(upstream_req, stream=True)

        resp_headers = {k: v for k, v in upstream_resp.headers.items()
                        if k.lower() not in ("content-length", "transfer-encoding")}
        media_type = upstream_resp.headers.get("content-type", "text/event-stream")

        async def gen():
            parser = SSEParser()
            first_chunk_ts: float | None = None
            first_delta_ts: float | None = None
            try:
                async for chunk in upstream_resp.aiter_raw():
                    now = time.time()
                    if chunk:
                        first_chunk_ts = first_chunk_ts or now
                        if first_delta_ts is None and b"content_block_delta" in chunk:
                            first_delta_ts = now
                    parser.feed(chunk)
                    yield chunk
            finally:
                await upstream_resp.aclose()
                end_ts = time.time()
                try:
                    usage, content = parser.parse()
                    t0 = first_delta_ts or first_chunk_ts or end_ts
                    out_tok = int(usage.get("output_tokens") or 0)
                    decode_ms = int((end_ts - t0) * 1000) if first_chunk_ts else 0
                    await logger.write(TurnRecord(
                        ts=started, instance_id=instance_id,
                        elapsed_ms=int((end_ts - started) * 1000),
                        ttft_ms=int((t0 - started) * 1000),
                        decode_ms=decode_ms,
                        itl_ms=(round(decode_ms / max(1, out_tok - 1), 3)
                                if out_tok > 1 else None),
                        usage=usage, content=content, request=req_obj,
                    ))
                except Exception as e:
                    print(f"[proxy] log error: {e!r}", file=sys.stderr, flush=True)

        return StreamingResponse(gen(),
                                 status_code=upstream_resp.status_code,
                                 headers=resp_headers,
                                 media_type=media_type)

    return app


# -------------------------------------------------------------------- CLI --

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream",            required=True,
                    help="e.g. http://localhost:8000 or https://api.anthropic.com")
    ap.add_argument("--port",                type=int, default=9001)
    ap.add_argument("--per-problem-csv-dir", type=Path, default=None)
    ap.add_argument("--dump-transcripts-dir",     type=Path, default=None)
    ap.add_argument("--max-tokens-cap",      type=int, default=4096,
                    help="clamp client's max_tokens; 0 disables")
    ap.add_argument("--passthrough-auth",    action="store_true",
                    help="forward Authorization / x-api-key headers (needed "
                         "when upstream is api.anthropic.com with OAuth)")
    args = ap.parse_args()

    cfg = Config(
        upstream=args.upstream.rstrip("/"),
        port=args.port,
        csv_dir=args.per_problem_csv_dir,
        transcripts_dir=args.dump_transcripts_dir,
        max_tokens_cap=max(0, args.max_tokens_cap),
        passthrough_auth=args.passthrough_auth,
    )
    if cfg.csv_dir:    cfg.csv_dir.mkdir(parents=True, exist_ok=True)
    if cfg.transcripts_dir: cfg.transcripts_dir.mkdir(parents=True, exist_ok=True)

    uvicorn.run(build_app(cfg), host="127.0.0.1", port=cfg.port,
                log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
