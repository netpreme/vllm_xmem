"""Reverse proxy between claude-cli and vLLM — server *and* its controller.

This one file plays two roles:

  * ``class Proxy`` — a context manager used in-process by the runner. It
    starts the per-turn telemetry subprocesses around one problem (the
    metrics_watcher, and — with capture=True — this proxy server) and
    tears them down on exit. ``with Proxy(...) as proxy`` → proxy.base_url.

  * a runnable reverse-proxy server — when launched as
    ``python proxy.py --upstream ... --listen-port ...`` (which is exactly
    what ``Proxy`` spawns). For each `POST /v1/messages` it parses the JSON
    request + SSE response and appends one RAW row to
    `<out_dir>/<instance_id>.proxy.jsonl`; other paths stream through.

Derivations (`agent` main/sub, category, …) are NOT done here — they live
in the analysis layer. The active `instance_id` is read from
`--control-file`, which ``Proxy`` rewrites per problem.

Server code layout:
    1. Request parsing  — pure functions over the JSON request body.
    2. SSE parsing      — walks the response stream, extracts content blocks.
    3. Reverse proxy    — forwards request, tees the response, writes the row.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import httpx
import uvicorn
from loguru import logger
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route

PIPELINE = Path(__file__).resolve().parent


# ===========================================================================
# 0. Proxy — context manager that runs the telemetry subprocesses for one
#    problem (the watcher + this proxy server). Used in-process by the runner.
# ===========================================================================


class Proxy:
    """Start/stop the watcher (+ optional reverse proxy) around one problem."""

    def __init__(
        self,
        save_dir: Path,
        instance_id: str,
        *,
        vllm_url: str,
        proxy_port: int = 8001,
        capture: bool = False,
    ) -> None:
        self.save_dir = save_dir
        self.instance_id = instance_id
        self.vllm_url = vllm_url
        self.proxy_port = proxy_port
        self.capture = capture

        # The watcher and proxy streams currently land in the same
        # per-problem directory (joined by turn index in analysis); the
        # separate attributes let callers split them later without a
        # signature change.
        self.per_problem_dir = save_dir / "per_problem"
        self.per_problem_dir.mkdir(parents=True, exist_ok=True)
        self.watcher_dir = self.per_problem_dir
        self.proxy_dir = self.per_problem_dir
        self.control = save_dir / ".active_instance"

        self.base_url = vllm_url
        self._watcher: tuple | None = None
        self._proxy: tuple | None = None

    def __enter__(self) -> Proxy:
        # Imported lazily so this module still runs as a bare `python
        # proxy.py` server (where the `pipeline` package isn't importable).
        from pipeline import http_utils

        self.control.write_text(self.instance_id)
        # Truncate any prior trace files for this id (retry-on-resume safety).
        for suffix in (".vllm.jsonl", ".proxy.jsonl"):
            (self.per_problem_dir / f"{self.instance_id}{suffix}").unlink(
                missing_ok=True
            )

        if self.capture:
            self._proxy = self._start(
                "proxy",
                [
                    sys.executable,
                    str(PIPELINE / "proxy.py"),
                    "--upstream",
                    self.vllm_url,
                    "--control-file",
                    str(self.control),
                    "--out-dir",
                    str(self.proxy_dir),
                    "--listen-port",
                    str(self.proxy_port),
                ],
                self.save_dir / ".proxy.log",
            )
            if not http_utils.check_server_initialized(
                f"http://127.0.0.1:{self.proxy_port}/v1/models", 10.0
            ):
                self.__exit__(None, None, None)
                raise RuntimeError(
                    f"proxy did not start; see {self.save_dir}/.proxy.log"
                )
            self.base_url = f"http://127.0.0.1:{self.proxy_port}"

        self._watcher = self._start(
            "metrics_watcher",
            [
                sys.executable,
                str(PIPELINE / "metrics_watcher.py"),
                "--vllm-url",
                self.vllm_url,
                "--control-file",
                str(self.control),
                "--out-dir",
                str(self.watcher_dir),
            ],
            self.save_dir / ".metrics_watcher.log",
        )
        return self

    def __exit__(self, *exc) -> bool:
        if self._watcher is not None:
            self._stop("metrics_watcher", self._watcher)
            self._watcher = None
        if self._proxy is not None:
            self._stop("proxy", self._proxy)
            self._proxy = None
        self.control.write_text("")
        return False

    @staticmethod
    def _start(name: str, argv: list[str], log_path: Path) -> tuple:
        """Launch `argv` in its own session, stdout/stderr → log_path."""
        log = log_path.open("w")
        proc = subprocess.Popen(
            argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        logger.info("started {} (pid {})", name, proc.pid)
        return proc, log

    @staticmethod
    def _stop(name: str, sidecar: tuple) -> None:
        """SIGTERM then wait; SIGKILL if it doesn't exit in 10 s."""
        proc, log = sidecar
        if proc.poll() is None:
            logger.info("stopping {} (pid {})", name, proc.pid)
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        log.close()

# ===========================================================================
# 1. Request parsing — pure functions over the request JSON.
# ===========================================================================


@dataclass(frozen=True)
class ParsedRequest:
    system_prompt_chars: int
    num_tool_defs: int
    num_messages: int


def _system_prompt_text(body: dict) -> str:
    """Anthropic accepts `system` as either str or a list of content blocks."""
    system = body.get("system")
    if isinstance(system, str):
        return system
    if isinstance(system, list):
        return "".join(
            b.get("text") or ""
            for b in system
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def parse_request(body: dict) -> ParsedRequest:
    return ParsedRequest(
        system_prompt_chars=len(_system_prompt_text(body)),
        num_tool_defs=len(body.get("tools") or []),
        num_messages=len(body.get("messages") or []),
    )


# ===========================================================================
# 2. SSE parsing — walk the response stream.
# ===========================================================================
#
# Anthropic's streaming format:
#     event: message_start         data: {... usage, model, ...}
#     event: content_block_start   data: {type, content_block: {type, ...}}
#     event: content_block_delta   data: {type, delta: {type, text | partial_json}}
#     event: content_block_stop    data: {...}
#     event: message_delta         data: {delta: {stop_reason, ...}, usage}
#     event: message_stop          data: {...}


@dataclass
class ParsedResponse:
    num_tool_calls: int = 0
    tool_names: list[str] = field(default_factory=list)
    has_thinking: bool = False
    response_text_chars: int = 0
    claude_stop_reason: str = ""


def parse_sse_response(body: bytes) -> ParsedResponse:
    r = ParsedResponse()
    text = body.decode("utf-8", errors="replace")
    for record in text.split("\n\n"):
        data = ""
        for line in record.splitlines():
            if line.startswith("data:"):
                data = line[5:].strip()
                break
        if not data or data == "[DONE]":
            continue
        try:
            ev = json.loads(data)
        except json.JSONDecodeError:
            continue
        t = ev.get("type")
        if t == "content_block_start":
            block = ev.get("content_block") or {}
            btype = block.get("type")
            if btype == "tool_use":
                r.num_tool_calls += 1
                if block.get("name"):
                    r.tool_names.append(block["name"])
            elif btype == "thinking":
                r.has_thinking = True
        elif t == "content_block_delta":
            delta = ev.get("delta") or {}
            if delta.get("type") == "text_delta":
                r.response_text_chars += len(delta.get("text") or "")
        elif t == "message_delta":
            stop = (ev.get("delta") or {}).get("stop_reason")
            if stop:
                r.claude_stop_reason = stop
    return r


# ===========================================================================
# 3. Reverse proxy.
# ===========================================================================

_HOP_BY_HOP_HEADERS: frozenset[str] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)


def _strip_hop_by_hop(headers: Iterable[tuple[str, str]]) -> dict[str, str]:
    return {k: v for k, v in headers if k.lower() not in _HOP_BY_HOP_HEADERS}


_SAFE_FILE_RE = re.compile(r"[^A-Za-z0-9_\-.]")


def safe_filename(s: str) -> str:
    return _SAFE_FILE_RE.sub("_", s)[:200]


class PerProblemJSONL:
    """Append-only writer, one file per instance_id in `out_dir`."""

    def __init__(self, out_dir: Path) -> None:
        self._dir = out_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def write(self, instance_id: str, row: dict) -> None:
        if not instance_id:
            return
        path = self._dir / f"{safe_filename(instance_id)}.proxy.jsonl"
        with path.open("a") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def read_active_instance(control_file: Path) -> str:
    try:
        return control_file.read_text().strip()
    except FileNotFoundError:
        return ""


async def _proxy_messages(
    client: httpx.AsyncClient,
    upstream: str,
    request: Request,
    raw_body: bytes,
    writer: PerProblemJSONL,
    control_file: Path,
) -> Response:
    """Forward a /v1/messages POST; buffer the response so we can parse
    it; write the combined request+response row; return the buffered
    response to claude-cli.

    Buffering breaks "live" streaming to claude-cli, but at concurrency=1
    that's invisible — claude still parses the SSE chunks the same way."""
    ts = time.time()
    try:
        parsed_req = parse_request(json.loads(raw_body))
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("request parse error: {!r}", exc)
        parsed_req = ParsedRequest(
            system_prompt_chars=0, num_tool_defs=0, num_messages=0
        )

    target = f"{upstream.rstrip('/')}{request.url.path}"
    if request.url.query:
        target = f"{target}?{request.url.query}"
    upstream_resp = await client.request(
        request.method,
        target,
        content=raw_body,
        headers=_strip_hop_by_hop(request.headers.items()),
    )

    parsed_resp = parse_sse_response(upstream_resp.content)
    instance_id = read_active_instance(control_file)
    writer.write(
        instance_id,
        {
            "ts": round(ts, 3),
            "system_prompt_chars": parsed_req.system_prompt_chars,
            "num_tool_defs": parsed_req.num_tool_defs,
            "num_messages": parsed_req.num_messages,
            "num_tool_calls": parsed_resp.num_tool_calls,
            "tool_names": parsed_resp.tool_names,
            "has_thinking": parsed_resp.has_thinking,
            "response_text_chars": parsed_resp.response_text_chars,
            "claude_stop_reason": parsed_resp.claude_stop_reason,
        },
    )

    return Response(
        content=upstream_resp.content,
        status_code=upstream_resp.status_code,
        headers=_strip_hop_by_hop(upstream_resp.headers.items()),
    )


async def _proxy_passthrough(
    client: httpx.AsyncClient,
    upstream: str,
    request: Request,
    raw_body: bytes,
) -> Response:
    """Stream non-/v1/messages requests through unchanged (e.g. /v1/models)."""
    target = f"{upstream.rstrip('/')}{request.url.path}"
    if request.url.query:
        target = f"{target}?{request.url.query}"
    upstream_req = client.build_request(
        request.method,
        target,
        content=raw_body,
        headers=_strip_hop_by_hop(request.headers.items()),
    )
    upstream_resp = await client.send(upstream_req, stream=True)
    return StreamingResponse(
        upstream_resp.aiter_raw(),
        status_code=upstream_resp.status_code,
        headers=_strip_hop_by_hop(upstream_resp.headers.items()),
        background=BackgroundTask(upstream_resp.aclose),
    )


def build_app(upstream: str, out_dir: Path, control_file: Path) -> Starlette:
    writer = PerProblemJSONL(out_dir)

    @asynccontextmanager
    async def lifespan(app: Starlette):
        async with httpx.AsyncClient(timeout=None) as client:
            app.state.client = client
            yield

    async def handler(request: Request) -> Response:
        raw_body = await request.body()
        client = request.app.state.client
        if request.method == "POST" and request.url.path == "/v1/messages":
            return await _proxy_messages(
                client, upstream, request, raw_body, writer, control_file
            )
        return await _proxy_passthrough(client, upstream, request, raw_body)

    return Starlette(
        routes=[
            Route(
                "/{path:path}",
                endpoint=handler,
                methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD"],
            )
        ],
        lifespan=lifespan,
    )


# ===========================================================================
# CLI.
# ===========================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", required=True)
    parser.add_argument(
        "--control-file",
        required=True,
        type=Path,
        help="text file that coding_agent.py updates with the current instance_id",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="directory to write per-problem <instance_id>.proxy.jsonl into",
    )
    parser.add_argument("--listen-host", default="127.0.0.1")
    parser.add_argument("--listen-port", type=int, default=8001)
    args = parser.parse_args()

    app = build_app(args.upstream, args.out_dir, args.control_file)
    config = uvicorn.Config(
        app,
        host=args.listen_host,
        port=args.listen_port,
        log_level="warning",
        access_log=False,
    )
    try:
        asyncio.run(uvicorn.Server(config).serve())
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
