"""``Proxy`` — context manager that runs the reverse-proxy for one problem.

With ``capture=True`` it serves the reverse-proxy app (pipeline.proxy.app) on
a background uvicorn thread, in-process — no subprocess; otherwise it's a
no-op and claude talks to vLLM directly. Exposes ``base_url`` for claude-cli.
"""

from __future__ import annotations

import threading
from pathlib import Path

import uvicorn
from loguru import logger
from pipeline.utils.jsonl import instance_dir
from pipeline.proxy.app import ProxyApp
from pipeline.vllm_server import check_server_initialized

# Loopback host the proxy binds to and claude-cli connects back through.
PROXY_HOST = "127.0.0.1"
# Seconds to wait for the proxy thread to come up.
PROXY_READY_TIMEOUT_S = 10.0


class Proxy:
    """Serve the reverse proxy on enter (if capturing), stop it on exit.

    Exposes ``base_url``: the proxy when capturing, else the upstream vLLM
    unchanged."""

    def __init__(
        self,
        save_dir: Path,
        instance_id: str,
        *,
        url: str,
        proxy_port: int = 8001,
        capture: bool = False,
        raw: bool = False,
        upstream_health: bool = True,
    ) -> None:
        self.save_dir = save_dir
        self.instance_id = instance_id
        self.url = url
        self.proxy_port = proxy_port
        self.capture = capture
        self.raw = raw
        # Remote backends do not use this proxy; upstream_health is kept for
        # tests or controlled callers that want to skip the forwarded probe.
        self.upstream_health = upstream_health
        self.out_dir = save_dir / "telemetry"
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.base_url = url
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Proxy:
        if not self.capture:
            return self  # no proxy; claude talks to vLLM directly

        # Truncate any prior file for this id (retry-on-resume safety).
        if self.raw:
            idir = instance_dir(self.out_dir, self.instance_id)
            (idir / "vllm_traces.jsonl").unlink(missing_ok=True)
        app = ProxyApp(
            self.url,
            self.out_dir,
            self.instance_id,
            raw=self.raw,
        ).build()
        self._server = uvicorn.Server(
            uvicorn.Config(
                app,
                host=PROXY_HOST,
                port=self.proxy_port,
                log_level="warning",
                access_log=False,
            )
        )
        # uvicorn skips signal-handler setup off the main thread, so .run is
        # safe here; .should_exit (set in __exit__) drives graceful shutdown.
        self._thread = threading.Thread(
            target=self._server.run, name="proxy", daemon=True
        )
        self._thread.start()

        proxy_url = f"http://{PROXY_HOST}:{self.proxy_port}"
        # This readiness probe exercises the proxy and the forwarded upstream.
        # Tests can disable it when they provide their own upstream stub.
        if self.upstream_health and not check_server_initialized(
            f"{proxy_url}/v1/models", PROXY_READY_TIMEOUT_S
        ):
            self.__exit__(None, None, None)
            raise RuntimeError(f"proxy did not start on {proxy_url}")
        self.base_url = proxy_url
        logger.info("proxy serving at {} → {}", proxy_url, self.url)
        return self

    def __exit__(self, *exc) -> bool:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=15)
            self._thread = None
        self._server = None
        return False
