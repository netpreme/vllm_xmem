"""Tiny HTTP helpers used by coding_agent.py.

We deliberately use stdlib `urllib` rather than `requests` to avoid an
extra dependency for a couple of one-shot calls.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request


def check_server_initialized(url: str, timeout: float) -> bool:
    """Poll `url` until it returns a 2xx (i.e. the service is up), or
    `timeout` in seconds elapses."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1.0) as r:
                if 200 <= r.status < 300:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.1)
    return False


def get_model_name(url: str) -> str:
    """Ask vLLM which model it's serving — that's what claude-cli sends."""
    with urllib.request.urlopen(f"{url}/v1/models", timeout=2.0) as r:
        return json.loads(r.read())["data"][0]["id"]
