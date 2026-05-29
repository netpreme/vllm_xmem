"""Reverse proxy between claude-cli and vLLM (in-process, per problem).

Public API:
    Proxy   context manager; serves the reverse-proxy on a uvicorn thread
            when capturing, else a no-op. ``with Proxy(...) as proxy`` →
            proxy.base_url (the proxy, or upstream vLLM unchanged).
"""

from pipeline.proxy.controller import Proxy

__all__ = ["Proxy"]
