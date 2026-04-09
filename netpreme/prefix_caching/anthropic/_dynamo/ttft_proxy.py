#!/usr/bin/env python3
"""
TTFT measurement proxy.

Sits between Claude Code and Dynamo:

    Claude Code  →  ttft_proxy (:8001)  →  Dynamo (:8000)  →  vLLM worker

For every /v1/messages call it:
  - Forwards the request unchanged
  - Times the stream until the first content_block_delta event = TTFT
  - Reads new lines from the vLLM worker log to check for CPU→GPU loads
  - Appends one JSON record to ttft_log.jsonl

Usage:
    python3 ttft_proxy.py [--proxy-port 8001] [--dynamo-url http://127.0.0.1:8000]
                          [--log ttft_log.jsonl] [--worker-log /tmp/dynamo_worker.log]

Then point Claude Code at the proxy:
    ANTHROPIC_BASE_URL=http://localhost:8001 claude ...
"""

import argparse
import json
import re
import sys
import time
import threading
import urllib.request
import urllib.error
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ── shared state ──────────────────────────────────────────────────────

_log_lock   = threading.Lock()
_log_file   = "ttft_log.jsonl"
_dynamo_url = "http://127.0.0.1:8000"
_worker_log = "/tmp/dynamo_worker.log"

# Current task context — set externally via POST /x-benchmark-context
_task_context: dict = {"task_id": "", "turn": 0}
_ctx_lock = threading.Lock()

# Per-task turn counters — auto-incremented on every /v1/messages call
_turn_counters: dict[str, int] = {}
_turn_lock = threading.Lock()

# Worker log offset for delta CPU transfer reads
_wlog_offset = 0
_wlog_lock   = threading.Lock()


def _read_worker_log_delta() -> dict[str, float]:
    """Read new KV Transfer metric lines from the worker log since last call."""
    global _wlog_offset
    delta: dict[str, float] = {}
    try:
        with open(_worker_log, 'r', errors='replace') as f:
            with _wlog_lock:
                f.seek(0, 2)
                eof = f.tell()
                if _wlog_offset > eof:
                    _wlog_offset = 0
                f.seek(_wlog_offset)
                text = f.read()
                _wlog_offset = f.tell()
        for m in re.finditer(r'KV Transfer metrics:\s+(.+?)(?:\n|$)', text):
            for kv in m.group(1).split(','):
                kv = kv.strip()
                if '=' in kv:
                    k, v = kv.split('=', 1)
                    try:
                        delta[k.strip()] = delta.get(k.strip(), 0.0) + float(v.strip())
                    except ValueError:
                        pass
    except FileNotFoundError:
        pass
    return delta


def _append_log(entry: dict):
    with _log_lock:
        with open(_log_file, 'a') as f:
            f.write(json.dumps(entry) + '\n')


# ── request handler ───────────────────────────────────────────────────

class ProxyHandler(BaseHTTPRequestHandler):
    # Suppress default request logging — we do our own
    def log_message(self, fmt, *args):
        pass

    def _forward_headers(self) -> dict:
        skip = {'host', 'transfer-encoding', 'connection',
                'x-benchmark-task-id', 'x-benchmark-turn'}
        return {k: v for k, v in self.headers.items()
                if k.lower() not in skip}

    # ── benchmark context endpoint ─────────────────────────────────────
    def _handle_context(self, body: bytes):
        """POST /x-benchmark-context  {"task_id": "...", "turn": 0}"""
        try:
            ctx = json.loads(body)
            new_task_id = ctx.get('task_id', '')
            with _ctx_lock:
                _task_context.update(ctx)
            # Reset turn counter whenever a new task starts
            if new_task_id:
                with _turn_lock:
                    _turn_counters[new_task_id] = 0
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'ok')
        except Exception as e:
            self.send_error(400, str(e))

    # ── pass-through for non-messages endpoints ────────────────────────
    def _passthrough(self, method: str, body: bytes | None = None):
        try:
            req = urllib.request.Request(
                _dynamo_url + self.path,
                data=body,
                headers=self._forward_headers(),
                method=method,
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ('transfer-encoding', 'connection',
                                         'content-length'):
                        self.send_header(k, v)
                self.end_headers()
                self.wfile.write(resp.read())
        except urllib.error.HTTPError as e:
            self.send_error(e.code, str(e))
        except Exception as e:
            self.send_error(502, str(e))

    # ── main POST handler ──────────────────────────────────────────────
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body   = self.rfile.read(length) if length else b''

        # Internal benchmark context endpoint
        if self.path == '/x-benchmark-context':
            self._handle_context(body)
            return

        # Only instrument /v1/messages
        if not self.path.startswith('/v1/messages'):
            self._passthrough('POST', body)
            return

        try:
            payload    = json.loads(body)
            streaming  = payload.get('stream', False)
        except (json.JSONDecodeError, AttributeError):
            streaming  = False

        with _ctx_lock:
            task_id = _task_context.get('task_id', '')

        # Auto-increment turn counter per task_id
        with _turn_lock:
            _turn_counters.setdefault(task_id, 0)
            turn = _turn_counters[task_id]
            _turn_counters[task_id] += 1

        # Snapshot log offset before the request
        _read_worker_log_delta()   # advance offset only

        t_start       = time.perf_counter()
        ttft_ms       = None
        input_tokens  = None
        output_tokens = None
        first_block_type = None   # 'text' or 'tool_use'
        error         = None

        req = urllib.request.Request(
            _dynamo_url + self.path,
            data=body,
            headers=self._forward_headers(),
            method='POST',
        )

        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ('transfer-encoding', 'connection',
                                         'content-length'):
                        self.send_header(k, v)
                self.end_headers()

                if streaming:
                    for raw_line in resp:
                        # Forward immediately — don't buffer
                        self.wfile.write(raw_line)
                        self.wfile.flush()

                        line = raw_line.decode('utf-8', errors='replace').strip()
                        if not line.startswith('data:'):
                            continue
                        data_str = line[5:].strip()
                        if data_str in ('', '[DONE]'):
                            continue
                        try:
                            ev    = json.loads(data_str)
                            etype = ev.get('type', '')
                            if etype == 'content_block_start':
                                # Record whether first block is text or tool_use
                                if first_block_type is None:
                                    first_block_type = ev.get('content_block', {}).get('type', 'unknown')
                            elif etype == 'content_block_delta' and ttft_ms is None:
                                ttft_ms = (time.perf_counter() - t_start) * 1000
                            elif etype == 'message_start':
                                usage = ev.get('message', {}).get('usage', {})
                                if usage.get('input_tokens'):
                                    input_tokens = usage['input_tokens']
                                if usage.get('output_tokens'):
                                    output_tokens = usage['output_tokens']
                            elif etype == 'message_delta':
                                usage = ev.get('usage', {})
                                if usage.get('output_tokens'):
                                    output_tokens = usage['output_tokens']
                                if usage.get('input_tokens'):
                                    input_tokens = usage['input_tokens']
                        except (json.JSONDecodeError, AttributeError):
                            pass
                else:
                    content = resp.read()
                    self.wfile.write(content)
                    try:
                        rj           = json.loads(content)
                        usage        = rj.get('usage', {})
                        input_tokens  = usage.get('input_tokens')
                        output_tokens = usage.get('output_tokens')
                        # Identify first content block type
                        for blk in rj.get('content', []):
                            if first_block_type is None:
                                first_block_type = blk.get('type', 'unknown')
                        ttft_ms       = (time.perf_counter() - t_start) * 1000
                    except (json.JSONDecodeError, AttributeError):
                        pass

        except urllib.error.HTTPError as e:
            error = f"HTTP {e.code}"
            self.send_error(e.code, str(e))
        except Exception as e:
            error = str(e)
            self.send_error(502, str(e))

        total_ms = (time.perf_counter() - t_start) * 1000

        # Read CPU transfer delta that occurred during this request
        cpu_delta = _read_worker_log_delta()
        g2c = cpu_delta.get('GPU_to_CPU_total_bytes', 0.0)
        c2g = cpu_delta.get('CPU_to_GPU_total_bytes', 0.0)

        entry = {
            'ts':            time.time(),
            'task_id':       task_id,
            'turn':          turn,
            'ttft_ms':       round(ttft_ms, 1) if ttft_ms is not None else None,
            'total_ms':      round(total_ms, 1),
            'input_tokens':  input_tokens,
            'output_tokens': output_tokens,
            'first_block_type': first_block_type,
            'streaming':     streaming,
            'g2c_bytes':     g2c,          # GPU→CPU during this call
            'c2g_bytes':     c2g,          # CPU→GPU during this call (load overhead)
            'error':         error,
        }
        _append_log(entry)

        label    = f"TTFT {ttft_ms:.0f}ms" if ttft_ms else "no TTFT"
        cpu_tag  = f"  [C→G {c2g/1e6:.1f}MB]" if c2g > 0 else ""
        blk_tag  = f"  [{first_block_type}]" if first_block_type else ""
        print(f"  [{task_id or 'req'}] turn={turn}  {label}  "
              f"in={input_tokens} out={output_tokens}{blk_tag}{cpu_tag}",
              flush=True)

    def do_GET(self):
        self._passthrough('GET')

    def do_HEAD(self):
        self._passthrough('HEAD')


# ── main ──────────────────────────────────────────────────────────────

def main():
    global _log_file, _dynamo_url, _worker_log

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--proxy-port',  type=int, default=8001)
    parser.add_argument('--dynamo-url',  default='http://127.0.0.1:8000')
    parser.add_argument('--log',         default='ttft_log.jsonl')
    parser.add_argument('--worker-log',  default='/tmp/dynamo_worker.log')
    args = parser.parse_args()

    _dynamo_url = args.dynamo_url
    _log_file   = args.log
    _worker_log = args.worker_log

    # Initialise log offset to current end of worker log
    global _wlog_offset
    try:
        with open(_worker_log, 'r') as f:
            f.seek(0, 2)
            _wlog_offset = f.tell()
    except FileNotFoundError:
        _wlog_offset = 0

    # Verify Dynamo is reachable
    try:
        urllib.request.urlopen(f"{_dynamo_url}/health", timeout=5)
    except Exception:
        print(f"ERROR: Dynamo not reachable at {_dynamo_url}", file=sys.stderr)
        sys.exit(1)

    server = ThreadingHTTPServer(('0.0.0.0', args.proxy_port), ProxyHandler)
    print(f"TTFT proxy  :  localhost:{args.proxy_port}  →  {_dynamo_url}")
    print(f"Log file    :  {_log_file}")
    print(f"Worker log  :  {_worker_log}")
    print(f"Ready — point Claude Code at http://localhost:{args.proxy_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nProxy stopped.")


if __name__ == '__main__':
    main()
