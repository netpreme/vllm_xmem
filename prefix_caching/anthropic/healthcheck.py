#!/usr/bin/env python3
"""Poll GET /health until the vLLM server is ready."""

import argparse
import sys
import time
import urllib.request
import urllib.error


def wait_for_server(base_url: str, timeout: int, interval: float) -> None:
    url = f"{base_url}/health"
    deadline = time.monotonic() + timeout
    attempt = 0

    print(f"Waiting for vLLM server at {url} ...")
    while time.monotonic() < deadline:
        attempt += 1
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                if resp.status == 200:
                    print(f"Server is ready (attempt {attempt}).")
                    return
        except (urllib.error.URLError, OSError):
            pass

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        print(f"  attempt {attempt}: not ready, retrying in {interval}s "
              f"({remaining:.0f}s remaining) ...", flush=True)
        time.sleep(interval)

    print(f"ERROR: server did not become ready within {timeout}s.", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poll vLLM /health until ready.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000",
                        help="Server base URL (default: http://127.0.0.1:8000)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Maximum seconds to wait (default: 300)")
    parser.add_argument("--interval", type=float, default=5.0,
                        help="Seconds between attempts (default: 5)")
    args = parser.parse_args()

    wait_for_server(args.base_url, args.timeout, args.interval)


if __name__ == "__main__":
    main()
