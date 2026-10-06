#!/usr/bin/env python3
"""Locate the serialization: is a turn slow because of its tokens, or itself?

Prefill cost scales with prompt length; queueing and per-request overhead do
not. Sending a deliberately tiny prompt strips the first away, so whatever
latency remains is the fixed cost of getting one request through the gateway
and the sampler. Sweeping concurrency on top of that says whether those fixed
costs overlap or stack.

If a short request is already slow, the conversation length was never the
problem. If wall time grows in step with concurrency, requests are being
served one at a time somewhere.

  python3 probe_concurrency.py <base_url> [max_concurrency]
"""

from __future__ import annotations

import json
import sys
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

MODEL = "Qwen/Qwen3.5-4B"


def one(url: str, nonce: str) -> float:
    body = {
        "model": MODEL,
        # Unique so nothing is served from cache, but far too short to prefill.
        "messages": [{"role": "user", "content": f"say ok ({nonce})"}],
        "max_tokens": 1,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer probe-{nonce}",
        },
    )
    t = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        r.read()
    return time.time() - t


def main() -> int:
    base = sys.argv[1].rstrip("/")
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    url = f"{base}/chat/completions"

    print(f"{'conc':>5}  {'wall':>8}  {'median':>8}  {'req/s':>8}  {'overlap':>8}")
    single = None
    n = 1
    while n <= top:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=n) as pool:
            lat = sorted(pool.map(lambda _: one(url, uuid.uuid4().hex[:8]), range(n)))
        wall = time.time() - t0
        med = lat[len(lat) // 2]
        if single is None:
            single = wall
        # 1.0 means concurrency bought nothing; n means perfect overlap.
        print(f"{n:>5}  {wall:>7.2f}s  {med:>7.2f}s  {n / wall:>8.2f}  "
              f"{n * single / wall:>7.1f}x")
        n *= 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
