#!/usr/bin/env python3
"""Does the sampler scale across its data-parallel replicas?

``--sample-gpus 6 --tensor-parallel 1`` should give six independent replicas.
One idle 11k-token request returns in ~0.5s, i.e. ~22k tok/s, which is about
what a single H200 does for a 4B model. Yet with 24 agents in flight the whole
fleet retires only ~0.6 requests/s -- *less* aggregate throughput than one
replica measured alone. Either the requests are not reaching all six replicas
or they are contending somewhere.

``probe_concurrency`` cannot answer this: its prompts are deliberately tiny, so
it measures per-request overhead rather than throughput. Prefill throughput
only shows up with prompts big enough to dominate, which is what this sends.

Aggregate tok/s that stays flat as concurrency rises means one replica is doing
all the work. Scaling with concurrency means the fleet is healthy and the
bottleneck is the number of tokens we ask it to process.

  python3 probe_sampler_scaling.py <base_url> [max_concurrency] [turns]
"""

from __future__ import annotations

import json
import statistics
import sys
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor

MODEL = "Qwen/Qwen3.5-4B"


def conversation(nonce: str, turns: int) -> list[dict]:
    msgs = [{"role": "system", "content": f"session {nonce}. You are a coding agent."}]
    for i in range(turns):
        msgs.append({"role": "assistant", "content": f"step {i}: inspecting the repo"})
        msgs.append({"role": "user", "content": f"output {i}:\n" + ("data " * 250)})
    return msgs


def one(url: str, turns: int) -> tuple[float, int]:
    nonce = uuid.uuid4().hex
    body = {
        "model": MODEL,
        "messages": conversation(nonce, turns),
        "max_tokens": 1,
        "temperature": 0.0,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode(),
        headers={
            "Content-Type": "application/json",
            # Unique per request: a shared key would pin every request to one
            # replica, which is the opposite of what we want to measure here.
            "Authorization": f"Bearer scale-{nonce[:8]}",
        },
    )
    t = time.time()
    with urllib.request.urlopen(req, timeout=900) as r:
        payload = json.loads(r.read())
    dt = time.time() - t
    return dt, int((payload.get("usage") or {}).get("prompt_tokens") or 0)


def main() -> int:
    base = sys.argv[1].rstrip("/")
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    turns = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    url = f"{base}/chat/completions"

    print(f"{'conc':>5} {'wall':>8} {'p50 lat':>8} {'tokens':>10} {'agg tok/s':>10} {'scaling':>8}")
    base_rate = None
    n = 1
    while n <= top:
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=n) as pool:
            res = list(pool.map(lambda _: one(url, turns), range(n)))
        wall = time.time() - t0
        toks = sum(t for _, t in res)
        lat = statistics.median(d for d, _ in res)
        rate = toks / wall
        if base_rate is None:
            base_rate = rate
        print(f"{n:>5} {wall:>7.2f}s {lat:>7.2f}s {toks:>10,} {rate:>10,.0f} "
              f"{rate / base_rate:>7.1f}x")
        n *= 2
    print("\nflat scaling => one replica is serving everything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
