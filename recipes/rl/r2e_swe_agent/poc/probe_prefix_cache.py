#!/usr/bin/env python3
"""Ask the live sampler whether it is re-prefilling what it already has.

A multi-turn agent re-sends its entire conversation every turn, so a step
processes about nineteen times more tokens than it has distinct context. That
is only wasted work if the prefix cache misses. Inferring the hit rate from
step time is guesswork; this measures it.

Three requests, each capped at one output token so the timing is prefill:

  cold  a long prompt with a unique prefix, so nothing can be cached
  warm  byte-identical to cold, which must hit if caching works at all
  grow  the same prompt with one more turn appended, the shape a real agent
        actually produces

If warm and grow are not markedly faster than cold, turns are re-prefilling
from scratch and the fix is cache capacity or request routing, not sampling.

  python3 probe_prefix_cache.py http://host:port/v1 [approx_tokens]
"""

import json
import sys
import time
import urllib.request
import uuid

MODEL = "Qwen/Qwen3.5-4B"


def post(url: str, body: dict, bearer: str | None = None, timeout: int = 600):
    data = json.dumps(body).encode()
    headers = {"Content-Type": "application/json"}
    if bearer:
        # The gateway reads the rollout id from the bearer token and forwards
        # it as the sampler's routing key, so sharing one across the three
        # requests is what makes them land on the same replica.
        headers["Authorization"] = f"Bearer {bearer}"
    req = urllib.request.Request(url, data=data, headers=headers)
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = json.loads(r.read())
    return time.time() - t0, payload


def conversation(nonce: str, turns: int, chunk: int) -> list[dict]:
    """Roughly the shape of a SWE rollout: long tool output every turn."""
    msgs = [{"role": "system", "content": f"session {nonce}. You are a coding agent."}]
    for i in range(turns):
        msgs.append({"role": "assistant", "content": f"step {i}: inspecting the repo"})
        msgs.append({"role": "user", "content": f"output {i}:\n" + ("data " * chunk)})
    return msgs


def main() -> int:
    base = sys.argv[1].rstrip("/")
    turns = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    # A shared id across all three requests is the affinity hint under test;
    # pass "none" to measure what happens without one.
    bearer = sys.argv[4] if len(sys.argv) > 4 else f"probe-{uuid.uuid4().hex[:8]}"
    if bearer == "none":
        bearer = None
    url = f"{base}/chat/completions"

    # The sampler is serving live rollouts, so any single timing is mostly
    # queueing. Repeat and keep the minimum: the fastest observation is the
    # one least contaminated by waiting behind someone else's prefill.
    colds, warms, grows = [], [], []
    n_prompt = 0

    for _ in range(rounds):
        msgs = conversation(uuid.uuid4().hex, turns, chunk=250)
        body = {"model": MODEL, "messages": msgs, "max_tokens": 1, "temperature": 0.0}

        dt, out = post(url, body, bearer)
        colds.append(dt)
        n_prompt = (out.get("usage") or {}).get("prompt_tokens", n_prompt)

        dt, _ = post(url, body, bearer)
        warms.append(dt)

        grown = msgs + [
            {"role": "assistant", "content": "one more step"},
            {"role": "user", "content": "output:\n" + ("data " * 250)},
        ]
        dt, _ = post(url, {**body, "messages": grown}, bearer)
        grows.append(dt)

    cold, warm, grow = min(colds), min(warms), min(grows)

    print(f"prompt tokens : {n_prompt:,}   (rounds={rounds}, minima, routing_key={bearer!r})")
    print(f"cold          : {cold:.2f}s   all={[round(x, 1) for x in colds]}")
    print(f"warm (same)   : {warm:.2f}s   {cold / max(warm, 1e-6):.1f}x faster"
          f"   all={[round(x, 1) for x in warms]}")
    print(f"grow (+1 turn): {grow:.2f}s   {cold / max(grow, 1e-6):.1f}x faster"
          f"   all={[round(x, 1) for x in grows]}")
    print()
    if warm > 0.6 * cold:
        print("VERDICT: prefix cache is NOT serving repeats -- every turn re-prefills.")
    elif grow > 0.6 * cold:
        print("VERDICT: exact repeats hit, but a grown conversation does not.")
    else:
        print("VERDICT: prefix cache is working; look elsewhere for the gap.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
