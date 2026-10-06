"""Measure how many chat completions the gateway can retire concurrently.

The sampler is remote and the generation call is already offloaded, so the only
thing this exercises is the gateway's own prompt preparation: applying the chat
template and tokenizing the result. That work is CPU-bound and grows with the
conversation, which is what makes it the ceiling for a multi-turn agent loop --
the interesting number is how the wall time for N concurrent requests compares
to N times the cost of one.

Run with a real tokenizer so the GIL behaviour is real::

    python bench_gateway.py --model Qwen/Qwen3.5-4B --concurrency 32 --turns 40
"""

from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.testclient import TestClient

from arctic_platform.openai_compat import router


class StubPool:
    """Stands in for the sampling job: returns instantly, so the measurement
    is the gateway's own work rather than the model's."""

    # The gateway serves 503 until the sampling job reports initialized.
    _config = {"ready": True}

    async def generate(self, _prompts, _params):
        return [{"text": "ok", "token_ids": [1], "finish_reason": "stop"}]


def build_app(model: str):
    from transformers import AutoTokenizer

    app = FastAPI()
    app.include_router(router)  # the router already carries its own /v1 prefix
    app.state.sampling_pool = StubPool()
    app.state.sampling_tokenizer = AutoTokenizer.from_pretrained(model)
    app.state.sampling_model_name = model
    app.state.sampling_created = 1
    return app


def conversation(turns: int) -> list[dict]:
    """A conversation shaped like a real SWE rollout: long tool output each
    turn, which is what makes the template render expensive."""
    msgs: list[dict] = [{"role": "system", "content": "You are a helpful assistant."}]
    for i in range(turns):
        msgs.append({"role": "assistant", "content": f"step {i}: inspecting the repo"})
        msgs.append({"role": "user", "content": "output:\n" + ("x" * 2000)})
    return msgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--turns", type=int, default=40)
    args = ap.parse_args()

    client = TestClient(build_app(args.model))
    body = {
        "model": args.model,
        "messages": conversation(args.turns),
        "max_tokens": 8,
    }

    def one() -> float:
        t = time.time()
        r = client.post("/v1/chat/completions", json=body)
        r.raise_for_status()
        return time.time() - t

    serial = one()  # warm the tokenizer, and get the single-request cost

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        list(pool.map(lambda _: one(), range(args.concurrency)))
    wall = time.time() - t0

    print(f"turns={args.turns} concurrency={args.concurrency}")
    print(f"single request      : {serial:.3f}s")
    print(f"{args.concurrency} concurrent (wall) : {wall:.3f}s")
    print(f"throughput          : {args.concurrency / wall:.2f} req/s")
    # 1.0 means fully serialized (no overlap at all); higher means the CPU
    # work genuinely overlaps across threads.
    print(f"speedup vs serial   : {args.concurrency * serial / wall:.1f}x")


if __name__ == "__main__":
    main()
