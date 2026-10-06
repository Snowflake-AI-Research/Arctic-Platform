"""Measure whether chat-template rendering is the generate throughput ceiling.

Collection wall-clock across four runs is ``total_generate_calls x ~1.33s`` and is
insensitive to both concurrency (24 -> 96) and sampling GPU count (16 -> 32). That
is the signature of one serialized stage, not of a per-call fixed overhead. The
gateway offloads ``_prepare_chat_prompt`` with ``asyncio.to_thread`` on the premise
that it releases the GIL -- true of the Rust tokenizer's ``encode``, but
``apply_chat_template`` renders Jinja in pure Python and so holds it.

If that premise is wrong, threaded throughput stays flat as threads rise, and its
ceiling is the ceiling we measured in the runs.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class _Msg:
    """The attribute surface ``_render_chat_prompt`` reads off a request message."""

    role: str
    content: str
    tool_calls: list | None = None
    tool_call_id: str | None = None
    name: str | None = None
    reasoning_content: str | None = None

DEFAULT_TEMPLATE = (
    "/modeling-code/boyiliu/prime-rl/prime_snowrl/configs/chat_templates/"
    "qwen35_preserve_all_thinking.jinja"
)


def build_conversation(turns: int) -> list[_Msg]:
    """A conversation shaped like a real mid-rollout turn (~62 turns, ~21.7k tokens)."""
    body = (
        "Let me look at the failing test and trace the call path through the module. "
        "I will inspect the relevant source, then apply a targeted fix.\n"
    ) * 6
    obs = "stdout:\n" + "\n".join(
        f"  file.py:{i}: some diagnostic output line here" for i in range(18)
    )
    msgs: list[_Msg] = [
        _Msg("system", "You are a software engineering agent. " * 8)
    ]
    for i in range(turns // 2):
        msgs.append(
            _Msg(
                "assistant",
                f"<think>{body}</think>\n<function=execute_bash>"
                f"<parameter=command>pytest -x tests/test_{i}.py</parameter></function>",
            )
        )
        msgs.append(_Msg("user", obs))
    return msgs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--chat-template", default=DEFAULT_TEMPLATE)
    ap.add_argument("--turns", type=int, default=62)
    ap.add_argument("--harbor", default="ap-harbor")
    ap.add_argument("--threads", default="1,8,32,64")
    args = ap.parse_args()

    sys.path.insert(0, args.harbor)
    from transformers import AutoTokenizer

    from arctic_platform.openai_compat import _prepare_chat_prompt

    tok = AutoTokenizer.from_pretrained(args.model)
    tpl = Path(args.chat_template)
    if tpl.exists():
        tok.chat_template = tpl.read_text()

    msgs = build_conversation(args.turns)
    text, ids = _prepare_chat_prompt(tok, msgs, {}, None)
    print(
        f"conversation: {len(msgs)} messages, {len(text)} chars, {len(ids)} tokens"
    )

    def once() -> None:
        _prepare_chat_prompt(tok, msgs, {}, None)

    once()  # warm any lazy template compile
    samples = []
    for _ in range(10):
        start = time.perf_counter()
        once()
        samples.append(time.perf_counter() - start)
    serial = statistics.median(samples)
    print(
        f"\nsingle-threaded _prepare_chat_prompt: {serial * 1000:.0f} ms"
        f"  (min {min(samples) * 1000:.0f}, max {max(samples) * 1000:.0f})"
    )

    print("\nthreaded throughput -- flat req/s means the GIL is serializing it:")
    print(f"{'threads':<9}{'wall':<9}{'req/s':<9}{'per-req':<10}{'speedup'}")
    for n in (int(x) for x in args.threads.split(",")):
        # Start the clock only once every thread is parked at the barrier, so
        # thread-spawn cost doesn't land inside the measured window.
        gate = threading.Barrier(n + 1)

        def worker() -> None:
            gate.wait()
            once()

        threads = [threading.Thread(target=worker) for _ in range(n)]
        for t in threads:
            t.start()
        gate.wait()
        start = time.perf_counter()
        for t in threads:
            t.join()
        wall = time.perf_counter() - start
        print(
            f"{n:<9}{wall:<9.2f}{n / wall:<9.2f}{wall / n * 1000:<10.0f}"
            f"{(n * serial) / wall:.2f}x"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
