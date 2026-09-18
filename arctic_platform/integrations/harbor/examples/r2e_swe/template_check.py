"""Check his chat template round-trips a multi-turn tool-using transcript.

His template is custom (thinking_retention = "all"), and our renderer feeds it
fields the stock one ignores. If it rejects a replayed assistant turn or drops
the prior reasoning, every turn after the first is sampled from a different
distribution than his run — silently, since the request still succeeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, "/modeling-code/karthik/abstract-remote-exps/ap-harbor")

from arctic_platform.openai_compat import ChatMessage  # noqa: E402
from arctic_platform.openai_compat import _render_chat_prompt  # noqa: E402

TEMPLATE = (
    "/modeling-code/boyiliu/prime-rl/prime_snowrl/configs/"
    "chat_templates/qwen35_preserve_all_thinking.jinja"
)

TOOLS = [{
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": "Run a bash command.",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
}]

TRANSCRIPT = [
    {"role": "user", "content": "Fix the bug in /testbed."},
    {
        "role": "assistant",
        "content": None,
        "reasoning_content": "FIRST_TURN_REASONING: I should look around.",
        "tool_calls": [{
            "id": "call_1",
            "type": "function",
            "function": {"name": "execute_bash", "arguments": '{"command": "ls /testbed"}'},
        }],
    },
    {"role": "tool", "tool_call_id": "call_1", "content": "setup.py  numpy/"},
    {
        "role": "assistant",
        "content": None,
        "reasoning_content": "SECOND_TURN_REASONING: now read the file.",
        "tool_calls": [{
            "id": "call_2",
            "type": "function",
            "function": {"name": "execute_bash", "arguments": '{"command": "cat setup.py"}'},
        }],
    },
    {"role": "tool", "tool_call_id": "call_2", "content": "from setuptools import setup"},
]


def main() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    tok.chat_template = Path(TEMPLATE).read_text()
    msgs = [ChatMessage.model_validate(m) for m in TRANSCRIPT]

    failures = []
    rendered = _render_chat_prompt(
        tok, msgs, template_kwargs={"enable_thinking": True}, tools=TOOLS
    )

    for label, needle in [
        ("tool schema", "execute_bash"),
        ("first tool call", "ls /testbed"),
        ("first tool result", "setup.py"),
        ("second tool call", "cat setup.py"),
        ("second tool result", "from setuptools import setup"),
        # The whole point of his template: earlier turns keep their thinking.
        ("turn-1 reasoning retained", "FIRST_TURN_REASONING"),
        ("turn-2 reasoning retained", "SECOND_TURN_REASONING"),
    ]:
        if needle not in rendered:
            failures.append(f"{label}: {needle!r} missing from rendered prompt")

    if not rendered.rstrip().endswith("<think>"):
        failures.append(f"prompt does not leave <think> open; tail={rendered[-120:]!r}")

    stock = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    stock_rendered = _render_chat_prompt(
        stock, msgs, template_kwargs={"enable_thinking": True}, tools=TOOLS
    )
    retained_by_stock = "FIRST_TURN_REASONING" in stock_rendered

    if failures:
        print("[template] FAIL")
        for f in failures:
            print(f"  - {f}")
        print(f"\n--- tail of rendered ---\n{rendered[-600:]}")
        return 1

    print(f"[template] PASS: {len(rendered)} chars, tools + both turns + "
          f"both reasoning blocks retained, <think> left open")
    print(f"[template] stock template retains turn-1 reasoning: {retained_by_stock} "
          f"(this is the difference his template makes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
