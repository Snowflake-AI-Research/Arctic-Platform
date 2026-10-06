#!/usr/bin/env python3
"""Check that a sampled assistant turn survives the replay round-trip verbatim.

This is the property trajectory packing depends on. A turn is sampled as raw
text, split into ``reasoning_content`` plus ``tool_calls``, handed to the
harness, and later replayed through the chat template to build the next turn's
prompt. If what the template renders is not byte-identical to what the model
emitted, the packed sequence cannot contain the sampled tokens and the
append-only check rejects the trajectory.

Runs real completions from a run's ``raw_completions.jsonl`` through the real
parse and render functions, so it measures the actual pipeline rather than a
reconstruction of it.

  python3 diag_roundtrip.py <raw_completions.jsonl> <chat_template.jinja> [limit]
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ap-harbor"))

from arctic_platform.openai_compat import (  # noqa: E402
    _parse_tool_calls,
    _split_reasoning,
    _tool_call_for_template,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_via_str_replace",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old_str": {"type": "string"},
                    "new_str": {"type": "string"},
                },
            },
        },
    },
]


def main() -> int:
    raw_path = sys.argv[1]
    template_path = sys.argv[2]
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 400

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
    tok.chat_template = Path(template_path).read_text()

    rows = []
    with open(raw_path) as fh:
        for line in fh:
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break

    ok = 0
    bad = []
    for r in rows:
        text = r.get("text") or ""
        if "<tool_call>" not in text:
            continue
        # Mirror the gateway exactly: it splits into (visible, reasoning) and
        # then pulls the call out of whichever side carries it.
        body, think = _split_reasoning(text, think_open=True)
        if think and not body:
            think, calls = _parse_tool_calls(think, TOOLS)
        else:
            body, calls = _parse_tool_calls(body, TOOLS)
        if not calls:
            bad.append(("no call parsed", text[:200]))
            continue

        msg = {"role": "assistant", "content": body or ""}
        if think:
            msg["reasoning_content"] = think
        msg["tool_calls"] = [_tool_call_for_template(c) for c in calls]

        rendered = tok.apply_chat_template(
            [{"role": "user", "content": "x"}, msg],
            tokenize=False,
            add_generation_prompt=False,
        )

        # The model's own output, minus the end-of-turn marker the template adds.
        sampled = text.replace("<|im_end|>", "").rstrip("\n")
        if sampled in rendered:
            ok += 1
        else:
            bad.append(("not verbatim", sampled, rendered))

    total = ok + len(bad)
    print(f"round-trip verbatim: {ok}/{total}")
    if bad:
        kind, *rest = bad[0]
        print(f"\nfirst failure ({kind}):")
        if kind == "not verbatim":
            sampled, rendered = rest
            # Show where they part company, in text.
            i = next(
                (j for j in range(min(len(sampled), len(rendered))) if sampled[j] != rendered[j]),
                0,
            )
            print("  sampled  :", repr(sampled[max(0, i - 80) : i + 80]))
            print("  rendered :", repr(rendered[max(0, i - 80) : i + 80]))
        else:
            print("  ", repr(rest[0][:300]))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
