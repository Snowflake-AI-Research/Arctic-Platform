"""Replay captured sampler text through the serving parser.

The harness rejects a turn on the shape of what it receives -- reasoning in
``reasoning_content``, calls in ``tool_calls`` -- not on the model's raw text.
So a rollout marked ``format_invalid`` has two very different explanations: the
model wrote something malformed, or it wrote the right thing and the parser
lost it on the way through. This script settles which, by running the exact
text the sampler produced through the exact functions the server uses.

    python diag_parser.py runs/<run>/raw_completions.jsonl
"""

from __future__ import annotations

import collections
import json
import sys

from arctic_platform.openai_compat import _parse_tool_calls, _split_reasoning


def classify(text: str, think_open: bool = True) -> str:
    # Both helpers return (remaining_text, extracted): content before
    # reasoning, leftover before calls.
    content, reasoning = _split_reasoning(text, think_open)
    leftover, calls = _parse_tool_calls(content)

    if not calls:
        if "<tool_call>" in text:
            return "tool_call_in_text_but_not_parsed"
        return "no_tool_call_at_all"
    # The harness requires the visible answer to be empty: reasoning in
    # reasoning_content, the action in tool_calls, nothing in between.
    if (leftover or "").strip():
        return "nonempty_content"
    if not (reasoning or "").strip():
        return "missing_reasoning"
    return "ok"


def main(path: str) -> None:
    rows = [json.loads(line) for line in open(path)]
    print(f"{len(rows)} sampled turns")
    print("finish_reason:", dict(collections.Counter(r["finish_reason"] for r in rows)))

    # The flag is derived at serve time from the rendered prompt, so check both
    # settings: if they disagree, the template is what decides format validity.
    for think_open in (True, False):
        counts = collections.Counter()
        examples: dict[str, str] = {}
        for row in rows:
            kind = classify(row["text"], think_open)
            counts[kind] += 1
            examples.setdefault(kind, row["text"])

        print(f"\n=== think_open={think_open} ===")
        for kind, n in counts.most_common():
            print(f"  {n:4d}  {100 * n / len(rows):5.1f}%  {kind}")

        if think_open:
            for kind in counts:
                if kind == "ok":
                    continue
                print(f"\n----- example: {kind} -----")
                print(examples[kind][:1000])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "raw_completions.jsonl")
