"""Parameter values must survive parsing byte-for-byte.

Two things depend on this beyond "the arguments look right". A tool like
``edit_via_str_replace`` matches ``old_str`` against file contents, so losing
leading indentation turns a valid edit into a no-op. And a trajectory can only
be packed into one training sequence if the replayed turn is identical to the
sampled one, so any whitespace the parser drops shows up later as an
append-only violation and forces a fallback that costs ~27x more tokens.
"""

import json

from arctic_platform.openai_compat import _parse_tool_calls

EDIT_TOOL = [
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
    }
]

BASH_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "execute_bash",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        },
    }
]


def _args(text, tools):
    _, calls = _parse_tool_calls(text, tools)
    assert calls, f"no tool call parsed from {text!r}"
    return json.loads(calls[0]["function"]["arguments"])


def test_leading_indentation_is_preserved():
    body = "        self._cache: Dict[str, Any] = {}\n        url = message.url"
    text = (
        "<tool_call>\n<function=edit_via_str_replace>\n"
        "<parameter=path>\n/testbed/x.py\n</parameter>\n"
        f"<parameter=old_str>\n{body}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert _args(text, EDIT_TOOL)["old_str"] == body


def test_trailing_blank_line_is_preserved():
    # A heredoc ends with a blank line before the closing tag; the old parser
    # collapsed "\n\n" to "\n" and the replayed turn stopped matching.
    body = "cat <<'EOF' > /tmp/t.py\nprint(1)\nEOF\n"
    text = (
        "<tool_call>\n<function=execute_bash>\n"
        f"<parameter=command>\n{body}\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert _args(text, BASH_TOOL)["command"] == body


def test_only_one_framing_newline_is_removed_each_side():
    text = (
        "<tool_call>\n<function=execute_bash>\n"
        "<parameter=command>\n\n  ls  \n\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert _args(text, BASH_TOOL)["command"] == "\n  ls  \n"


def test_string_typed_value_is_not_coerced_by_looking_like_json():
    # "null", "true" and bare numbers are all valid JSON, and a shell command
    # or commit message is allowed to be any of them.
    for literal in ("123", "true", "[1, 2]"):
        text = (
            "<tool_call>\n<function=execute_bash>\n"
            f"<parameter=command>\n{literal}\n</parameter>\n"
            "</function>\n</tool_call>"
        )
        assert _args(text, BASH_TOOL)["command"] == literal


def test_integer_typed_value_is_coerced():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "f",
                "parameters": {"type": "object", "properties": {"n": {"type": "integer"}}},
            },
        }
    ]
    text = (
        "<tool_call>\n<function=f>\n<parameter=n>\n42\n</parameter>\n"
        "</function>\n</tool_call>"
    )
    assert _args(text, tools)["n"] == 42


def test_unclosed_parameter_still_parses():
    text = (
        "<tool_call>\n<function=execute_bash>\n"
        "<parameter=command>\nls -la\n</function>\n</tool_call>"
    )
    assert _args(text, BASH_TOOL)["command"] == "ls -la"


def test_malformed_span_is_left_in_content():
    text = "here is some prose <tool_call>not a function</tool_call>"
    content, calls = _parse_tool_calls(text, BASH_TOOL)
    assert calls == []
    assert content == text
