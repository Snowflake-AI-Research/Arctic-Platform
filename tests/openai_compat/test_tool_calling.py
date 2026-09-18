"""Native tool calling + reasoning on ``/v1/chat/completions``.

Harnesses like ``mini-swe-agent-plus`` send a ``tools`` array, require a
``tool_calls`` reply, and score the analysis separately as
``reasoning_content``. Before this path existed the router accepted ``tools``
and silently ignored them, so such a harness failed protocol validation on its
first turn and every rollout graded invalid.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from arctic_platform.openai_compat import ChatMessage
from arctic_platform.openai_compat import _parse_tool_calls
from arctic_platform.openai_compat import _render_chat_prompt
from arctic_platform.openai_compat import router


class _ToolTokenizer:
    """Chat template that accepts ``tools`` and echoes what it was handed."""

    chat_template = "<template>"

    def __init__(self) -> None:
        self.seen_tools: list[dict[str, Any]] | None = None
        self.seen_messages: list[dict[str, Any]] | None = None
        self.seen_thinking: bool | None = None

    def apply_chat_template(
        self, messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False, tools=None,
    ) -> str:
        self.seen_tools = tools
        self.seen_messages = messages
        self.seen_thinking = enable_thinking
        head = f"[tools={len(tools or [])}]"
        return head + "\n".join(f"{m['role']}: {m.get('content')}" for m in messages)


class _Pool:
    def __init__(self, text: str, finish: str = "stop") -> None:
        self._config = object()
        self._text = text
        self._finish = finish

    async def generate(self, prompts, sampling_params=None, model_id=None,
                       routing_key=None, strict=False):
        return [{
            "text": self._text,
            "token_ids": [1, 2, 3],
            "finish_reason": self._finish,
            "prompt_len": 5,
            "generation_len": 3,
            "prefix_cache_len": 0,
        } for _ in prompts]


def _app(pool, tokenizer) -> FastAPI:
    app = FastAPI()
    app.include_router(router)
    app.state.sampling_pool = pool
    app.state.sampling_tokenizer = tokenizer
    app.state.sampling_model_name = "Qwen/Qwen3.5-4B"
    app.state.sampling_created = 1
    return app


TOOLS = [{
    "type": "function",
    "function": {
        "name": "execute_bash",
        "description": "Run a bash command.",
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
    },
}]


def _body(**over: Any) -> dict[str, Any]:
    base = {
        "model": "Qwen/Qwen3.5-4B",
        "messages": [{"role": "user", "content": "list the files"}],
        "tools": TOOLS,
    }
    base.update(over)
    return base


def test_tools_reach_the_chat_template():
    tok = _ToolTokenizer()
    with TestClient(_app(_Pool("ok"), tok)) as c:
        resp = c.post("/v1/chat/completions", json=_body())
    assert resp.status_code == 200, resp.text
    assert tok.seen_tools is not None
    assert tok.seen_tools[0]["function"]["name"] == "execute_bash"


def _call(name: str, **params: str) -> str:
    """Qwen3.5's nested-XML tool-call envelope, as its own template specifies."""
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


def test_tool_call_block_becomes_openai_tool_calls():
    with TestClient(_app(_Pool(_call("execute_bash", cmd="ls -la")), _ToolTokenizer())) as c:
        resp = c.post("/v1/chat/completions", json=_body())
    choice = resp.json()["choices"][0]
    calls = choice["message"]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["function"]["name"] == "execute_bash"
    assert json.loads(calls[0]["function"]["arguments"]) == {"cmd": "ls -la"}
    # Clients dispatch on this rather than on the raw stop reason.
    assert choice["finish_reason"] == "tool_calls"
    assert not (choice["message"]["content"] or "")


def test_multiline_parameter_survives_verbatim():
    """Heredocs and patches are the payloads that matter for a SWE agent."""
    script = "cat > /tmp/f.py <<'EOF'\ndef f():\n    return 1\nEOF"
    with TestClient(_app(_Pool(_call("execute_bash", cmd=script)), _ToolTokenizer())) as c:
        resp = c.post("/v1/chat/completions", json=_body())
    args = json.loads(resp.json()["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"])
    assert args["cmd"] == script


def test_multiple_tool_calls_are_indexed():
    text = _call("execute_bash", cmd="a") + _call("execute_bash", cmd="b")
    with TestClient(_app(_Pool(text), _ToolTokenizer())) as c:
        resp = c.post("/v1/chat/completions", json=_body())
    calls = resp.json()["choices"][0]["message"]["tool_calls"]
    assert [c_["index"] for c_ in calls] == [0, 1]
    assert len({c_["id"] for c_ in calls}) == 2


def test_thinking_is_split_into_reasoning_content():
    text = "<think>I should list files first.</think>Here you go."
    with TestClient(_app(_Pool(text), _ToolTokenizer())) as c:
        resp = c.post("/v1/chat/completions", json=_body())
    msg = resp.json()["choices"][0]["message"]
    assert msg["reasoning_content"] == "I should list files first."
    assert msg["content"] == "Here you go."


def test_unpaired_closing_think_tag_is_still_reasoning():
    """Qwen3.5's generation prompt opens ``<think>`` itself, so completions
    usually carry only the closing tag."""
    text = "I should list files first.</think>\n\nHere you go."
    with TestClient(_app(_Pool(text), _ToolTokenizer())) as c:
        resp = c.post("/v1/chat/completions", json=_body())
    msg = resp.json()["choices"][0]["message"]
    assert msg["reasoning_content"] == "I should list files first."
    assert msg["content"] == "Here you go."


def test_thinking_and_tool_call_together():
    text = "list them</think>" + _call("execute_bash", cmd="ls")
    with TestClient(_app(_Pool(text), _ToolTokenizer())) as c:
        resp = c.post("/v1/chat/completions", json=_body())
    msg = resp.json()["choices"][0]["message"]
    assert msg["reasoning_content"] == "list them"
    assert msg["tool_calls"][0]["function"]["name"] == "execute_bash"


def test_assistant_tool_calls_round_trip_into_the_prompt():
    """The agent replays its own prior turns; the template must see them."""
    tok = _ToolTokenizer()
    prior = [
        {"role": "user", "content": "list the files"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "execute_bash", "arguments": '{"cmd": "ls"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "a.py b.py"},
    ]
    with TestClient(_app(_Pool("done"), tok)) as c:
        resp = c.post("/v1/chat/completions", json=_body(messages=prior))
    assert resp.status_code == 200, resp.text
    seen = tok.seen_messages
    assert seen[1]["tool_calls"][0]["function"]["name"] == "execute_bash"
    assert seen[2]["tool_call_id"] == "call_1"


def test_malformed_tool_call_is_left_as_content():
    """A broken call is the harness's business to penalize, not a 500."""
    text = "<tool_call>no function block here</tool_call>"
    with TestClient(_app(_Pool(text), _ToolTokenizer())) as c:
        resp = c.post("/v1/chat/completions", json=_body())
    assert resp.status_code == 200
    msg = resp.json()["choices"][0]["message"]
    assert "tool_calls" not in msg
    assert "no function block here" in msg["content"]


def test_no_tools_means_no_tool_parsing():
    """Without a tools array the text is returned verbatim, so a text-protocol
    agent that legitimately prints ``<tool_call>`` is not mangled."""
    text = _call("execute_bash", cmd="ls")
    body = {"model": "Qwen/Qwen3.5-4B",
            "messages": [{"role": "user", "content": "hi"}]}
    with TestClient(_app(_Pool(text), _ToolTokenizer())) as c:
        resp = c.post("/v1/chat/completions", json=body)
    msg = resp.json()["choices"][0]["message"]
    assert "tool_calls" not in msg
    assert msg["content"] == text


class _ThinkOpenTokenizer(_ToolTokenizer):
    """Template that pre-opens ``<think>``, as Qwen3.5's real one does."""

    def apply_chat_template(self, messages, **kw) -> str:
        return super().apply_chat_template(messages, **kw) + "\n<think>\n"


# Verbatim shape of what Qwen3.5-4B actually returned on an R2E task: prose,
# then a call, and no closing tag anywhere.
UNCLOSED = (
    "I'll help you solve this issue. Let me start by exploring the repository.\n\n"
    "<tool_call>\n<function=execute_bash>\n<parameter=cmd>\nls /testbed\n"
    "</parameter>\n</function>\n</tool_call>"
)


def test_unclosed_think_block_makes_the_whole_completion_reasoning():
    """The bug that zeroed a live run.

    mini-swe-agent-plus rejects an assistant turn that has content next to a
    tool call, and separately rejects one with no reasoning. Filing this prose
    as ``content`` tripped both at once, so every rollout died on turn one with
    ``format_invalid`` and the reward curve was flat zero for reasons that had
    nothing to do with the policy.
    """
    with TestClient(_app(_Pool(UNCLOSED), _ThinkOpenTokenizer())) as c:
        resp = c.post("/v1/chat/completions", json=_body())
    assert resp.status_code == 200
    choice = resp.json()["choices"][0]
    msg = choice["message"]

    assert msg["content"] is None, "no visible answer exists inside an open think block"
    assert "exploring the repository" in msg["reasoning_content"]
    assert choice["finish_reason"] == "tool_calls"
    (call,) = msg["tool_calls"]
    assert call["function"]["name"] == "execute_bash"
    assert json.loads(call["function"]["arguments"])["cmd"] == "ls /testbed"
    # The call must not be left behind as text in the reasoning it came from.
    assert "<tool_call>" not in msg["reasoning_content"]


def test_unclosed_think_without_a_pre_opened_block_stays_content():
    """Absent the open tag there is no reason to reinterpret plain output, and
    a text-protocol agent's prose must keep arriving as ``content``."""
    with TestClient(_app(_Pool("just prose"), _ToolTokenizer())) as c:
        resp = c.post("/v1/chat/completions", json=_body())
    msg = resp.json()["choices"][0]["message"]
    assert msg["content"] == "just prose"
    assert "reasoning_content" not in msg


def test_closing_tag_still_wins_over_the_open_prompt():
    """A model that does close the block gets the normal split, so the
    fallback cannot swallow a real answer."""
    text = "weighing options</think>the answer is 4"
    with TestClient(_app(_Pool(text), _ThinkOpenTokenizer())) as c:
        resp = c.post("/v1/chat/completions", json=_body())
    msg = resp.json()["choices"][0]["message"]
    assert msg["content"] == "the answer is 4"
    assert msg["reasoning_content"] == "weighing options"


def test_empty_completion_in_an_open_think_block_has_no_reasoning():
    with TestClient(_app(_Pool("   "), _ThinkOpenTokenizer())) as c:
        resp = c.post("/v1/chat/completions", json=_body())
    msg = resp.json()["choices"][0]["message"]
    assert "reasoning_content" not in msg


def test_real_qwen_template_opens_think_only_when_thinking_is_enabled():
    """Both halves matter, and they are what the router's default decides.

    With thinking off the template pre-*closes* the block, so the model emits
    no reasoning and a harness that requires a reasoning block rejects every
    turn. With it on the block is left open for the model to close itself.
    """
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"tokenizer unavailable: {exc}")

    msgs = [ChatMessage.model_validate({"role": "user", "content": "hi"})]
    off = _render_chat_prompt(tok, msgs, tools=TOOLS)
    on = _render_chat_prompt(
        tok, msgs, template_kwargs={"enable_thinking": True}, tools=TOOLS
    )
    assert not off.rstrip().endswith("<think>")
    assert on.rstrip().endswith("<think>")


def test_top_level_chat_template_kwargs_are_honored():
    """vLLM accepts the key at the top level; the SDK nests it in extra_body.
    A caller that works against vLLM must work here unchanged."""
    tok = _ToolTokenizer()
    with TestClient(_app(_Pool("ok"), tok)) as c:
        c.post("/v1/chat/completions",
               json=_body(chat_template_kwargs={"enable_thinking": True}))
    assert tok.seen_thinking is True


def test_extra_body_chat_template_kwargs_still_work():
    tok = _ToolTokenizer()
    with TestClient(_app(_Pool("ok"), tok)) as c:
        c.post("/v1/chat/completions",
               json=_body(extra_body={"chat_template_kwargs": {"enable_thinking": True}}))
    assert tok.seen_thinking is True


def test_real_qwen_template_accepts_a_full_tool_round_trip():
    """The fakes prove plumbing; only the real template proves correctness.

    Renders an assistant tool call and its tool reply through Qwen3.5-4B's own
    chat template — the case that raised "Can only get item pairs from a
    mapping" until ``arguments`` was converted back to a dict.
    """
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")
    except Exception as exc:  # noqa: BLE001 — offline CI shouldn't fail here
        pytest.skip(f"tokenizer unavailable: {exc}")

    msgs = [
        {"role": "user", "content": "list the files"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_1",
                "type": "function",
                "function": {"name": "execute_bash", "arguments": '{"cmd": "ls"}'},
            }],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "a.py b.py"},
    ]
    rendered = _render_chat_prompt(
        tok, [ChatMessage.model_validate(m) for m in msgs], tools=TOOLS
    )
    assert "execute_bash" in rendered
    assert "a.py b.py" in rendered

    # And the format the template *instructs* the model to emit must be the
    # format we parse back out.
    _, calls = _parse_tool_calls(_call("execute_bash", cmd="ls"))
    assert calls[0]["function"]["name"] == "execute_bash"
