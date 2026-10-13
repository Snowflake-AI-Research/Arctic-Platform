# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""One episode per model call, with the sampled tokens kept.

A later request that continues the same messages is encoded as the stored
prompt, the stored response, and the template's tokens for the new messages.
A request that does not continue starts another episode. A continuation chain
becomes one training sequence. A divergent context stays its own sequence.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field
from typing import Any


def _text_content(content: Any) -> Any:
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content if isinstance(part, dict))
    return content


def normalize_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten content parts and decode tool-call argument strings.

    Qwen's chat template runs ``arguments | tojson``. It needs a mapping, which
    is what vLLM produces before rendering.
    """
    normalized = []
    for message in messages:
        message = dict(message)
        if "content" in message:
            message["content"] = _text_content(message["content"])
        if message.get("tool_calls"):
            calls = []
            for call in message["tool_calls"]:
                call = json.loads(json.dumps(call))
                arguments = call.get("function", {}).get("arguments")
                if isinstance(arguments, str):
                    try:
                        call["function"]["arguments"] = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                calls.append(call)
            message["tool_calls"] = calls
        normalized.append(message)
    return normalized


@dataclass
class Episode:
    """Tokens from one ``generate`` call. ``extended`` means the prompt was spliced."""

    prompt_token_ids: list[int]
    response_token_ids: list[int]
    logprobs: list[float] | None
    sampling_params: dict[str, Any]
    extended: bool


@dataclass
class Session:
    """Messages and episodes for one rollout URL."""

    episodes: list[Episode] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    raw_messages: list[dict[str, Any]] = field(default_factory=list)
    assistant: dict[str, Any] | None = None
    api_key: str | None = None


def _arguments(value: Any) -> Any:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return value
    if isinstance(value, dict):
        return {key: value[key] for key in sorted(value)}
    return value


def canonical_message(message: dict[str, Any]) -> dict[str, Any]:
    """Role, content, and tool-call name plus arguments. Ids and extra keys are dropped."""
    content = message.get("content")
    if content is None:
        content = ""
    canonical = dict(role=message.get("role"), content=content)
    if message.get("tool_call_id"):
        canonical["tool_call_id"] = message["tool_call_id"]
    calls = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        calls.append(dict(name=function.get("name"), arguments=_arguments(function.get("arguments"))))
    if len(calls) > 0:
        canonical["tool_calls"] = calls
    return canonical


def canonical_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [canonical_message(message) for message in normalize_messages(messages)]


def extends(session: Session, messages: list[dict[str, Any]]) -> bool:
    """The new request replays the stored messages and then the assistant we returned."""
    if session.assistant is None:
        return False
    incoming = canonical_messages(messages)
    if len(incoming) < len(session.messages) + 1:
        return False
    if incoming[: len(session.messages)] != session.messages:
        return False
    return incoming[len(session.messages)].get("role") == "assistant"


def render_ids(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    tools: Any,
    template_kwargs: dict[str, Any],
    add_generation_prompt: bool,
) -> list[int]:
    kwargs = dict(template_kwargs)
    kwargs.pop("add_generation_prompt", None)
    text = tokenizer.apply_chat_template(
        normalize_messages(messages),
        tools=tools or None,
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
        **kwargs,
    )
    return list(tokenizer.encode(text, add_special_tokens=False))


def close_ids(
    tokenizer: Any,
    messages_before_assistant: list[dict[str, Any]],
    tools: Any,
    template_kwargs: dict[str, Any],
) -> list[int]:
    """Tokens the template writes after an assistant turn.

    Measured with an empty assistant message, so the result is the end marker
    and not a re-encoding of the sampled text.
    """
    prompted = render_ids(tokenizer, messages_before_assistant, tools, template_kwargs, True)
    closed = render_ids(
        tokenizer,
        list(messages_before_assistant) + [dict(role="assistant", content="")],
        tools,
        template_kwargs,
        False,
    )
    if closed[: len(prompted)] != prompted:
        return []
    return closed[len(prompted) :]


def unmatched_close(response_ids: list[int], close: list[int]) -> list[int]:
    """The part of ``close`` that the sampled response does not already end with."""
    if len(close) == 0:
        return []
    limit = min(len(response_ids), len(close))
    for size in range(limit, -1, -1):
        if size == 0 or response_ids[-size:] == close[:size]:
            return list(close[size:])
    return list(close)


def plan_prompt(session: Session, tokenizer: Any, body: dict[str, Any]) -> tuple[list[int], bool]:
    """Prompt ids to send, and whether they keep the previous sampled tokens.

    A continuation is the stored prompt, the stored response, any missing end
    marker, and the template suffix for the new messages. Anything else is a
    fresh render and a new episode.
    """
    messages = list(body.get("messages") or [])
    tools = body.get("tools") or None
    template_kwargs = dict(body.get("chat_template_kwargs") or {})
    fresh = render_ids(tokenizer, messages, tools, template_kwargs, True)
    if len(session.episodes) == 0 or not extends(session, messages):
        return fresh, False
    through = messages[: len(session.raw_messages) + 1]
    closed = render_ids(tokenizer, through, tools, template_kwargs, False)
    if fresh[: len(closed)] != closed:
        return fresh, False
    suffix = fresh[len(closed) :]
    end = close_ids(tokenizer, session.raw_messages, tools, template_kwargs)
    previous = session.episodes[-1]
    gap = unmatched_close(previous.response_token_ids, end)
    prompt = list(previous.prompt_token_ids) + list(previous.response_token_ids) + gap + suffix
    return prompt, True


def remember(
    session: Session,
    body: dict[str, Any],
    assistant_message: dict[str, Any],
    prompt_ids: list[int],
    response_ids: list[int],
    sampling_params: dict[str, Any],
    extended: bool,
    logprobs: list[float] | None,
) -> Episode:
    """Append the episode and the messages the next request has to replay."""
    episode = Episode(
        prompt_token_ids=list(prompt_ids),
        response_token_ids=list(response_ids),
        logprobs=list(logprobs) if logprobs is not None else None,
        sampling_params=dict(sampling_params),
        extended=extended,
    )
    session.episodes.append(episode)
    raw = normalize_messages(list(body.get("messages") or []))
    session.raw_messages = raw
    session.messages = [canonical_message(message) for message in raw]
    session.assistant = canonical_message(assistant_message)
    return episode


def split_chains(episodes: list[Episode]) -> list[list[Episode]]:
    """Group spliced calls. A call that did not extend the previous one starts a chain."""
    chains: list[list[Episode]] = []
    current: list[Episode] = []
    for episode in episodes:
        if len(current) > 0 and episode.extended:
            current.append(episode)
        else:
            if len(current) > 0:
                chains.append(current)
            current = [episode]
    if len(current) > 0:
        chains.append(current)
    return chains


def sequence_from_chain(chain: list[Episode]) -> tuple[list[int], list[int]]:
    """First prompt, and every later sampled token that the last prompt still contains."""
    first = list(chain[0].prompt_token_ids)
    last = list(chain[-1].prompt_token_ids) + list(chain[-1].response_token_ids)
    if last[: len(first)] != first:
        return list(chain[-1].prompt_token_ids), list(chain[-1].response_token_ids)
    return first, last[len(first) :]


def training_sequences(episodes: list[Episode], step_wise: bool) -> list[tuple[list[int], list[int]]]:
    """One sequence per chain.

    With step-wise off, a re-rendered history that still starts with the first
    prompt stays one sequence. That is the SkyRL default of one row per trajectory.
    """
    chains = split_chains(episodes)
    if len(chains) <= 1 or step_wise:
        return [sequence_from_chain(chain) for chain in chains]
    first = list(episodes[0].prompt_token_ids)
    last = list(episodes[-1].prompt_token_ids) + list(episodes[-1].response_token_ids)
    if last[: len(first)] == first:
        return [(first, last[len(first) :])]
    return [sequence_from_chain(chain) for chain in chains]
