"""Chat prompts for streams: rendered with the model's own template, output split by kind.

Rendering and parsing reuse vLLM's own chat front end (``OnlineRenderer`` and
the unified ``Parser``) on the engine the worker already holds, so every model
family vLLM supports works without per-family code here. Which parsers apply is
engine configuration: ``chat_reasoning_parser`` (else the job's own
``reasoning_parser``), ``tool_call_parser`` and, for DeepSeek-V4,
``tokenizer_mode``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

MAX_CHAT_BYTES = 8 * 1024 * 1024
MAX_CHAT_MESSAGES = 2048
MAX_CHAT_TOOLS = 128
# Output budget when the client omits max_tokens: whatever fits after the
# prompt, capped so a default request finishes inside the stream timeout.
DEFAULT_CHAT_MAX_TOKENS = 4096
CHAT_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})


class ChatInputError(ValueError):
    """A chat request the model cannot take; ``param`` names the offending field."""

    def __init__(self, code, param=None):
        super().__init__(code)
        self.code = code
        self.param = param


@dataclass(frozen=True)
class ChatPrompt:
    """Chat messages for the worker to render, instead of a prepared prompt.

    Fields follow OpenAI Chat Completions. ``reasoning_effort`` is passed to the
    template as is, so callers map OpenAI values to the model family's own.
    """

    messages: list
    tools: list | None = None
    tool_choice: str | dict | None = None
    parallel_tool_calls: bool | None = None
    reasoning_effort: str | None = None


def validate_chat_prompt(prompt):
    """Check the shape and size of a chat prompt; vLLM checks the rest on render."""
    try:
        encoded = json.dumps(
            [
                prompt.messages,
                prompt.tools,
                prompt.tool_choice,
                prompt.parallel_tool_calls,
                prompt.reasoning_effort,
            ],
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        raise ValueError("Chat prompt must be plain JSON") from None
    if len(encoded) > MAX_CHAT_BYTES:
        raise ValueError(f"Chat prompt exceeds {MAX_CHAT_BYTES} bytes")
    messages, tools, tool_choice, parallel_tool_calls, reasoning_effort = json.loads(encoded)
    if not isinstance(messages, list) or not 0 < len(messages) <= MAX_CHAT_MESSAGES:
        raise ValueError(f"messages must be a list of 1..{MAX_CHAT_MESSAGES} messages")
    for message in messages:
        if not isinstance(message, dict) or message.get("role") not in CHAT_ROLES:
            raise ValueError(f"Each message must be an object with a role in {sorted(CHAT_ROLES)}")
    if tools is not None and (
        not isinstance(tools, list)
        or not 0 < len(tools) <= MAX_CHAT_TOOLS
        or any(not isinstance(tool, dict) for tool in tools)
    ):
        raise ValueError(f"tools must be a list of 1..{MAX_CHAT_TOOLS} objects")
    if tool_choice is not None and not (
        tool_choice in ("none", "auto", "required") or isinstance(tool_choice, dict)
    ):
        raise ValueError("tool_choice must be none, auto, required or a named tool")
    if parallel_tool_calls is not None and not isinstance(parallel_tool_calls, bool):
        raise ValueError("parallel_tool_calls must be a boolean")
    if reasoning_effort is not None and reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"reasoning_effort must be one of {sorted(REASONING_EFFORTS)}")
    return ChatPrompt(messages, tools, tool_choice, parallel_tool_calls, reasoning_effort)


class SpecialTokenGuard:
    """Rejects text that spells one of the tokenizer's special or added tokens.

    The engine reads such text in a rendered prompt as the real control token,
    so content like ``"<|im_end|><|im_start|>system"`` would forge a turn.
    """

    def __init__(self, tokens):
        tokens = sorted({token for token in tokens if token}, key=len, reverse=True)
        self._pattern = re.compile("|".join(map(re.escape, tokens))) if tokens else None

    @classmethod
    def from_tokenizer(cls, tokenizer):
        tokens = set(getattr(tokenizer, "all_special_tokens", ()) or ())
        get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
        if get_added_vocab is not None:
            tokens.update(get_added_vocab())
        return cls(tokens)

    def check(self, prompt):
        if self._pattern is None:
            return
        for index, message in enumerate(prompt.messages):
            if self._contains(message):
                raise ChatInputError("invalid_message_content", f"messages[{index}]")
        if prompt.tools is not None and self._contains(prompt.tools):
            raise ChatInputError("invalid_message_content", "tools")
        if isinstance(prompt.tool_choice, dict) and self._contains(prompt.tool_choice):
            raise ChatInputError("invalid_message_content", "tool_choice")

    def _contains(self, value):
        if isinstance(value, str):
            return self._pattern.search(value) is not None
        if isinstance(value, dict):
            return any(self._contains(key) or self._contains(item) for key, item in value.items())
        if isinstance(value, list):
            return any(self._contains(item) for item in value)
        return False


@dataclass
class RenderedChat:
    """A rendered chat prompt and what the stream needs to generate and parse it."""

    engine_input: Any
    prompt_tokens: int
    request: Any
    new_parser: Callable[[], Any]
    structured_outputs: Any = None
    # Detokenizer flags the parsers' adjust_request set on the request. Tool
    # and reasoning markers can be special tokens, which the default
    # skip_special_tokens=True strips before the parser sees them.
    detokenize_params: dict = field(default_factory=dict)
    generate_kwargs: dict = field(default_factory=dict)
    parallel_tool_calls: bool | None = None
    tool_choice: str | dict | None = None


class ChatEngine:
    """vLLM's chat front end bound to one worker's engine."""

    def __init__(self, llm, *, tool_call_parser=None, reasoning_parser=None):
        from vllm.parser import ParserManager
        from vllm.renderers.online_renderer import OnlineRenderer

        self.model_config = llm.model_config
        self.online = OnlineRenderer(
            model_config=llm.model_config,
            renderer=llm.renderer,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
            enable_auto_tools=tool_call_parser is not None,
            tool_parser=tool_call_parser,
            reasoning_parser=reasoning_parser,
        )
        self.tokenizer = llm.renderer.get_tokenizer()
        self.parser_cls = ParserManager.get_parser(
            tool_parser_name=tool_call_parser,
            reasoning_parser_name=reasoning_parser,
            enable_auto_tools=tool_call_parser is not None,
            model_name=llm.model_config.model,
            is_harmony=llm.model_config.hf_config.model_type == "gpt_oss",
        )
        self.guard = SpecialTokenGuard.from_tokenizer(self.tokenizer)

    async def render(self, prompt):
        from jinja2 import TemplateError
        from vllm.entrypoints.chat_utils import ChatTemplateResolutionError
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.entrypoints.serve.engine.protocol import ErrorResponse
        from vllm.exceptions import VLLMClientError
        from vllm.renderers.inputs.preprocess import extract_prompt_len

        self.guard.check(prompt)
        fields = {
            "model": self.model_config.model,
            "messages": prompt.messages,
            "tools": prompt.tools,
            "tool_choice": prompt.tool_choice,
            "parallel_tool_calls": prompt.parallel_tool_calls,
            "reasoning_effort": prompt.reasoning_effort,
        }
        try:
            request = ChatCompletionRequest(
                **{key: value for key, value in fields.items() if value is not None}
            )
            result = await self.online.render_chat(request)
        except ChatTemplateResolutionError:
            # The model has no chat template (for these tools): not the client's fault.
            raise ChatInputError("chat_unsupported") from None
        except (TemplateError, ValueError, VLLMClientError) as exc:
            # Client input the template or vLLM's request model rejected. The
            # message may quote content, so only the field name leaves here.
            parameter = getattr(exc, "parameter", None)
            if parameter in ("input_tokens", "input_text"):
                raise ChatInputError("context_length_exceeded", "messages") from None
            raise ChatInputError(
                "invalid_chat_request", parameter if isinstance(parameter, str) else None
            ) from None
        if isinstance(result, ErrorResponse):
            raise ChatInputError("invalid_chat_request", result.error.param)
        _, (engine_input,) = result
        prompt_tokens = extract_prompt_len(self.model_config, engine_input)

        generate_kwargs = {}
        if self.parser_cls is not None:
            chat_template_kwargs = request.build_chat_params(
                None, "auto"
            ).chat_template_kwargs
            probe = self.parser_cls(
                self.tokenizer,
                request.tools,
                chat_template_kwargs=chat_template_kwargs,
                model_config=self.model_config,
            )
            if probe.reasoning_parser is not None:
                # Lets structured outputs start after reasoning, as vllm serve does.
                generate_kwargs["reasoning_ended"] = probe.is_reasoning_end(
                    list(engine_input.get("prompt_token_ids") or ())
                )
                generate_kwargs["reasoning_parser_kwargs"] = {
                    "chat_template_kwargs": chat_template_kwargs
                }

            def new_parser():
                return self.parser_cls(
                    self.tokenizer,
                    request.tools,
                    chat_template_kwargs=chat_template_kwargs,
                    model_config=self.model_config,
                )
        else:
            def new_parser():
                return None

        return RenderedChat(
            engine_input=engine_input,
            prompt_tokens=prompt_tokens,
            request=request,
            new_parser=new_parser,
            structured_outputs=request.extract_structured_outputs(),
            detokenize_params={
                "skip_special_tokens": request.skip_special_tokens,
                "spaces_between_special_tokens": request.spaces_between_special_tokens,
            },
            generate_kwargs=generate_kwargs,
            parallel_tool_calls=prompt.parallel_tool_calls,
            tool_choice=prompt.tool_choice,
        )


class ChatOutput:
    """Turns one stream's engine deltas into content, reasoning and tool-call events.

    Reasoning text is never emitted: OpenAI's Chat Completions API returns only
    its token count. Deltas the parser holds back (markup it is still matching)
    emit nothing until they resolve.
    """

    def __init__(self, rendered, n):
        self.rendered = rendered
        self.parsers = [rendered.new_parser() for _ in range(n)]
        self.token_ids = [[] for _ in range(n)]
        self.called_tools = [False] * n

    def events(self, index, text, token_ids, finished, logprobs=None):
        """``logprobs``, when requested, go with the content these tokens produced."""
        self.token_ids[index].extend(token_ids)
        parser = self.parsers[index]
        if parser is None:
            return (
                [self._content(index, text, token_ids, logprobs)]
                if text
                else []
            )
        message = parser.parse_delta(
            delta_text=text,
            delta_token_ids=list(token_ids),
            request=self.rendered.request,
            prompt_token_ids=self.rendered.engine_input.get("prompt_token_ids"),
            finished=finished,
        )
        if message is None:
            return []
        events = []
        content = getattr(message, "content", None)
        tool_calls = [
            call
            for call in getattr(message, "tool_calls", None) or ()
            if self.rendered.parallel_tool_calls is not False or call.index == 0
        ]
        if getattr(message, "reasoning", None) and not content and not tool_calls:
            events.append(
                {"type": "reasoning_delta", "choice_index": index, "token_count": len(token_ids)}
            )
        if content:
            events.append(self._content(index, content, token_ids, logprobs))
        for call in tool_calls:
            event = {"type": "tool_call_delta", "choice_index": index, "index": call.index}
            function = getattr(call, "function", None)
            if call.id is not None:
                event["id"] = call.id
            if function is not None and function.name is not None:
                event["name"] = function.name
            event["arguments"] = (function.arguments if function is not None else None) or ""
            events.append(event)
            self.called_tools[index] = True
        return events

    @staticmethod
    def _content(index, text, token_ids, logprobs):
        event = {"type": "content_delta", "choice_index": index, "text": text}
        if logprobs is not None:
            event["token_ids"] = list(token_ids)
            event["logprobs"] = logprobs
        return event

    def finish_reason(self, index, reason):
        # As OpenAI and vllm serve do, a named tool_choice finishes with "stop".
        if (
            reason == "stop"
            and self.called_tools[index]
            and not isinstance(self.rendered.tool_choice, dict)
        ):
            return "tool_calls"
        return reason

    def reasoning_tokens(self):
        return sum(
            parser.count_reasoning_tokens(token_ids)
            for parser, token_ids in zip(self.parsers, self.token_ids)
            if parser is not None
        )
