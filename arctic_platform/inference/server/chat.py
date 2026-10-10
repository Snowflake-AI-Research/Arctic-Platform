"""Chat prompts for streams: rendered with the model's own template, output split by kind.

Rendering and parsing reuse vLLM's own chat front end (``OnlineRenderer`` and
the unified ``Parser``) on the engine the worker already holds. Which parsers
apply comes from ``CHAT_MODELS``, keyed by the architecture vLLM resolved for
the checkpoint; the ``chat_reasoning_parser`` and ``tool_call_parser``
``ModelConfig`` fields override it. Chat parses with the engine's reasoner when
it has one. A probe render of the model's template narrows the rest (see
``ChatEngine.probe``).
"""

from __future__ import annotations

import asyncio
import functools
import itertools
import json
import logging
import re
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Callable, Mapping

logger = logging.getLogger(__name__)

MAX_CHAT_BYTES = 8 * 1024 * 1024
MAX_CHAT_MESSAGES = 2048
MAX_CHAT_TOOLS = 128
CHAT_ROLES = frozenset({"system", "developer", "user", "assistant", "tool"})
# Content part types vLLM reads as text. Media parts would make the worker fetch
# URLs or load tensors on the client's behalf.
TEXT_PART_TYPES = frozenset({"text", "input_text", "output_text", "refusal", "thinking"})
REASONING_EFFORTS = frozenset({"none", "minimal", "low", "medium", "high", "xhigh", "max"})


@dataclass(frozen=True)
class ChatModel:
    """How chat mode renders and parses one model architecture.

    ``reasoning_efforts`` maps a requested ``reasoning_effort`` to the level the
    family's template names; values it does not list reach the template as is.

    ``reasoning_parser=None`` means chat parses no reasoning: everything the
    model writes is content, so logprobs are allowed. A job's own engine
    reasoner still replaces it, as for every entry. ``thinking_optional`` is
    the model's, not the parser's: a model that always thinks keeps refusing
    "none" when the worker runs it without a reasoner.
    """

    reasoning_parser: str | None
    tool_call_parser: str | None
    # Whether reasoning_effort="none" turns thinking off (vLLM passes the
    # template enable_thinking=False). If not, "none" is refused.
    thinking_optional: bool = False
    reasoning_efforts: Mapping[str, str] = MappingProxyType({})

    def template_reasoning_effort(self, effort):
        if effort == "none" and not self.thinking_optional:
            raise ChatInputError("invalid_chat_request", "reasoning_effort")
        return self.reasoning_efforts.get(effort, effort)


# Qwen3.8's template takes low, medium and xhigh and raises on anything else;
# Qwen3.5's and 3.6's, on the same architectures, ignore the value.
_QWEN3_5 = ChatModel(
    "qwen3",
    "qwen3_coder",
    thinking_optional=True,
    reasoning_efforts=MappingProxyType({"minimal": "low", "high": "xhigh", "max": "xhigh"}),
)
_QWEN3 = ChatModel("qwen3", "hermes", thinking_optional=True)
_DEEPSEEK_V4 = ChatModel("deepseek_v4", "deepseek_v4", thinking_optional=True)
# GLM-5.3-Flash always thinks, and its template takes Low and High, reading
# anything else as Max.
_GLM5_NEXT = ChatModel(
    "glm47",
    "glm47",
    reasoning_efforts=MappingProxyType({"minimal": "low", "medium": "high", "xhigh": "max"}),
)
# Names as vLLM resolves them (``model_config.architecture``). Parser names are
# vLLM's registered ones, as its recipes pair them. tokenizer_mode is left to
# vLLM, which picks deepseek_v4 for DeepSeek-V4 by architecture. vLLM's
# DeepSeek-V4 tokenizer maps every reasoning_effort itself. Families whose
# templates ignore reasoning_effort get no mapping.
CHAT_MODELS = MappingProxyType(
    {
        "Qwen3ForCausalLM": _QWEN3,
        "Qwen3MoeForCausalLM": _QWEN3,
        "Qwen3_5ForCausalLM": _QWEN3_5,
        "Qwen3_5ForConditionalGeneration": _QWEN3_5,
        "Qwen3_5MoeForCausalLM": _QWEN3_5,
        "Qwen3_5MoeForConditionalGeneration": _QWEN3_5,
        # Qwen3.8-Flash-Next.
        "Qwen4ExpForCausalLM": _QWEN3_5,
        "Qwen4ExpForConditionalGeneration": _QWEN3_5,
        # GLM-5.2's template has High for "high" and Max for anything else (its
        # default), so lower requests map to High rather than Max. GLM-5's and
        # 5.1's ignore the value.
        "GlmMoeDsaForCausalLM": ChatModel(
            "glm47",
            "glm47",
            thinking_optional=True,
            reasoning_efforts=MappingProxyType(
                {"minimal": "high", "low": "high", "medium": "high", "xhigh": "max"}
            ),
        ),
        # GLM-4.5 to 4.7; vLLM's glm45 parsers are the glm47 ones.
        "Glm4MoeForCausalLM": ChatModel("glm47", "glm47", thinking_optional=True),
        "Glm5NextForCausalLM": _GLM5_NEXT,
        "Glm5NextForConditionalGeneration": _GLM5_NEXT,
        "DeepseekV4ForCausalLM": _DEEPSEEK_V4,
        "DeepseekV4ForConditionalGeneration": _DEEPSEEK_V4,
        # Harmony takes low, medium and high, and gpt-oss always reasons.
        "GptOssForCausalLM": ChatModel(
            "openai_gptoss",
            "openai",
            reasoning_efforts=MappingProxyType({"minimal": "low", "xhigh": "high", "max": "high"}),
        ),
        # MiniMax-M2 and M2.1 always think.
        "MiniMaxM2ForCausalLM": ChatModel("minimax_m2", "minimax_m2"),
        # Arcee Trinity, as Trinity-Large-Preview's template writes it: no
        # thinking, hermes JSON tool calls. Its checkpoints differ under one
        # architecture: Trinity-Large-Thinking opens <think> and writes XML tool
        # calls (chat_reasoning_parser=deepseek_r1, tool_call_parser=qwen3_coder);
        # Trinity-Mini opens <think> with hermes calls (chat_reasoning_parser=
        # deepseek_r1). A reasoner on a model that never closes </think> would
        # hide its whole answer as reasoning.
        "AfmoeForCausalLM": ChatModel(None, "hermes", thinking_optional=True),
        # Nemotron 3 Nano and Super.
        "NemotronHForCausalLM": ChatModel("nemotron_v3", "qwen3_coder", thinking_optional=True),
    }
)


def resolve_chat_model(architecture, *, reasoning_parser=None, tool_call_parser=None):
    """Chat settings for an architecture, with explicit parser names taking precedence.

    None when chat mode is unsupported: the architecture is not in
    ``CHAT_MODELS`` and no parser was given.
    """
    model = CHAT_MODELS.get(architecture)
    if model is None:
        if reasoning_parser is None and tool_call_parser is None:
            return None
        # With no reasoner named, there is no thinking to turn off.
        return ChatModel(
            reasoning_parser=reasoning_parser,
            tool_call_parser=tool_call_parser,
            thinking_optional=reasoning_parser is None,
        )
    if reasoning_parser is not None:
        model = replace(model, reasoning_parser=reasoning_parser)
    if tool_call_parser is not None:
        model = replace(model, tool_call_parser=tool_call_parser)
    return model


@functools.cache
def without_tool_parser(parser_cls):
    """``parser_cls`` (a vLLM unified Parser class) with no tool parser, or None if nothing is left.

    vllm serve parses tool calls whenever a tool parser is configured, so a
    model that writes tool-call markup unprompted would get a tool call and
    finish with ``tool_calls`` though no tools were offered.
    """
    if parser_cls is None or parser_cls.tool_parser_cls is None:
        return parser_cls
    if parser_cls.reasoning_parser_cls is None:
        return None
    # Parser.__init__ builds its parts from these class attributes.
    return type(parser_cls.__name__, (parser_cls,), {"tool_parser_cls": None})


def has_chat_template(renderer, tokenizer, model_config, *, harmony):
    """Whether vLLM finds a chat template to render this model's prompts with."""
    from vllm.renderers.hf import HfRenderer, resolve_chat_template

    if harmony or not isinstance(renderer, HfRenderer):
        # Harmony and the other renderers (DeepSeek-V4's) build prompts in code.
        return True
    return resolve_chat_template(tokenizer, None, None, model_config=model_config) is not None


class ChatInputError(ValueError):
    """A chat request the model cannot take; ``param`` names the offending field."""

    def __init__(self, code, param=None):
        super().__init__(code)
        self.code = code
        self.param = param


@dataclass(frozen=True)
class ChatPrompt:
    """Chat messages for the worker to render, instead of a prepared prompt.

    Fields follow OpenAI Chat Completions. ``reasoning_effort`` reaches the
    template as the model's ``ChatModel`` maps it.
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


def check_chat_input(prompt, structured_outputs=None):
    """Checks vLLM's renderer would not make: text-only content, and one grammar.

    Content must be a string or a list of text parts. A forced tool call has its
    own grammar, and vLLM would silently drop a JSON format given with one.
    """
    if structured_outputs is not None and prompt.tools and (
        prompt.tool_choice == "required" or isinstance(prompt.tool_choice, dict)
    ):
        raise ChatInputError("invalid_chat_request", "structured_outputs")
    for index, message in enumerate(prompt.messages):
        content = message.get("content")
        if content is None or isinstance(content, str):
            continue
        if not isinstance(content, list) or any(
            not isinstance(part, str)
            and not (isinstance(part, dict) and part.get("type") in TEXT_PART_TYPES)
            for part in content
        ):
            raise ChatInputError("invalid_chat_request", f"messages[{index}].content")


# Plain text for each message of the probe conversation, in order. The two
# user turns let the probe see what opens a user turn after each other role.
PROBE_MESSAGES = (
    ("system", "ProbeSystemText"),
    ("user", "ProbeUserText"),
    ("assistant", "ProbeAssistantText"),
    ("user", "ProbeLastUserText"),
)


def _added_tokens(tokenizer):
    """``{token_id: (text, special)}`` for every added token of ``tokenizer``."""
    added = {
        token_id: (token.content, bool(getattr(token, "special", False)))
        for token_id, token in (getattr(tokenizer, "added_tokens_decoder", None) or {}).items()
    }
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    for text, token_id in (get_added_vocab() if get_added_vocab else {}).items():
        added.setdefault(token_id, (text, False))
    return added


def structural_tokens(tokenizer, renders):
    """Added tokens a template writes around the probe's text, from its ``renders``.

    ``renders`` holds the prompt token ids of ``PROBE_MESSAGES`` as rendered
    under each setting probed. Returns the ids of added tokens outside the
    messages' text, and whether one that isn't special opens a user turn.
    Raises ValueError if a render lacks the probe's text.
    """
    added = _added_tokens(tokenizer)
    structural = set()
    plain_turn_opener = False
    for token_ids in renders:
        pieces = [tokenizer.decode([token_id]) for token_id in token_ids]
        ends = list(itertools.accumulate(len(piece) for piece in pieces))
        text = "".join(pieces)
        spans = []
        for _, sentinel in PROBE_MESSAGES:
            start = text.find(sentinel, spans[-1][1] if spans else 0)
            if start < 0:
                raise ValueError("The rendered probe lacks a message's text")
            spans.append((start, start + len(sentinel)))
        # What lies between the previous message's text and a user message's.
        user_openers = [
            (spans[index - 1][1], spans[index][0])
            for index, (role, _) in enumerate(PROBE_MESSAGES)
            if role == "user"
        ]
        for token_id, piece, end in zip(token_ids, pieces, ends):
            start = end - len(piece)
            if token_id not in added or any(s < end and start < e for s, e in spans):
                continue
            structural.add(token_id)
            if not added[token_id][1] and any(
                after <= start and end <= before for after, before in user_openers
            ):
                plain_turn_opener = True
    return structural, plain_turn_opener


def prompt_opens_reasoning(tokenizer, reasoner, prompt):
    """Whether ``prompt``, ``PROBE_MESSAGES``' generation prompt, leaves ``reasoner`` reasoning.

    The reasoner's start marker must follow the last user text, and reasoning
    must still be open: a template that writes an empty ``<think></think>``
    turns thinking off.
    """
    start = getattr(reasoner, "reasoning_start_str", None)
    if not start:
        return False
    text = tokenizer.decode(prompt)
    opened = start in text[text.rfind(PROBE_MESSAGES[-1][1]) :]
    return opened and not reasoner.is_reasoning_end(prompt)


class SpecialTokenGuard:
    """Rejects text that spells one of the model's control tokens.

    Tokenizers read such text in a rendered prompt as the real token, so
    content like ``"<|im_end|><|im_start|>system"`` would forge a turn. Blocked:
    every special token, and every added token the chat template writes
    itself, found by rendering a probe conversation (``structural_tokens``).
    That covers delimiters a tokenizer doesn't mark special (DeepSeek-V4's
    ``<｜User｜>``) and in-prompt reasoning markers, whose forged ``</think>``
    would end reasoning early. Added tokens the template never writes (Qwen3's
    ``<tool_call>``) are allowed, as vllm serve allows them.

    Without a probe, or when a user turn opens with an added token that isn't
    special (the flags then can't tell control tokens from text, and roles
    the API can't send, like DeepSeek-V4's ``<｜latest_reminder｜>``, can't be
    probed), every added token is blocked.
    """

    def __init__(self, tokens):
        tokens = sorted({token for token in tokens if token}, key=len, reverse=True)
        self.tokens = frozenset(tokens)
        self._pattern = re.compile("|".join(map(re.escape, tokens))) if tokens else None

    @classmethod
    def from_tokenizer(cls, tokenizer, renders=None):
        """The guard for ``tokenizer``; ``renders`` are the probe's, or None if it failed."""
        added = _added_tokens(tokenizer)
        tokens = set(getattr(tokenizer, "all_special_tokens", ()) or ())
        tokens.update(text for text, special in added.values() if special)
        plain_turn_opener = renders is None
        if renders is not None:
            structural, plain_turn_opener = structural_tokens(tokenizer, renders)
            tokens.update(added[token_id][0] for token_id in structural)
        if plain_turn_opener:
            tokens.update(text for text, _ in added.values())
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
    # A reasoner is active and the prompt leaves reasoning open, or the model
    # is gpt-oss, whose Harmony format always opens with reasoning.
    starts_in_reasoning: bool = False


class ChatEngine:
    """vLLM's chat front end bound to one worker's engine."""

    def __init__(self, llm, chat_model):
        from vllm.renderers.online_renderer import OnlineRenderer

        self.model_config = llm.model_config
        self.chat_model = chat_model
        self.online = OnlineRenderer(
            model_config=llm.model_config,
            renderer=llm.renderer,
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
            enable_auto_tools=chat_model.tool_call_parser is not None,
            tool_parser=chat_model.tool_call_parser,
            reasoning_parser=chat_model.reasoning_parser,
        )
        self.harmony = self.online.use_harmony
        self.tokenizer = llm.renderer.get_tokenizer()
        # The renderer's own unified Parser class, so rendering and parsing agree.
        self.parser_cls = self.online.parser
        # Set by probe().
        self.guard = None
        self._probe_task = None
        self.has_chat_template = has_chat_template(
            llm.renderer, self.tokenizer, llm.model_config, harmony=self.harmony
        )

    async def probe(self):
        """Fit ``guard`` and ``chat_model`` to the model's template, rendering it once.

        Run on first use: the renderer is async, so the constructor can't.
        """
        if self.guard is None:
            if self._probe_task is None:
                self._probe_task = asyncio.ensure_future(self._probe())
            # Shielded: a cancelled stream must not cancel the shared probe.
            await asyncio.shield(self._probe_task)

    async def _probe(self):
        try:
            if self.chat_model.thinking_optional and await self._template_forces_thinking():
                # The table can only narrow what the template does: GLM-5.3
                # shares GLM-5.2's architecture but always opens <think>.
                logger.warning(
                    "The chat template opens reasoning despite enable_thinking=False; "
                    "refusing reasoning_effort='none'"
                )
                self.chat_model = replace(self.chat_model, thinking_optional=False)
            self.guard = SpecialTokenGuard.from_tokenizer(
                self.tokenizer, await self._render_probe_settings()
            )
        except Exception as exc:
            logger.warning(
                "Chat template probe failed (%s); chat rejects text spelling any added "
                "token and refuses reasoning_effort='none'",
                type(exc).__name__,
            )
            self.chat_model = replace(self.chat_model, thinking_optional=False)
            self.guard = SpecialTokenGuard.from_tokenizer(self.tokenizer)
            return
        logger.info(
            "Chat rejects text spelling these tokens beyond the special ones: %s",
            sorted(self.guard.tokens - SpecialTokenGuard.from_tokenizer(self.tokenizer, []).tokens),
        )

    async def _template_forces_thinking(self):
        """Whether the generation prompt opens reasoning even with enable_thinking=False.

        Identical prompts alone could also mean a model that never thinks
        (Qwen3-Instruct-2507 under Qwen3's architecture), so the prompt must
        also leave the reasoner's start marker open.
        """
        on, off = [
            await self._render_probe(chat_template_kwargs={"enable_thinking": enable})
            for enable in (True, False)
        ]
        if on != off or self.parser_cls is None:
            return False
        # Built as for a thinking request: glm47's parser has no start marker
        # when enable_thinking is False.
        reasoner = self.parser_cls(
            self.tokenizer,
            None,
            chat_template_kwargs={"enable_thinking": True},
            model_config=self.model_config,
        ).reasoning_parser
        return prompt_opens_reasoning(self.tokenizer, reasoner, off)

    async def _render_probe_settings(self):
        """``PROBE_MESSAGES`` rendered as requests can render them.

        With and without the generation prompt, at the default and a set
        reasoning effort, and with thinking off where the model allows it.
        """
        efforts = [None, "medium"] + (["none"] if self.chat_model.thinking_optional else [])
        renders = []
        for effort, generation_prompt in itertools.product(efforts, (True, False)):
            fields = {"add_generation_prompt": generation_prompt}
            if effort is not None:
                fields["reasoning_effort"] = self.chat_model.template_reasoning_effort(effort)
            renders.append(await self._render_probe(**fields))
        return renders

    async def _render_probe(self, **fields):
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.entrypoints.serve.engine.protocol import ErrorResponse

        request = ChatCompletionRequest(
            model=self.model_config.model,
            messages=[{"role": role, "content": text} for role, text in PROBE_MESSAGES],
            **fields,
        )
        result = await self.online.render_chat(request)
        if isinstance(result, ErrorResponse):
            raise ValueError("The renderer refused the probe")
        _, (engine_input,) = result
        return list(engine_input["prompt_token_ids"])

    async def render(self, prompt, structured_outputs=None):
        """Render ``prompt``; ``structured_outputs`` is the stream's JSON format, if any.

        The format goes on the request, as vllm serve puts ``response_format``
        there, so the parsers' ``adjust_request`` fits it to the model (Harmony
        wraps it in its final channel) and combines it with any tool grammar.
        """
        from jinja2 import TemplateError
        from vllm.entrypoints.chat_utils import ChatTemplateResolutionError
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        from vllm.entrypoints.serve.engine.protocol import ErrorResponse
        from vllm.exceptions import VLLMClientError
        from vllm.renderers.inputs.preprocess import extract_prompt_len

        check_chat_input(prompt, structured_outputs)
        await self.probe()
        self.guard.check(prompt)
        fields = {
            "model": self.model_config.model,
            "messages": prompt.messages,
            "tools": prompt.tools,
            "tool_choice": prompt.tool_choice,
            "parallel_tool_calls": prompt.parallel_tool_calls,
            "reasoning_effort": self.chat_model.template_reasoning_effort(
                prompt.reasoning_effort
            ),
            "structured_outputs": structured_outputs,
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
        # Without tools, tool-call markup the model writes is content, as OpenAI returns it.
        parser_cls = self.parser_cls if request.tools else without_tool_parser(self.parser_cls)
        if parser_cls is None:
            def new_parser():
                return None
        else:
            chat_template_kwargs = request.build_chat_params(
                None, "auto"
            ).chat_template_kwargs

            def new_parser():
                return parser_cls(
                    self.tokenizer,
                    request.tools,
                    chat_template_kwargs=chat_template_kwargs,
                    model_config=self.model_config,
                )

            probe = new_parser()
            if probe.reasoning_parser is not None:
                # Lets structured outputs start after reasoning, as vllm serve does.
                generate_kwargs["reasoning_ended"] = probe.is_reasoning_end(
                    list(engine_input.get("prompt_token_ids") or ())
                )
                generate_kwargs["reasoning_parser_kwargs"] = {
                    "chat_template_kwargs": chat_template_kwargs
                }

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
            # gpt-oss's reasoner reports reasoning as ended before any token,
            # since it only detects boundaries, though Harmony always reasons.
            starts_in_reasoning=self.harmony or generate_kwargs.get("reasoning_ended") is False,
        )


def content_token_ids(parser, token_ids):
    """The content tokens of a delta that ends reasoning, split by ``parser``'s reasoner.

    vLLM's unified parsers keep this split private and delegate it to their
    reasoning parser, so that is called directly.
    """
    reasoner = parser.reasoning_parser
    return token_ids if reasoner is None else reasoner.extract_content_ids(token_ids)


class ChatOutput:
    """Turns one stream's engine deltas into content, reasoning and tool-call events.

    Reasoning the parser splits out is never emitted as text: OpenAI's Chat
    Completions API returns only its token count. Deltas the parser holds back
    (markup it is still matching) emit nothing until they resolve.

    With logprobs, a content delta carries its own engine delta's tokens plus
    the held-back deltas it released: the latest held deltas whose text the
    content repeats just before this delta's text. Held deltas it does not
    repeat were markup, and a reasoning or tool-call delta discards them, so
    reasoning and tool-call tokens carry no logprobs.
    """

    def __init__(self, rendered, n):
        self.rendered = rendered
        self.parsers = [rendered.new_parser() for _ in range(n)]
        self.token_ids = [[] for _ in range(n)]
        self.called_tools = [False] * n
        # Per choice, (text, token_ids, logprobs) of deltas that emitted nothing.
        self.held = [[] for _ in range(n)]

    def events(self, index, text, token_ids, finished, logprobs=None):
        """``logprobs``, when requested, go with the content these tokens produced."""
        self.token_ids[index].extend(token_ids)
        events = self._parse(index, text, token_ids, finished, logprobs)
        if logprobs is not None:
            if events:
                self.held[index].clear()
            elif not finished:
                self.held[index].append((text, list(token_ids), logprobs))
        return events

    def _parse(self, index, text, token_ids, finished, logprobs):
        parser = self.parsers[index]
        if parser is None:
            return [self._content(index, text, text, token_ids, logprobs)] if text else []
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
        if getattr(message, "reasoning", None):
            token_count = len(token_ids)
            if content or tool_calls:
                # The delta that ends reasoning; split it as vLLM's parser does.
                try:
                    token_count -= len(content_token_ids(parser, list(token_ids)))
                except NotImplementedError:
                    # gpt-oss's parser can't split one; it counts as reasoning.
                    pass
            events.append(
                {"type": "reasoning_delta", "choice_index": index, "token_count": token_count}
            )
        if content:
            events.append(self._content(index, content, text, token_ids, logprobs))
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

    def _content(self, index, content, delta_text, token_ids, logprobs):
        event = {"type": "content_delta", "choice_index": index, "text": content}
        if logprobs is not None:
            released = []
            if content.endswith(delta_text):
                before = content[: len(content) - len(delta_text)]
                for held in reversed(self.held[index]):
                    if not before.endswith(held[0]):
                        break
                    before = before[: len(before) - len(held[0])]
                    released.append(held)
            released.reverse()
            event["token_ids"] = [t for held in released for t in held[1]] + list(token_ids)
            event["logprobs"] = [e for held in released for e in held[2]] + list(logprobs)
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
