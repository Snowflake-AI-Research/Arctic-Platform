from __future__ import annotations

import hashlib
import json
import os
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from arctic_platform.inference.server.action_masks import (
    ActionMaskEntry,
    ActionMasks,
    action_mask_entry_from_bitmask_words,
    action_masks_from_entries,
)


class ActionMaskReplayError(ValueError):
    pass


@dataclass(frozen=True)
class _ReplayGrammar:
    compiled: object
    vocab_size: int
    kind: str
    cache_size_bytes: int


@dataclass(frozen=True)
class _GrammarTokenSpan:
    start: int
    end: int
    source: str
    boundary_index: int | None = None
    boundary_token_id: int | None = None


_TOKENIZER_CONTEXTS: dict[int, tuple[object, object, int]] = {}
_COMPILED_GRAMMAR_CACHE_MB_ENV = (
    "ARCTIC_INFERENCE_ACTION_MASK_GRAMMAR_CACHE_MB"
)
_COMPILED_GRAMMAR_CACHE_MAX_ENTRIES_ENV = (
    "ARCTIC_INFERENCE_ACTION_MASK_GRAMMAR_CACHE_SIZE"
)
_DEFAULT_COMPILED_GRAMMAR_CACHE_MB = 4096
_DEFAULT_COMPILED_GRAMMAR_CACHE_MAX_ENTRIES = 1024
_COMPILED_GRAMMARS: OrderedDict[
    tuple[int, tuple[str, str, bool]], _ReplayGrammar
] = OrderedDict()
_COMPILED_GRAMMAR_CACHE_BYTES = 0
_COMPILED_GRAMMAR_CACHE_PEAK_BYTES = 0
_COMPILED_GRAMMAR_EVICTIONS = 0
_COMPILED_GRAMMAR_OVERSIZED_SKIPS = 0
_COMPILED_GRAMMAR_UNMEASURABLE_SKIPS = 0


def _nonnegative_env_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.environ.get(name, default)))
    except ValueError:
        return default


def _compiled_grammar_cache_budget_bytes() -> int:
    cache_mb = _nonnegative_env_int(
        _COMPILED_GRAMMAR_CACHE_MB_ENV,
        _DEFAULT_COMPILED_GRAMMAR_CACHE_MB,
    )
    return cache_mb * 1024 * 1024


def _compiled_grammar_cache_max_entries() -> int:
    return _nonnegative_env_int(
        _COMPILED_GRAMMAR_CACHE_MAX_ENTRIES_ENV,
        _DEFAULT_COMPILED_GRAMMAR_CACHE_MAX_ENTRIES,
    )


def _compiled_grammar_size_bytes(compiled: object, spec: str) -> int | None:
    memory_size_bytes = getattr(compiled, "memory_size_bytes", None)
    if not callable(memory_size_bytes):
        return None
    try:
        native_bytes = int(memory_size_bytes())
    except (OverflowError, RuntimeError, TypeError, ValueError):
        return None
    if native_bytes < 0:
        return None
    return native_bytes + len(spec.encode("utf-8"))


def _cache_compiled_grammar(
    cache_key: tuple[int, tuple[str, str, bool]],
    replay_grammar: _ReplayGrammar,
) -> None:
    global _COMPILED_GRAMMAR_CACHE_BYTES
    global _COMPILED_GRAMMAR_CACHE_PEAK_BYTES
    global _COMPILED_GRAMMAR_EVICTIONS
    global _COMPILED_GRAMMAR_OVERSIZED_SKIPS

    budget_bytes = _compiled_grammar_cache_budget_bytes()
    max_entries = _compiled_grammar_cache_max_entries()
    if budget_bytes == 0 or max_entries == 0:
        return
    if replay_grammar.cache_size_bytes > budget_bytes:
        _COMPILED_GRAMMAR_OVERSIZED_SKIPS += 1
        return

    existing = _COMPILED_GRAMMARS.pop(cache_key, None)
    if existing is not None:
        _COMPILED_GRAMMAR_CACHE_BYTES -= existing.cache_size_bytes
    while _COMPILED_GRAMMARS and (
        len(_COMPILED_GRAMMARS) >= max_entries
        or _COMPILED_GRAMMAR_CACHE_BYTES + replay_grammar.cache_size_bytes
        > budget_bytes
    ):
        _, evicted = _COMPILED_GRAMMARS.popitem(last=False)
        _COMPILED_GRAMMAR_CACHE_BYTES -= evicted.cache_size_bytes
        _COMPILED_GRAMMAR_EVICTIONS += 1

    _COMPILED_GRAMMARS[cache_key] = replay_grammar
    _COMPILED_GRAMMAR_CACHE_BYTES += replay_grammar.cache_size_bytes
    _COMPILED_GRAMMAR_CACHE_PEAK_BYTES = max(
        _COMPILED_GRAMMAR_CACHE_PEAK_BYTES,
        _COMPILED_GRAMMAR_CACHE_BYTES,
    )


def action_mask_replay_cache_stats() -> dict[str, int]:
    return {
        "tokenizer_contexts": len(_TOKENIZER_CONTEXTS),
        "compiled_grammars": len(_COMPILED_GRAMMARS),
        "compiled_grammar_cache_bytes": _COMPILED_GRAMMAR_CACHE_BYTES,
        "compiled_grammar_cache_peak_bytes": _COMPILED_GRAMMAR_CACHE_PEAK_BYTES,
        "compiled_grammar_cache_budget_bytes": (
            _compiled_grammar_cache_budget_bytes()
        ),
        "compiled_grammar_cache_max_entries": (
            _compiled_grammar_cache_max_entries()
        ),
        "compiled_grammar_evictions": _COMPILED_GRAMMAR_EVICTIONS,
        "compiled_grammar_oversized_skips": (
            _COMPILED_GRAMMAR_OVERSIZED_SKIPS
        ),
        "compiled_grammar_unmeasurable_skips": (
            _COMPILED_GRAMMAR_UNMEASURABLE_SKIPS
        ),
    }


def _xgrammar() -> Any:
    import xgrammar as xgr

    return xgr


def _tokenizer_vocab_size(tokenizer: Any) -> int:
    try:
        return int(len(tokenizer))
    except TypeError:
        pass
    vocab = getattr(tokenizer, "vocab", None)
    if isinstance(vocab, Mapping):
        return int(len(vocab))
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        return int(len(get_vocab()))
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        raise ActionMaskReplayError("Unable to determine tokenizer vocab size")
    return int(vocab_size)


def _compiler_for_tokenizer(tokenizer: Any) -> tuple[object, int]:
    key = id(tokenizer)
    cached = _TOKENIZER_CONTEXTS.get(key)
    if cached is not None and cached[0] is tokenizer:
        return cached[1], cached[2]

    xgr = _xgrammar()
    vocab_size = _tokenizer_vocab_size(tokenizer)
    tokenizer_info = None
    try:
        from vllm.utils.mistral import is_mistral_tokenizer

        if is_mistral_tokenizer(tokenizer):
            # Match vLLM's XgrammarBackend tokenizer setup for Mistral/Tekken
            # tokenizers so replay sees the same vocab/byte-fallback semantics.
            vocab = getattr(tokenizer, "vocab")
            vocab_size = len(vocab)
            tokenizer_info = xgr.TokenizerInfo(
                encoded_vocab=vocab,
                vocab_type=(
                    xgr.VocabType.RAW
                    if getattr(tokenizer, "is_tekken", False)
                    else xgr.VocabType.BYTE_FALLBACK
                ),
                vocab_size=vocab_size,
                stop_token_ids=[tokenizer.eos_token_id],
                add_prefix_space=True,
            )
    except Exception:
        tokenizer_info = None

    if tokenizer_info is None:
        try:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(
                tokenizer,
                vocab_size=vocab_size,
            )
        except TypeError:
            tokenizer_info = xgr.TokenizerInfo.from_huggingface(tokenizer)
            vocab_size = int(getattr(tokenizer_info, "vocab_size", vocab_size))

    # The replay cache owns compiled grammars so its eviction policy is the
    # only source of retention in this process.
    compiler = xgr.GrammarCompiler(
        tokenizer_info,
        max_threads=8,
        cache_enabled=False,
    )
    _TOKENIZER_CONTEXTS[key] = (tokenizer, compiler, vocab_size)
    return compiler, vocab_size


def _json_spec(value: object) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value)


def _structured_output_key(params: Any) -> tuple[str, str, bool] | None:
    if params is None:
        return None
    all_none = getattr(params, "all_constraints_none", None)
    if callable(all_none) and all_none():
        return None

    disable_any_whitespace = bool(getattr(params, "disable_any_whitespace", False))
    if getattr(params, "json", None) is not None:
        return "json", _json_spec(getattr(params, "json")), disable_any_whitespace
    if getattr(params, "json_object", None):
        return "json_object", "", disable_any_whitespace
    if getattr(params, "regex", None) is not None:
        return "regex", str(getattr(params, "regex")), disable_any_whitespace
    if getattr(params, "choice", None) is not None:
        try:
            from vllm.v1.structured_output.utils import choice_as_grammar
        except Exception as exc:
            raise ActionMaskReplayError(
                "Unable to compile structured-output choice: vLLM choice helper is unavailable"
            ) from exc
        return "grammar", choice_as_grammar(getattr(params, "choice")), disable_any_whitespace
    if getattr(params, "grammar", None) is not None:
        grammar = str(getattr(params, "grammar"))
        try:
            from vllm.v1.structured_output.utils import (
                convert_lark_to_ebnf,
                grammar_is_likely_lark,
            )

            if grammar_is_likely_lark(grammar):
                grammar = convert_lark_to_ebnf(grammar)
        except ImportError:
            pass
        return "grammar", grammar, disable_any_whitespace
    if getattr(params, "structural_tag", None) is not None:
        return "structural_tag", str(getattr(params, "structural_tag")), disable_any_whitespace
    return None


def _structured_output_hash(params: Any) -> str:
    key = _structured_output_key(params)
    if key is None:
        return "none"
    payload = json.dumps(key, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _compiled_grammar(tokenizer: Any, params: Any) -> _ReplayGrammar | None:
    key = _structured_output_key(params)
    if key is None:
        return None

    cache_key = (id(tokenizer), key)
    cached = _COMPILED_GRAMMARS.pop(cache_key, None)
    if cached is not None:
        _COMPILED_GRAMMARS[cache_key] = cached
        return cached

    compiler, vocab_size = _compiler_for_tokenizer(tokenizer)
    kind, spec, disable_any_whitespace = key
    xgr = _xgrammar()
    try:
        if kind == "json":
            compiled = compiler.compile_json_schema(
                spec,
                any_whitespace=not disable_any_whitespace,
            )
        elif kind == "json_object":
            compiled = compiler.compile_json_schema(
                '{"type": "object"}',
                any_whitespace=not disable_any_whitespace,
            )
        elif kind == "regex":
            compiled = compiler.compile_regex(spec)
        elif kind == "grammar":
            compiled = compiler.compile_grammar(spec)
        elif kind == "structural_tag":
            structural_tag = json.loads(spec)
            if isinstance(structural_tag, Mapping) and "structures" in structural_tag:
                tags = [
                    xgr.StructuralTagItem(
                        begin=item["begin"],
                        schema=json.dumps(item["schema"]),
                        end=item["end"],
                    )
                    for item in structural_tag["structures"]
                ]
                compiled = compiler.compile_structural_tag(
                    tags,
                    structural_tag["triggers"],
                )
            else:
                compiled = compiler.compile_structural_tag(spec)
        else:
            raise ActionMaskReplayError(f"Unsupported structured-output constraint: {kind}")
    except Exception as exc:
        raise ActionMaskReplayError(
            f"Failed to compile structured-output {kind} constraint for action-mask replay: {exc}"
        ) from exc

    cache_size_bytes = _compiled_grammar_size_bytes(compiled, spec)
    replay_grammar = _ReplayGrammar(
        compiled=compiled,
        vocab_size=int(vocab_size),
        kind=kind,
        cache_size_bytes=cache_size_bytes or 0,
    )
    if cache_size_bytes is None:
        global _COMPILED_GRAMMAR_UNMEASURABLE_SKIPS
        _COMPILED_GRAMMAR_UNMEASURABLE_SKIPS += 1
    else:
        _cache_compiled_grammar(cache_key, replay_grammar)
    return replay_grammar


def _find_last_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> int | None:
    if not needle or len(needle) > len(haystack):
        return None
    needle_len = len(needle)
    for start in range(len(haystack) - needle_len, -1, -1):
        if list(haystack[start : start + needle_len]) == list(needle):
            return start
    return None


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if callable(encode):
        return [int(token_id) for token_id in encode(text, add_special_tokens=False)]
    raise ActionMaskReplayError("Tokenizer does not support encode(..., add_special_tokens=False)")


def _decode_tokens(tokenizer: Any, token_ids: Sequence[int]) -> str:
    decode = getattr(tokenizer, "decode", None)
    if callable(decode):
        return str(decode(list(token_ids), skip_special_tokens=False))
    raise ActionMaskReplayError("Tokenizer does not support decode(..., skip_special_tokens=False)")


def _json_equivalent_text(left: str, right: str) -> bool:
    if left.strip() == right.strip():
        return True
    try:
        return json.loads(left) == json.loads(right)
    except json.JSONDecodeError:
        return False


def _decoded_completion_content_span(
    *,
    completion_token_ids: Sequence[int],
    tokenizer: Any,
    content_text: str,
) -> tuple[int, int] | None:
    special_ids = set(getattr(tokenizer, "all_special_ids", ()) or ())

    def trimmable(token_id: int) -> bool:
        return token_id in special_ids or not _decode_tokens(tokenizer, [token_id]).strip()

    start = 0
    end = len(completion_token_ids)
    while end > start and trimmable(int(completion_token_ids[end - 1])):
        end -= 1
    while start < end and trimmable(int(completion_token_ids[start])):
        start += 1
    if start >= end:
        return None
    decoded = _decode_tokens(tokenizer, completion_token_ids[start:end])
    if _json_equivalent_text(decoded, content_text):
        return start, end
    return None


def _extract_reasoning_content(
    reasoning_parser: Any,
    text: str,
    *,
    reasoning_ended: bool | None,
) -> tuple[str | None, str]:
    if reasoning_parser is None or not text:
        return None, text
    reasoning, content = reasoning_parser.extract_reasoning(text, None)
    if not content and reasoning_ended is True:
        return None, text
    if content is None and reasoning is None:
        return None, text
    return reasoning, content or ""


def _content_token_span(
    completion_token_ids: Sequence[int],
    *,
    text: str,
    tokenizer: Any,
    reasoning_parser: Any | None,
    reasoning_ended: bool | None,
) -> tuple[int, int]:
    if not completion_token_ids:
        return 0, 0
    if reasoning_parser is None:
        return 0, len(completion_token_ids)

    if reasoning_ended is True:
        return 0, len(completion_token_ids)

    extract_content_ids = getattr(reasoning_parser, "extract_content_ids", None)
    if callable(extract_content_ids):
        content_ids = [int(token_id) for token_id in extract_content_ids(list(completion_token_ids))]
        if content_ids:
            start = _find_last_subsequence(completion_token_ids, content_ids)
            if start is not None:
                return start, start + len(content_ids)

    _reasoning, content = _extract_reasoning_content(
        reasoning_parser,
        text,
        reasoning_ended=reasoning_ended,
    )
    if not content:
        return len(completion_token_ids), len(completion_token_ids)

    think_close_ids = _encode_text(tokenizer, "</think>")
    think_close_start = _find_last_subsequence(completion_token_ids, think_close_ids)
    if think_close_start is not None:
        start = think_close_start + len(think_close_ids)
        if start < len(completion_token_ids):
            return start, len(completion_token_ids)

    end_token_id = getattr(reasoning_parser, "end_token_id", None)
    if end_token_id is not None:
        for index in range(len(completion_token_ids) - 1, -1, -1):
            if int(completion_token_ids[index]) == int(end_token_id):
                start = index + 1
                if start < len(completion_token_ids):
                    return start, len(completion_token_ids)
                break

    decoded_span = _decoded_completion_content_span(
        completion_token_ids=completion_token_ids,
        tokenizer=tokenizer,
        content_text=content,
    )
    if decoded_span is not None:
        return decoded_span

    raise ActionMaskReplayError(
        "Could not align structured-output content with generated token ids"
    )


def _grammar_token_span(
    completion_token_ids: Sequence[int],
    *,
    prompt_token_ids: Sequence[int],
    text: str,
    tokenizer: Any,
    reasoning_parser: Any | None,
    reasoning_ended: bool | None,
    structured_output_kind: str,
) -> _GrammarTokenSpan:
    if not completion_token_ids:
        return _GrammarTokenSpan(0, 0, "empty_completion")
    if reasoning_parser is None or reasoning_ended is True:
        source = (
            "no_reasoning_parser"
            if reasoning_parser is None
            else "reasoning_ended_in_prompt"
        )
        return _GrammarTokenSpan(0, len(completion_token_ids), source)

    # Structural tags model their own begin/content/end transitions. Preserve
    # the complete parser content span so the trigger token reaches that FSM.
    if structured_output_kind == "structural_tag":
        start, end = _content_token_span(
            completion_token_ids,
            text=text,
            tokenizer=tokenizer,
            reasoning_parser=reasoning_parser,
            reasoning_ended=reasoning_ended,
        )
        return _GrammarTokenSpan(start, end, "structural_tag_content")

    initial_reasoning_ended = reasoning_ended
    if initial_reasoning_ended is None:
        is_reasoning_end = getattr(reasoning_parser, "is_reasoning_end", None)
        if callable(is_reasoning_end):
            initial_reasoning_ended = bool(
                is_reasoning_end(list(prompt_token_ids))
            )
    if initial_reasoning_ended:
        return _GrammarTokenSpan(
            0,
            len(completion_token_ids),
            "reasoning_ended_in_prompt",
        )

    is_reasoning_end_streaming = getattr(
        reasoning_parser,
        "is_reasoning_end_streaming",
        None,
    )
    if not callable(is_reasoning_end_streaming):
        start, end = _content_token_span(
            completion_token_ids,
            text=text,
            tokenizer=tokenizer,
            reasoning_parser=reasoning_parser,
            reasoning_ended=reasoning_ended,
        )
        return _GrammarTokenSpan(start, end, "parser_content")

    all_token_ids = [int(token_id) for token_id in prompt_token_ids]
    for completion_index, token_id in enumerate(completion_token_ids):
        token_id = int(token_id)
        all_token_ids.append(token_id)
        if is_reasoning_end_streaming(all_token_ids, [token_id]):
            # vLLM records the reasoning transition on this token but defers
            # ordinary JSON/regex/choice/grammar FSM advancement until the
            # following decode step.
            return _GrammarTokenSpan(
                completion_index + 1,
                len(completion_token_ids),
                "streaming_reasoning_boundary",
                boundary_index=completion_index,
                boundary_token_id=token_id,
            )

    # Reasoning never ended, so vLLM never enabled the grammar bitmask.
    return _GrammarTokenSpan(
        len(completion_token_ids),
        len(completion_token_ids),
        "reasoning_never_ended",
    )


def _replay_error_context(
    *,
    completion_ids: Sequence[int],
    completion_index: int,
    prompt_len: int,
    span: _GrammarTokenSpan,
    structured_outputs: Any,
    tokenizer: Any,
    vocab_size: int,
) -> str:
    window_start = max(0, completion_index - 8)
    window_end = min(len(completion_ids), completion_index + 9)
    window_ids = [int(token_id) for token_id in completion_ids[window_start:window_end]]
    try:
        token_text = _decode_tokens(tokenizer, [completion_ids[completion_index]])
        window_text = _decode_tokens(tokenizer, window_ids)
    except Exception as exc:
        token_text = f"<decode failed: {exc}>"
        window_text = token_text
    return (
        f"token_text={token_text!r} completion_index={completion_index} "
        f"prompt_len={prompt_len} completion_len={len(completion_ids)} "
        f"grammar_span={span.start}:{span.end} grammar_span_source={span.source} "
        f"reasoning_boundary_index={span.boundary_index} "
        f"reasoning_boundary_token_id={span.boundary_token_id} "
        f"schema_hash={_structured_output_hash(structured_outputs)} "
        f"vocab_size={vocab_size} token_window={window_start}:{window_end} "
        f"token_window_ids={window_ids} token_window_text={window_text!r}"
    )


def _bitmask_row_words(row: Any) -> np.ndarray:
    if hasattr(row, "detach"):
        row = row.detach()
    if hasattr(row, "cpu"):
        row = row.cpu()
    if hasattr(row, "numpy"):
        row = row.numpy()
    return np.asarray(row, dtype=np.uint32)


def _empty_action_masks(*, seq_len: int, vocab_size: int) -> ActionMasks:
    return {
        "seq_len": int(seq_len),
        "vocab_size": int(vocab_size),
        "positions": [],
        "set_indices": [],
        "set_modes_allow": [],
        "set_offsets": [0],
        "token_ids": [],
    }


def build_action_masks_for_output(
    *,
    prompt_token_ids: Sequence[int],
    completion_token_ids: Sequence[int],
    text: str,
    sampling_params: Any,
    tokenizer: Any,
    reasoning_parser: Any | None = None,
    reasoning_ended: bool | None = None,
    structured_outputs_enabled_in_reasoning: bool = False,
    grammar_stop_token_ids: Sequence[int] = (),
) -> ActionMasks | None:
    structured_outputs = getattr(sampling_params, "structured_outputs", None)
    replay_grammar = _compiled_grammar(tokenizer, structured_outputs)
    if replay_grammar is None:
        return None

    completion_ids = [int(token_id) for token_id in completion_token_ids]
    seq_len = len(prompt_token_ids) + len(completion_ids)
    if not completion_ids:
        return _empty_action_masks(seq_len=seq_len, vocab_size=replay_grammar.vocab_size)

    if structured_outputs_enabled_in_reasoning:
        grammar_span = _GrammarTokenSpan(
            0,
            len(completion_ids),
            "structured_outputs_enabled_in_reasoning",
        )
    else:
        grammar_span = _grammar_token_span(
            completion_ids,
            prompt_token_ids=prompt_token_ids,
            text=text,
            tokenizer=tokenizer,
            reasoning_parser=reasoning_parser,
            reasoning_ended=reasoning_ended,
            structured_output_kind=replay_grammar.kind,
        )
    content_start, content_end = grammar_span.start, grammar_span.end
    if content_start >= content_end:
        return _empty_action_masks(seq_len=seq_len, vocab_size=replay_grammar.vocab_size)

    xgr = _xgrammar()
    # Empty means the live sampler is not masking its own stop set either, so
    # replay keeps xgrammar's default (the tokenizer eos alone) to stay
    # consistent with whatever constrained the sample.
    matcher = xgr.GrammarMatcher(
        replay_grammar.compiled,
        override_stop_tokens=sorted(grammar_stop_token_ids) or None,
    )
    bitmask = xgr.allocate_token_bitmask(1, replay_grammar.vocab_size)
    entries: list[ActionMaskEntry] = []
    prompt_len = len(prompt_token_ids)

    for completion_index in range(content_start, content_end):
        token_id = completion_ids[completion_index]
        position = prompt_len + completion_index
        if token_id >= replay_grammar.vocab_size:
            context = _replay_error_context(
                completion_ids=completion_ids,
                completion_index=completion_index,
                prompt_len=prompt_len,
                span=grammar_span,
                structured_outputs=structured_outputs,
                tokenizer=tokenizer,
                vocab_size=replay_grammar.vocab_size,
            )
            raise ActionMaskReplayError(
                f"Generated token id {token_id} exceeds structured-output vocab_size={replay_grammar.vocab_size} "
                f"at full_sequence_position={position}; {context}"
            )
        xgr.reset_token_bitmask(bitmask)
        matcher.fill_next_token_bitmask(bitmask, 0)
        entry = action_mask_entry_from_bitmask_words(
            _bitmask_row_words(bitmask[0]),
            position=position,
            vocab_size=replay_grammar.vocab_size,
        )
        if entry is not None:
            entries.append(entry)
        if not matcher.accept_token(token_id):
            context = _replay_error_context(
                completion_ids=completion_ids,
                completion_index=completion_index,
                prompt_len=prompt_len,
                span=grammar_span,
                structured_outputs=structured_outputs,
                tokenizer=tokenizer,
                vocab_size=replay_grammar.vocab_size,
            )
            raise ActionMaskReplayError(
                f"Structured-output grammar rejected token id {token_id} "
                f"at full_sequence_position={position}; {context}"
            )

    return action_masks_from_entries(
        seq_len=seq_len,
        vocab_size=replay_grammar.vocab_size,
        entries=entries,
    ) or _empty_action_masks(
        seq_len=seq_len,
        vocab_size=replay_grammar.vocab_size,
    )
