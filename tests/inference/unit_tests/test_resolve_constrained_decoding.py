from __future__ import annotations

import asyncio
from types import SimpleNamespace


def _new_worker(worker_cls):
    worker = worker_cls.__new__(worker_cls)
    worker._active_lora_int_id = None
    worker._active_lora_name = None
    worker._active_lora_is_3d = False
    worker._grammar_stop_token_ids = ()
    worker._stream_cleanup_failed = False
    worker._engine_streams = {}
    return worker


class _FakeTokenizer:

    vocab_size = 16

    def __len__(self):
        return self.vocab_size

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        mapping = {
            "{": 5,
            "}": 6,
            "ok": 7,
            "<think>": 10,
            "</think>": 11,
            "r": 12,
            "<tool_call>": 13,
        }
        if text in mapping:
            return [mapping[text]]
        if text == "{}":
            return [5, 6]
        if text == "ok":
            return [7]
        return [mapping[ch] for ch in text if ch in mapping]

    def decode(self, token_ids, skip_special_tokens=False):
        assert skip_special_tokens is False
        mapping = {
            5: "{",
            6: "}",
            7: "ok",
            10: "<think>",
            11: "</think>",
            12: "r",
            13: "<tool_call>",
        }
        return "".join(mapping[int(token_id)] for token_id in token_ids)


class _FakeCompiledGrammar(dict):

    def memory_size_bytes(self):
        return 100


class _FakeCompiler:

    def compile_json_schema(self, spec, any_whitespace=True):
        return _FakeCompiledGrammar(
            kind="json", spec=spec, allowed=[{5}, {6}]
        )

    def compile_regex(self, spec):
        return _FakeCompiledGrammar(kind="regex", spec=spec, allowed=[{7}])

    def compile_grammar(self, spec):
        return _FakeCompiledGrammar(kind="grammar", spec=spec, allowed=[{7}])

    def compile_structural_tag(self, spec, triggers=None):
        if '"full": true' in str(spec):
            return _FakeCompiledGrammar(
                kind="structural_tag",
                spec=spec,
                allowed=[{12}, {7}],
            )
        return _FakeCompiledGrammar(
            kind="structural_tag", spec=spec, allowed=[{7}]
        )


class _FakeMatcher:

    def __init__(self, compiled, override_stop_tokens=None):
        self.compiled = compiled
        self.override_stop_tokens = override_stop_tokens
        self.index = 0

    def fill_next_token_bitmask(self, bitmask, row):
        bitmask[row, :] = 0
        allowed = self.compiled["allowed"][min(self.index, len(self.compiled["allowed"]) - 1)]
        for token_id in allowed:
            bitmask[row, int(token_id) // 32] |= 1 << (int(token_id) % 32)

    def accept_token(self, token_id):
        allowed = self.compiled["allowed"][min(self.index, len(self.compiled["allowed"]) - 1)]
        self.index += 1
        return int(token_id) in allowed


class _FakeXgrammar:

    GrammarMatcher = _FakeMatcher

    @staticmethod
    def allocate_token_bitmask(rows, vocab_size):
        import numpy as np

        return np.zeros((rows, (int(vocab_size) + 31) // 32), dtype=np.uint32)

    @staticmethod
    def reset_token_bitmask(bitmask):
        bitmask[:, :] = 0

    class TokenizerInfo:

        @staticmethod
        def from_huggingface(_tokenizer, vocab_size=None):
            return SimpleNamespace(vocab_size=vocab_size)

    class GrammarCompiler:

        last_init_kwargs = None

        def __init__(self, *_args, **kwargs):
            type(self).last_init_kwargs = kwargs

        def compile_json_schema(self, *args, **kwargs):
            return _FakeCompiler().compile_json_schema(*args, **kwargs)

        def compile_regex(self, *args, **kwargs):
            return _FakeCompiler().compile_regex(*args, **kwargs)

        def compile_grammar(self, *args, **kwargs):
            return _FakeCompiler().compile_grammar(*args, **kwargs)

        def compile_structural_tag(self, *args, **kwargs):
            return _FakeCompiler().compile_structural_tag(*args, **kwargs)


def _install_fake_replay(monkeypatch):
    import arctic_platform.inference.server.action_mask_replay as replay

    replay._TOKENIZER_CONTEXTS.clear()
    replay._COMPILED_GRAMMARS.clear()
    replay._COMPILED_GRAMMAR_CACHE_BYTES = 0
    replay._COMPILED_GRAMMAR_CACHE_PEAK_BYTES = 0
    replay._COMPILED_GRAMMAR_EVICTIONS = 0
    replay._COMPILED_GRAMMAR_OVERSIZED_SKIPS = 0
    replay._COMPILED_GRAMMAR_UNMEASURABLE_SKIPS = 0
    monkeypatch.setattr(replay, "_xgrammar", lambda: _FakeXgrammar)
    return replay


def test_action_mask_replay_grammar_cache_uses_memory_budget(monkeypatch):
    replay = _install_fake_replay(monkeypatch)
    monkeypatch.setattr(
        replay, "_compiled_grammar_cache_budget_bytes", lambda: 250
    )
    monkeypatch.setattr(
        replay, "_compiled_grammar_cache_max_entries", lambda: 10
    )
    tokenizer = _FakeTokenizer()

    def params(index):
        return SimpleNamespace(
            json={"const": index},
            disable_any_whitespace=False,
        )

    first = replay._compiled_grammar(tokenizer, params(1))
    replay._compiled_grammar(tokenizer, params(2))
    assert replay._compiled_grammar(tokenizer, params(1)) is first
    replay._compiled_grammar(tokenizer, params(3))

    cached_specs = [cache_key[1][1] for cache_key in replay._COMPILED_GRAMMARS]
    assert '{"const": 1}' in cached_specs
    assert '{"const": 2}' not in cached_specs
    assert '{"const": 3}' in cached_specs
    assert _FakeXgrammar.GrammarCompiler.last_init_kwargs == {
        "max_threads": 8,
        "cache_enabled": False,
    }
    stats = replay.action_mask_replay_cache_stats()
    assert stats["compiled_grammars"] == 2
    assert stats["compiled_grammar_cache_bytes"] <= 250
    assert stats["compiled_grammar_cache_peak_bytes"] <= 250
    assert stats["compiled_grammar_evictions"] == 1


def test_action_mask_replay_grammar_cache_enforces_entry_limit(monkeypatch):
    replay = _install_fake_replay(monkeypatch)
    monkeypatch.setattr(
        replay, "_compiled_grammar_cache_budget_bytes", lambda: 10_000
    )
    monkeypatch.setattr(
        replay, "_compiled_grammar_cache_max_entries", lambda: 2
    )
    tokenizer = _FakeTokenizer()

    for index in range(3):
        replay._compiled_grammar(
            tokenizer,
            SimpleNamespace(
                json={"const": index},
                disable_any_whitespace=False,
            ),
        )

    assert len(replay._COMPILED_GRAMMARS) == 2
    assert replay.action_mask_replay_cache_stats()[
        "compiled_grammar_evictions"
    ] == 1


def test_action_mask_replay_does_not_cache_grammar_over_budget(monkeypatch):
    replay = _install_fake_replay(monkeypatch)
    monkeypatch.setattr(
        replay, "_compiled_grammar_cache_budget_bytes", lambda: 50
    )
    tokenizer = _FakeTokenizer()
    params = SimpleNamespace(
        json={"type": "object"},
        disable_any_whitespace=False,
    )

    first = replay._compiled_grammar(tokenizer, params)
    second = replay._compiled_grammar(tokenizer, params)

    assert first is not second
    assert not replay._COMPILED_GRAMMARS
    assert replay.action_mask_replay_cache_stats()[
        "compiled_grammar_oversized_skips"
    ] == 2


def test_action_mask_cache_stats_reach_scheduler_metrics(monkeypatch):
    replay = _install_fake_replay(monkeypatch)
    expected = replay.action_mask_replay_cache_stats()

    import arctic_platform.inference.server.worker as worker_module
    from arctic_platform.inference.server.scheduler import Scheduler

    collector = SimpleNamespace(drain_snapshots=lambda: [{"timestamp": 1.0}])
    monkeypatch.setattr(worker_module, "get_collector", lambda: collector)
    WorkerClass = worker_module.InferenceWorker.__ray_metadata__.modified_class
    worker = _new_worker(WorkerClass)
    worker_metrics = worker.drain_metrics()
    assert worker_metrics["action_mask_replay_cache"] == expected

    class DrainRemote:

        def remote(self):
            async def result():
                return worker_metrics

            return result()

    scheduler = Scheduler.__new__(Scheduler)
    scheduler._workers = [
        SimpleNamespace(handle=SimpleNamespace(drain_metrics=DrainRemote()))
    ]
    scheduler._concurrency_history = []
    scheduler._request_records = SimpleNamespace(drain=lambda: [])

    metrics = asyncio.run(scheduler.drain_metrics())

    assert metrics["replicas"][0]["action_mask_replay_cache"] == expected


def test_vllm_qwen_parser_natively_ends_reasoning_at_tool_call():
    from vllm.parser.qwen3 import Qwen3Parser, qwen3_config
    from vllm.reasoning import ReasoningParserManager

    parser_engine = Qwen3Parser.__new__(Qwen3Parser)
    parser_engine.parser_engine_config = qwen3_config(thinking=True)
    parser_engine._reasoning_start_token_id = 10
    parser_engine._reasoning_end_token_id = 11
    parser_engine._reasoning_end_token_ids = frozenset({11, 13})
    parser_engine._tool_call_token_id = 13
    parser_engine._tool_call_end_token_id = 14
    parser_engine._turn_boundary_token_ids = frozenset()

    parser_cls = ReasoningParserManager.get_reasoning_parser("qwen3")
    parser = parser_cls.__new__(parser_cls)
    parser._parser_engine = parser_engine

    input_ids = [10, 12, 13]
    assert parser.is_reasoning_end(input_ids) is True
    assert parser.is_reasoning_end_streaming(input_ids, iter([13])) is True


def test_disabled_arctic_plugin_applies_required_runtime_patches(monkeypatch):
    from arctic_platform.inference.vllm import (
        fp32_lm_head,
        plugin,
        router_replay,
        xgrammar_stop_mask,
    )

    calls = []
    monkeypatch.setattr(plugin.envs, "ARCTIC_INFERENCE_SKIP_VERSION_CHECK", True)
    monkeypatch.setattr(plugin.envs, "ARCTIC_INFERENCE_ENABLED", False)
    monkeypatch.setattr(
        router_replay,
        "ensure_router_replay_vllm_patches",
        lambda: calls.append("router"),
    )
    monkeypatch.setattr(
        fp32_lm_head,
        "ensure_fp32_lm_head_vllm_patches",
        lambda: calls.append("fp32"),
    )
    monkeypatch.setattr(
        xgrammar_stop_mask,
        "ensure_xgrammar_stop_mask_fix",
        lambda: calls.append("xgrammar"),
    )

    plugin.arctic_inference_plugin()

    assert calls == ["router", "xgrammar", "fp32"]


def test_worker_coerces_openai_structured_outputs_for_vllm():
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    from arctic_platform.inference.server.worker import _coerce_structured_outputs_params

    params = {
        "max_tokens": 4,
        "structured_outputs": {"json": {"type": "object"}},
    }
    _coerce_structured_outputs_params(params)

    sampling_params = SamplingParams(**params)
    assert isinstance(sampling_params.structured_outputs, StructuredOutputsParams)
    assert sampling_params.structured_outputs.json == {"type": "object"}


def test_worker_coerces_guided_json_for_vllm():
    from vllm import SamplingParams

    from arctic_platform.inference.server.worker import _coerce_structured_outputs_params

    params = {
        "max_tokens": 4,
        "guided_json": {"type": "object"},
    }
    _coerce_structured_outputs_params(params)

    sampling_params = SamplingParams(**params)
    assert sampling_params.structured_outputs.json == {"type": "object"}
    assert "guided_json" not in params


def test_worker_prefills_think_for_thinking_request():
    from arctic_platform.inference.server.worker import InferenceWorker, WorkerLifecycleState

    WorkerClass = InferenceWorker.__ray_metadata__.modified_class
    captured = {}

    async def fake_generate_once(
        prompt_input, _params, _request_id, *, reasoning_ended, **_kwargs
    ):
        captured["prompt_input"] = prompt_input
        captured["reasoning_ended"] = reasoning_ended
        choice = SimpleNamespace(text="r</think>{}", token_ids=[1], finish_reason="stop", logprobs=None)
        return SimpleNamespace(outputs=[choice], prompt_token_ids=[9, 10], num_cached_tokens=0, prompt_logprobs=None)

    fake_parser = SimpleNamespace(
        start_token_id=10,
        end_token_id=11,
        is_reasoning_end=lambda _ids: False,
    )
    worker_instance = _new_worker(WorkerClass)
    worker_instance.state = WorkerLifecycleState.READY
    worker_instance._router_replay_tx = None
    worker_instance._replica_label = None
    worker_instance._reasoning_parser = fake_parser
    worker_instance._return_reasoning_content = False
    worker_instance._generate_once = fake_generate_once
    worker_instance.llm = SimpleNamespace(get_tokenizer=lambda: object())

    asyncio.run(
        worker_instance.generate(
            [1, 2],
            {
                "structured_outputs": {"json": {"type": "object"}},
                "enable_thinking": True,
                "max_tokens": 1,
            },
        )
    )

    assert captured["prompt_input"] == {"prompt_token_ids": [1, 2, 10]}
    assert captured["reasoning_ended"] is False


def test_worker_prefills_think_without_structured_outputs():
    from arctic_platform.inference.server.worker import InferenceWorker, WorkerLifecycleState

    WorkerClass = InferenceWorker.__ray_metadata__.modified_class
    captured = {}

    async def fake_generate_once(
        prompt_input, _params, _request_id, *, reasoning_ended, **_kwargs
    ):
        captured["prompt_input"] = prompt_input
        captured["reasoning_ended"] = reasoning_ended
        choice = SimpleNamespace(text="r", token_ids=[1], finish_reason="stop", logprobs=None)
        return SimpleNamespace(outputs=[choice], prompt_token_ids=[9, 10], num_cached_tokens=0, prompt_logprobs=None)

    worker_instance = _new_worker(WorkerClass)
    worker_instance.state = WorkerLifecycleState.READY
    worker_instance._router_replay_tx = None
    worker_instance._replica_label = None
    worker_instance._reasoning_parser = SimpleNamespace(start_token_id=10, end_token_id=11)
    worker_instance._return_reasoning_content = False
    worker_instance._generate_once = fake_generate_once
    worker_instance.llm = SimpleNamespace(get_tokenizer=lambda: object())

    asyncio.run(worker_instance.generate([1, 2], {"enable_thinking": True, "max_tokens": 1}))

    assert captured["prompt_input"] == {"prompt_token_ids": [1, 2, 10]}
    assert captured["reasoning_ended"] is False


def test_worker_preserves_renderer_prefilled_think_for_thinking_request():
    from arctic_platform.inference.server.worker import InferenceWorker

    WorkerClass = InferenceWorker.__ray_metadata__.modified_class
    worker_instance = _new_worker(WorkerClass)
    worker_instance._reasoning_parser = SimpleNamespace(
        start_token_id=10,
        end_token_id=11,
        is_reasoning_end=lambda _ids: False,
    )

    prompt, prompt_ids, reasoning_ended = worker_instance._prefill_think_for_generation(
        [1, 2, 10, 198],
        enable_thinking=True,
    )

    assert prompt == [1, 2, 10, 198]
    assert prompt_ids == [1, 2, 10, 198]
    assert reasoning_ended is False


def test_worker_disables_reasoning_for_structured_nothink_request():
    from arctic_platform.inference.server.worker import InferenceWorker, WorkerLifecycleState

    WorkerClass = InferenceWorker.__ray_metadata__.modified_class
    captured = {}

    async def fake_generate_once(
        prompt_input, _params, _request_id, *, reasoning_ended, **_kwargs
    ):
        captured["prompt_input"] = prompt_input
        captured["reasoning_ended"] = reasoning_ended
        choice = SimpleNamespace(text="{}", token_ids=[1], finish_reason="stop", logprobs=None)
        return SimpleNamespace(outputs=[choice], prompt_token_ids=[9], num_cached_tokens=0, prompt_logprobs=None)

    worker_instance = _new_worker(WorkerClass)
    worker_instance.state = WorkerLifecycleState.READY
    worker_instance._router_replay_tx = None
    worker_instance._replica_label = None
    worker_instance._reasoning_parser = None
    worker_instance._return_reasoning_content = False
    worker_instance._generate_once = fake_generate_once
    worker_instance.llm = SimpleNamespace(get_tokenizer=lambda: object())

    asyncio.run(
        worker_instance.generate(
            [1, 2],
            {
                "structured_outputs": {"json": {"type": "object"}},
                "enable_thinking": False,
                "max_tokens": 1,
            },
        )
    )

    assert captured["prompt_input"] == {"prompt_token_ids": [1, 2]}
    assert captured["reasoning_ended"] is True


def test_worker_uses_parser_without_returning_reasoning_or_masks():
    from arctic_platform.inference.server.worker import InferenceWorker, WorkerLifecycleState

    WorkerClass = InferenceWorker.__ray_metadata__.modified_class

    async def fake_generate_once(*args, **kwargs):
        choice = SimpleNamespace(text="<think>r</think>{}", token_ids=[1], finish_reason="stop", logprobs=None)
        return SimpleNamespace(outputs=[choice], prompt_token_ids=[9], num_cached_tokens=0, prompt_logprobs=None)

    fake_parser = SimpleNamespace(
        is_reasoning_end=lambda _ids: False,
        extract_content_ids=lambda ids: ids[3:],
        extract_reasoning=lambda _text, _request: ("r", "{}"),
    )
    worker_instance = _new_worker(WorkerClass)
    worker_instance.state = WorkerLifecycleState.READY
    worker_instance._router_replay_tx = None
    worker_instance._replica_label = None
    worker_instance._reasoning_parser = fake_parser
    worker_instance._return_reasoning_content = False
    worker_instance._generate_once = fake_generate_once
    worker_instance.llm = SimpleNamespace(get_tokenizer=lambda: object())

    result = asyncio.run(
        worker_instance.generate([9], {"structured_outputs": {"json": {"type": "object"}}, "max_tokens": 1})
    )

    assert "action_masks" not in result
    assert "reasoning" not in result
    assert "content" not in result


def test_worker_replays_action_masks_with_raw_completion_token_ids(monkeypatch):
    from arctic_platform.inference.server import action_mask_replay
    from arctic_platform.inference.server.worker import InferenceWorker, WorkerLifecycleState

    WorkerClass = InferenceWorker.__ray_metadata__.modified_class
    captured = {}

    def fake_build_action_masks_for_output(**kwargs):
        captured.update(kwargs)
        return {
            "seq_len": len(kwargs["prompt_token_ids"]) + len(kwargs["completion_token_ids"]),
            "vocab_size": 16,
            "positions": [],
            "set_indices": [],
            "set_modes_allow": [],
            "set_offsets": [0],
            "token_ids": [],
        }

    monkeypatch.setattr(action_mask_replay, "build_action_masks_for_output", fake_build_action_masks_for_output)

    async def fake_generate_once(*args, **kwargs):
        choice = SimpleNamespace(text="<think>r</think>{}", token_ids=[10, 12, 11, 5, 6], finish_reason="stop", logprobs=None)
        return SimpleNamespace(outputs=[choice], prompt_token_ids=[1, 2], num_cached_tokens=0, prompt_logprobs=None)

    fake_parser = SimpleNamespace(
        is_reasoning_end=lambda _ids: False,
        extract_content_ids=lambda ids: ids[3:],
        extract_reasoning=lambda _text, _request: ("r", "{}"),
    )
    worker_instance = _new_worker(WorkerClass)
    worker_instance.state = WorkerLifecycleState.READY
    worker_instance._router_replay_tx = None
    worker_instance._replica_label = None
    worker_instance._reasoning_parser = fake_parser
    worker_instance._return_reasoning_content = True
    worker_instance._structured_outputs_enabled_in_reasoning = False
    worker_instance._grammar_stop_token_ids = ()
    worker_instance._generate_once = fake_generate_once
    worker_instance.llm = SimpleNamespace(get_tokenizer=lambda: object())

    result = asyncio.run(
        worker_instance.generate(
            [1, 2],
            {
                "structured_outputs": {"json": {"type": "object"}},
                "return_action_masks": True,
                "max_tokens": 5,
            },
        )
    )

    assert result["token_ids"] == [10, 12, 11, 5, 6]
    assert result["content_token_ids"] == [5, 6]
    assert captured["completion_token_ids"] == [10, 12, 11, 5, 6]
    assert result["action_masks"]["seq_len"] == 7


def test_replays_action_masks_from_structured_output(monkeypatch):
    from vllm.sampling_params import StructuredOutputsParams

    replay = _install_fake_replay(monkeypatch)

    masks = replay.build_action_masks_for_output(
        prompt_token_ids=[1, 2],
        completion_token_ids=[5, 6],
        text="{}",
        sampling_params=SimpleNamespace(
            structured_outputs=StructuredOutputsParams(json={"type": "object"})
        ),
        tokenizer=_FakeTokenizer(),
    )

    assert masks == {
        "seq_len": 4,
        "vocab_size": 16,
        "positions": [2, 3],
        "set_indices": [0, 1],
        "set_modes_allow": [True, True],
        "set_offsets": [0, 1, 2],
        "token_ids": [5, 6],
    }


def test_replay_supports_xgrammar_constraint_shapes(monkeypatch):
    from vllm.sampling_params import StructuredOutputsParams

    replay = _install_fake_replay(monkeypatch)
    cases = [
        (StructuredOutputsParams(json_object=True), [5, 6], "{}"),
        (StructuredOutputsParams(regex="ok"), [7], "ok"),
        (StructuredOutputsParams(grammar='root ::= "ok"'), [7], "ok"),
        (StructuredOutputsParams(structural_tag="{}"), [7], "ok"),
        (StructuredOutputsParams(choice=["ok"]), [7], "ok"),
    ]

    for structured_outputs, token_ids, text in cases:
        replay._COMPILED_GRAMMARS.clear()
        masks = replay.build_action_masks_for_output(
            prompt_token_ids=[1],
            completion_token_ids=token_ids,
            text=text,
            sampling_params=SimpleNamespace(structured_outputs=structured_outputs),
            tokenizer=_FakeTokenizer(),
        )
        assert masks is not None


def test_replay_returns_empty_mask_for_empty_completion(monkeypatch):
    from vllm.sampling_params import StructuredOutputsParams

    replay = _install_fake_replay(monkeypatch)

    masks = replay.build_action_masks_for_output(
        prompt_token_ids=[1, 2],
        completion_token_ids=[],
        text="",
        sampling_params=SimpleNamespace(
            structured_outputs=StructuredOutputsParams(json={"type": "object"})
        ),
        tokenizer=_FakeTokenizer(),
    )

    assert masks == {
        "seq_len": 2,
        "vocab_size": 16,
        "positions": [],
        "set_indices": [],
        "set_modes_allow": [],
        "set_offsets": [0],
        "token_ids": [],
    }


def test_replay_returns_empty_mask_for_empty_content_span(monkeypatch):
    from vllm.sampling_params import StructuredOutputsParams

    replay = _install_fake_replay(monkeypatch)
    parser = SimpleNamespace(
        extract_content_ids=lambda _ids: [],
        extract_reasoning=lambda _text, _request: ("all reasoning", ""),
    )

    masks = replay.build_action_masks_for_output(
        prompt_token_ids=[1],
        completion_token_ids=[10, 12],
        text="<think>r",
        sampling_params=SimpleNamespace(
            structured_outputs=StructuredOutputsParams(json={"type": "object"})
        ),
        tokenizer=_FakeTokenizer(),
        reasoning_parser=parser,
    )

    assert masks == {
        "seq_len": 3,
        "vocab_size": 16,
        "positions": [],
        "set_indices": [],
        "set_modes_allow": [],
        "set_offsets": [0],
        "token_ids": [],
    }


def test_replay_returns_empty_mask_when_rows_are_allow_all(monkeypatch):
    from vllm.sampling_params import StructuredOutputsParams

    replay = _install_fake_replay(monkeypatch)

    class AllowAllCompiler:
        def compile_json_schema(self, spec, any_whitespace=True):
            return {"kind": "json", "spec": spec, "allowed": [set(range(_FakeTokenizer.vocab_size))]}

    class AllowAllGrammarCompiler(_FakeXgrammar.GrammarCompiler):
        def compile_json_schema(self, *args, **kwargs):
            return AllowAllCompiler().compile_json_schema(*args, **kwargs)

    class AllowAllXgrammar(_FakeXgrammar):
        GrammarCompiler = AllowAllGrammarCompiler

    monkeypatch.setattr(replay, "_xgrammar", lambda: AllowAllXgrammar)
    replay._TOKENIZER_CONTEXTS.clear()
    replay._COMPILED_GRAMMARS.clear()

    masks = replay.build_action_masks_for_output(
        prompt_token_ids=[1],
        completion_token_ids=[5],
        text="{",
        sampling_params=SimpleNamespace(
            structured_outputs=StructuredOutputsParams(json={"type": "object"})
        ),
        tokenizer=_FakeTokenizer(),
    )

    assert masks == {
        "seq_len": 2,
        "vocab_size": 16,
        "positions": [],
        "set_indices": [],
        "set_modes_allow": [],
        "set_offsets": [0],
        "token_ids": [],
    }


def test_replay_fails_when_generated_token_violates_mask(monkeypatch):
    import pytest
    from vllm.sampling_params import StructuredOutputsParams

    replay = _install_fake_replay(monkeypatch)

    with pytest.raises(replay.ActionMaskReplayError, match="rejected token id 9") as exc_info:
        replay.build_action_masks_for_output(
            prompt_token_ids=[1],
            completion_token_ids=[9],
            text="bad",
            sampling_params=SimpleNamespace(
                structured_outputs=StructuredOutputsParams(json={"type": "object"})
            ),
            tokenizer=_FakeTokenizer(),
        )
    message = str(exc_info.value)
    assert "completion_index=0" in message
    assert "prompt_len=1" in message
    assert "grammar_span=0:1" in message
    assert "grammar_span_source=no_reasoning_parser" in message
    assert "schema_hash=" in message
    assert "token_window_ids=[9]" in message


def test_replay_constrains_only_reasoning_content_span(monkeypatch):
    from vllm.sampling_params import StructuredOutputsParams

    replay = _install_fake_replay(monkeypatch)
    parser = SimpleNamespace(
        end_token_id=11,
        extract_reasoning=lambda _text, _request: ("r", "{}"),
    )

    masks = replay.build_action_masks_for_output(
        prompt_token_ids=[1],
        completion_token_ids=[10, 12, 11, 5, 6],
        text="<think>r</think>{}",
        sampling_params=SimpleNamespace(
            structured_outputs=StructuredOutputsParams(json={"type": "object"})
        ),
        tokenizer=_FakeTokenizer(),
        reasoning_parser=parser,
    )

    assert masks["positions"] == [4, 5]
    assert masks["token_ids"] == [5, 6]


def test_replay_skips_qwen_tool_call_reasoning_boundary_for_json(monkeypatch):
    from vllm.sampling_params import StructuredOutputsParams

    replay = _install_fake_replay(monkeypatch)
    parser = SimpleNamespace(
        is_reasoning_end=lambda ids: 13 in ids,
        is_reasoning_end_streaming=lambda _ids, delta_ids: 13 in delta_ids,
        # Qwen's API content includes the opener, but vLLM does not advance
        # an ordinary JSON grammar on the token that ends reasoning.
        extract_content_ids=lambda ids: ids[ids.index(13) :],
        extract_reasoning=lambda _text, _request: ("r", "<tool_call>{}"),
    )

    masks = replay.build_action_masks_for_output(
        prompt_token_ids=[1, 10],
        completion_token_ids=[12, 13, 5, 6],
        text="r<tool_call>{}",
        sampling_params=SimpleNamespace(
            structured_outputs=StructuredOutputsParams(json={"type": "object"})
        ),
        tokenizer=_FakeTokenizer(),
        reasoning_parser=parser,
        reasoning_ended=False,
    )

    assert masks["positions"] == [4, 5]
    assert masks["token_ids"] == [5, 6]


def test_replay_covers_full_completion_when_structured_outputs_are_enabled_in_reasoning(
    monkeypatch,
):
    from vllm.sampling_params import StructuredOutputsParams

    replay = _install_fake_replay(monkeypatch)
    parser = SimpleNamespace(
        extract_content_ids=lambda ids: ids[-1:],
        extract_reasoning=lambda _text, _request: ("r", "ok"),
    )

    masks = replay.build_action_masks_for_output(
        prompt_token_ids=[1],
        completion_token_ids=[12, 7],
        text="rok",
        sampling_params=SimpleNamespace(
            structured_outputs=StructuredOutputsParams(structural_tag='{"full": true}')
        ),
        tokenizer=_FakeTokenizer(),
        reasoning_parser=parser,
        reasoning_ended=False,
        structured_outputs_enabled_in_reasoning=True,
    )

    assert masks["positions"] == [1, 2]
    assert masks["token_ids"] == [12, 7]


def test_replay_error_reports_streaming_reasoning_boundary(monkeypatch):
    import pytest
    from vllm.sampling_params import StructuredOutputsParams

    replay = _install_fake_replay(monkeypatch)
    parser = SimpleNamespace(
        is_reasoning_end=lambda ids: 13 in ids,
        is_reasoning_end_streaming=lambda _ids, delta_ids: 13 in delta_ids,
    )

    with pytest.raises(replay.ActionMaskReplayError, match="rejected token id 9") as exc_info:
        replay.build_action_masks_for_output(
            prompt_token_ids=[1, 10],
            completion_token_ids=[12, 13, 9],
            text="r<tool_call>bad",
            sampling_params=SimpleNamespace(
                structured_outputs=StructuredOutputsParams(json={"type": "object"})
            ),
            tokenizer=_FakeTokenizer(),
            reasoning_parser=parser,
            reasoning_ended=False,
        )

    message = str(exc_info.value)
    assert "completion_index=2" in message
    assert "grammar_span=2:3" in message
    assert "grammar_span_source=streaming_reasoning_boundary" in message
    assert "reasoning_boundary_index=1" in message
    assert "reasoning_boundary_token_id=13" in message


def test_replay_returns_empty_masks_when_reasoning_never_ends(monkeypatch):
    from vllm.sampling_params import StructuredOutputsParams

    replay = _install_fake_replay(monkeypatch)
    parser = SimpleNamespace(
        is_reasoning_end=lambda _ids: False,
        is_reasoning_end_streaming=lambda _ids, _delta_ids: False,
    )

    masks = replay.build_action_masks_for_output(
        prompt_token_ids=[1, 10],
        completion_token_ids=[12],
        text="r",
        sampling_params=SimpleNamespace(
            structured_outputs=StructuredOutputsParams(json={"type": "object"})
        ),
        tokenizer=_FakeTokenizer(),
        reasoning_parser=parser,
        reasoning_ended=False,
    )

    assert masks == {
        "seq_len": 3,
        "vocab_size": 16,
        "positions": [],
        "set_indices": [],
        "set_modes_allow": [],
        "set_offsets": [0],
        "token_ids": [],
    }


def test_replay_uses_parser_content_ids_when_they_cover_full_completion(monkeypatch):
    from vllm.sampling_params import StructuredOutputsParams

    replay = _install_fake_replay(monkeypatch)
    parser = SimpleNamespace(
        extract_content_ids=lambda ids: ids,
        extract_reasoning=lambda _text, _request: (None, "{}"),
    )

    masks = replay.build_action_masks_for_output(
        prompt_token_ids=[1],
        completion_token_ids=[5, 6],
        text="{}",
        sampling_params=SimpleNamespace(
            structured_outputs=StructuredOutputsParams(json={"type": "object"})
        ),
        tokenizer=_FakeTokenizer(),
        reasoning_parser=parser,
    )

    assert masks["positions"] == [1, 2]
    assert masks["token_ids"] == [5, 6]


def test_replay_prefers_parser_content_ids_before_text_parsing(monkeypatch):
    from vllm.sampling_params import StructuredOutputsParams

    replay = _install_fake_replay(monkeypatch)

    def fail_extract_reasoning(_text, _request):
        raise AssertionError("extract_reasoning should not be called when token ids align")

    parser = SimpleNamespace(
        extract_content_ids=lambda _ids: [5, 6],
        extract_reasoning=fail_extract_reasoning,
    )

    masks = replay.build_action_masks_for_output(
        prompt_token_ids=[1],
        completion_token_ids=[10, 12, 11, 5, 6],
        text="<think>r</think>{}",
        sampling_params=SimpleNamespace(
            structured_outputs=StructuredOutputsParams(json={"type": "object"})
        ),
        tokenizer=_FakeTokenizer(),
        reasoning_parser=parser,
    )

    assert masks["positions"] == [4, 5]
    assert masks["token_ids"] == [5, 6]


def test_replay_does_not_retokenize_content_when_parser_ids_align(monkeypatch):
    from vllm.sampling_params import StructuredOutputsParams

    replay = _install_fake_replay(monkeypatch)
    tokenizer = _FakeTokenizer()
    original_encode = tokenizer.encode

    def encode_without_content(text, add_special_tokens=False):
        if text == "not-tokenized-content":
            raise AssertionError("content text should not be re-tokenized")
        return original_encode(text, add_special_tokens=add_special_tokens)

    tokenizer.encode = encode_without_content
    parser = SimpleNamespace(
        extract_content_ids=lambda _ids: [5, 6],
        extract_reasoning=lambda _text, _request: ("r", "not-tokenized-content"),
    )

    masks = replay.build_action_masks_for_output(
        prompt_token_ids=[1],
        completion_token_ids=[10, 12, 11, 5, 6],
        text="<think>r</think>{}",
        sampling_params=SimpleNamespace(
            structured_outputs=StructuredOutputsParams(json={"type": "object"})
        ),
        tokenizer=tokenizer,
        reasoning_parser=parser,
    )

    assert masks["positions"] == [4, 5]
    assert masks["token_ids"] == [5, 6]


def test_replay_uses_full_completion_when_reasoning_already_ended(monkeypatch):
    from vllm.sampling_params import StructuredOutputsParams

    replay = _install_fake_replay(monkeypatch)

    def fail_extract_reasoning(_text, _request):
        raise AssertionError("extract_reasoning should not be called after reasoning ended")

    parser = SimpleNamespace(
        extract_content_ids=lambda _ids: [],
        extract_reasoning=fail_extract_reasoning,
    )

    masks = replay.build_action_masks_for_output(
        prompt_token_ids=[1],
        completion_token_ids=[5, 6],
        text="{}",
        sampling_params=SimpleNamespace(
            structured_outputs=StructuredOutputsParams(json={"type": "object"})
        ),
        tokenizer=_FakeTokenizer(),
        reasoning_parser=parser,
        reasoning_ended=True,
    )

    assert masks["positions"] == [1, 2]
    assert masks["token_ids"] == [5, 6]


def test_replay_falls_back_to_think_close_when_content_text_does_not_retokenize(monkeypatch):
    from vllm.sampling_params import StructuredOutputsParams

    replay = _install_fake_replay(monkeypatch)
    parser = SimpleNamespace(
        extract_reasoning=lambda _text, _request: ("r", "not-tokenized-content"),
    )

    masks = replay.build_action_masks_for_output(
        prompt_token_ids=[1],
        completion_token_ids=[10, 12, 11, 5, 6],
        text="<think>r</think>{}",
        sampling_params=SimpleNamespace(
            structured_outputs=StructuredOutputsParams(json={"type": "object"})
        ),
        tokenizer=_FakeTokenizer(),
        reasoning_parser=parser,
    )

    assert masks["positions"] == [4, 5]
    assert masks["token_ids"] == [5, 6]


def test_replay_fails_without_parser_ids_or_reasoning_boundary(monkeypatch):
    import pytest
    from vllm.sampling_params import StructuredOutputsParams

    replay = _install_fake_replay(monkeypatch)
    parser = SimpleNamespace(
        extract_content_ids=lambda _ids: [],
        extract_reasoning=lambda _text, _request: ("r", "not-tokenized-content"),
    )

    with pytest.raises(replay.ActionMaskReplayError, match="Could not align"):
        replay.build_action_masks_for_output(
            prompt_token_ids=[1],
            completion_token_ids=[10, 12, 5, 6],
            text="<think>r{}",
            sampling_params=SimpleNamespace(
                structured_outputs=StructuredOutputsParams(json={"type": "object"})
            ),
            tokenizer=_FakeTokenizer(),
            reasoning_parser=parser,
        )


def test_action_mask_normalizer_accepts_integral_floats_and_rejects_fractional():
    import pytest

    from arctic_platform.inference.server.action_masks import normalize_action_masks

    masks = normalize_action_masks(
        {
            "seq_len": 4.0,
            "vocab_size": 16.0,
            "positions": [2.0],
            "set_indices": [0.0],
            "set_modes_allow": [True],
            "set_offsets": [0.0, 1.0],
            "token_ids": [5.0],
        }
    )

    assert masks["seq_len"] == 4
    assert isinstance(masks["seq_len"], int)
    assert masks["positions"] == [2]
    assert masks["token_ids"] == [5]
    with pytest.raises(TypeError, match="action_masks.seq_len"):
        normalize_action_masks({**masks, "seq_len": 4.5})


def test_worker_returns_action_masks_from_server_replay(monkeypatch):
    from arctic_platform.inference.server.worker import InferenceWorker, WorkerLifecycleState

    _install_fake_replay(monkeypatch)
    WorkerClass = InferenceWorker.__ray_metadata__.modified_class
    captured = {}
    expected_masks = {
        "seq_len": 3,
        "vocab_size": 16,
        "positions": [1, 2],
        "set_indices": [0, 1],
        "set_modes_allow": [True, True],
        "set_offsets": [0, 1, 2],
        "token_ids": [5, 6],
    }

    async def fake_generate_once(_prompt_input, params, _request_id, **_kwargs):
        captured["extra_args"] = params.extra_args
        choice = SimpleNamespace(text="{}", token_ids=[5, 6], finish_reason="stop", logprobs=None)
        return SimpleNamespace(
            outputs=[choice],
            prompt_token_ids=[1],
            num_cached_tokens=0,
            prompt_logprobs=None,
        )

    worker_instance = _new_worker(WorkerClass)
    worker_instance.state = WorkerLifecycleState.READY
    worker_instance._router_replay_tx = None
    worker_instance._replica_label = None
    worker_instance._reasoning_parser = None
    worker_instance._return_reasoning_content = False
    worker_instance._structured_outputs_enabled_in_reasoning = False
    worker_instance._grammar_stop_token_ids = ()
    worker_instance._generate_once = fake_generate_once
    worker_instance.llm = SimpleNamespace(get_tokenizer=lambda: _FakeTokenizer())

    result = asyncio.run(
        worker_instance.generate(
            [1],
            {
                "max_tokens": 3,
                "structured_outputs": {"json": {"type": "object"}},
                "return_action_masks": True,
            },
        )
    )

    assert result["action_masks"] == expected_masks
    assert captured["extra_args"] is None


def test_worker_can_return_reasoning_fields_when_enabled():
    from arctic_platform.inference.server.worker import InferenceWorker, WorkerLifecycleState

    WorkerClass = InferenceWorker.__ray_metadata__.modified_class

    async def fake_generate_once(*args, **kwargs):
        choice = SimpleNamespace(text="<think>r</think>{}", token_ids=[1], finish_reason="stop", logprobs=None)
        return SimpleNamespace(outputs=[choice], prompt_token_ids=[9], num_cached_tokens=0, prompt_logprobs=None)

    fake_parser = SimpleNamespace(
        is_reasoning_end=lambda _ids: False,
        extract_content_ids=lambda ids: ids,
        extract_reasoning=lambda _text, _request: ("r", "{}"),
    )
    worker_instance = _new_worker(WorkerClass)
    worker_instance.state = WorkerLifecycleState.READY
    worker_instance._router_replay_tx = None
    worker_instance._replica_label = None
    worker_instance._reasoning_parser = fake_parser
    worker_instance._return_reasoning_content = True
    worker_instance._generate_once = fake_generate_once
    worker_instance.llm = SimpleNamespace(get_tokenizer=lambda: object())

    result = asyncio.run(worker_instance.generate([9], {"max_tokens": 1}))

    assert result["reasoning"] == "r"
    assert result["content"] == "{}"
    assert result["reasoning_token_ids"] is None
    assert result["content_token_ids"] == [1]


def test_worker_returns_reasoning_and_content_token_ids_when_enabled():
    from arctic_platform.inference.server.worker import InferenceWorker, WorkerLifecycleState

    WorkerClass = InferenceWorker.__ray_metadata__.modified_class

    async def fake_generate_once(*args, **kwargs):
        choice = SimpleNamespace(text="<think>r</think>{}", token_ids=[10, 12, 11, 5, 6], finish_reason="stop", logprobs=None)
        return SimpleNamespace(outputs=[choice], prompt_token_ids=[9], num_cached_tokens=0, prompt_logprobs=None)

    fake_parser = SimpleNamespace(
        extract_content_ids=lambda ids: ids[3:],
        extract_reasoning=lambda _text, _request: ("r", "{}"),
    )
    worker_instance = _new_worker(WorkerClass)
    worker_instance.state = WorkerLifecycleState.READY
    worker_instance._router_replay_tx = None
    worker_instance._replica_label = None
    worker_instance._reasoning_parser = fake_parser
    worker_instance._return_reasoning_content = True
    worker_instance._generate_once = fake_generate_once
    worker_instance.llm = SimpleNamespace(get_tokenizer=lambda: _FakeTokenizer())

    result = asyncio.run(worker_instance.generate([9], {"max_tokens": 5}))

    assert result["reasoning"] == "r"
    assert result["content"] == "{}"
    assert result["reasoning_token_ids"] == [10, 12, 11]
    assert result["content_token_ids"] == [5, 6]


def test_reasoning_parser_empty_result_treats_text_as_content():
    from arctic_platform.inference.server.worker import _extract_reasoning_content

    reasoning, content = _extract_reasoning_content(
        SimpleNamespace(extract_reasoning=lambda _text, _request: (None, None)),
        '{"kind":"final_answer"}',
    )

    assert reasoning is None
    assert content == '{"kind":"final_answer"}'


def test_reasoning_ended_forces_text_content_when_parser_returns_reasoning():
    from arctic_platform.inference.server.worker import _extract_reasoning_content

    reasoning, content = _extract_reasoning_content(
        SimpleNamespace(extract_reasoning=lambda _text, _request: ('{"kind":"final_answer"}', None)),
        '{"kind":"final_answer"}',
        reasoning_ended=True,
    )

    assert reasoning is None
    assert content == '{"kind":"final_answer"}'


def test_reasoning_parser_reaches_vllm_structured_outputs_config():
    from arctic_platform.inference.server.worker import _create_async_engine_args

    args = _create_async_engine_args(
        {
            "model": "Qwen/Qwen3-0.6B",
            "tokenizer": "Qwen/Qwen3-0.6B",
            "max_model_len": 1024,
            "tensor_parallel_size": 1,
            "trust_remote_code": True,
            "reasoning_parser": "qwen3",
            "structured_outputs_config": {"enable_in_reasoning": True},
        }
    )
    config = args.create_engine_config()

    assert config.structured_outputs_config.reasoning_parser == "qwen3"
    assert config.structured_outputs_config.reasoning_parser_plugin == ""
    assert config.structured_outputs_config.enable_in_reasoning is True


def test_worker_does_not_inject_resolve_reasoning_parser_plugin(monkeypatch):
    from arctic_platform.inference.server.worker import InferenceWorker

    WorkerClass = InferenceWorker.__ray_metadata__.modified_class
    worker_instance = _new_worker(WorkerClass)
    worker_instance._return_reasoning_content = False
    worker_instance._cleanup_registered_router_replay_shm = lambda: None
    worker_instance._register_router_replay_shm = lambda *_args, **_kwargs: None
    worker_instance._maybe_init_router_replay_tx = lambda *_args, **_kwargs: None
    worker_instance._maybe_init_reasoning_parser = lambda *_args, **_kwargs: None

    class FakeEngineArgs:
        def __init__(self, kwargs):
            self.kwargs = kwargs

        def create_engine_config(self):
            return SimpleNamespace(
                parallel_config=SimpleNamespace(data_parallel_rank=0),
                structured_outputs_config=SimpleNamespace(enable_in_reasoning=False),
                model_config=SimpleNamespace(skip_tokenizer_init=True),
            )

    captured = {}

    def fake_create_async_engine_args(kwargs, **_ignored):
        captured.update(kwargs)
        return FakeEngineArgs(kwargs)

    class FakeAsyncLLM:
        @classmethod
        def from_vllm_config(cls, *_args, **_kwargs):
            return object()

    monkeypatch.setattr("arctic_platform.inference.server.worker._create_async_engine_args", fake_create_async_engine_args)
    monkeypatch.setattr("vllm.v1.engine.async_llm.AsyncLLM", FakeAsyncLLM)

    import asyncio

    asyncio.run(
        worker_instance.initialize(
            {
                "model": "Qwen/Qwen3-0.6B",
                "reasoning_parser": "qwen3",
            }
        )
    )

    assert captured["reasoning_parser"] == "qwen3"
    assert "reasoning_parser_plugin" not in captured
    assert worker_instance._structured_outputs_enabled_in_reasoning is False
