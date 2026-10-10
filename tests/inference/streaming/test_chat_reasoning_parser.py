"""chat_reasoning_parser: a reasoning parser for chat streams that leaves /generate as it was."""

import asyncio
from dataclasses import asdict
import os
import sys
import types
from types import SimpleNamespace

import pytest

from cpu_support import load_library

if os.environ.get("ARCTIC_RUN_GPU_TESTS") == "1":
    pytest.skip(
        "CPU fake-engine harness must run separately from GPU tests",
        allow_module_level=True,
    )

load_library()
from arctic_platform.inference.server import worker as worker_module
from arctic_platform.inference.server.chat import ChatPrompt
from arctic_platform.inference.server.streaming import StreamLimits

THINK, END_THINK = 10, 11
# A tokenizer whose vocabulary has no think tokens.
NO_THINK_TOKENS = "no-think-tokenizer"
THINK_TOKENS_MISSING = "reasoning parser could not locate think start/end tokens in the tokenizer!"
# A tokenizer the parser builds on but finds no think tokens in, as vLLM's
# MiniMaxM2AppendThinkReasoningParser does (minimax_m2_reasoning_parser.py:31).
THINK_IDS_NONE = "think-ids-none-tokenizer"
# A tokenizer the parser rejects with some other message.
REWORDED_ERROR = "reworded-error-tokenizer"


class FakeReasoningParser:
    start_token = "<think>"
    start_token_id = THINK
    end_token_id = END_THINK

    def __init__(self, tokenizer):
        if tokenizer == NO_THINK_TOKENS:
            # As vLLM's BaseThinkingReasoningParser (reasoning/basic_parsers.py:63).
            raise RuntimeError(f"FakeReasoningParser {THINK_TOKENS_MISSING}")
        if tokenizer == REWORDED_ERROR:
            raise RuntimeError("FakeReasoningParser found no reasoning markers")
        if tokenizer == THINK_IDS_NONE:
            self.start_token_id = self.end_token_id = None
        self.tokenizer = tokenizer

    def is_reasoning_end(self, token_ids):
        return END_THINK in token_ids

    def extract_reasoning(self, text, request=None):
        return "r", text


class FakeSamplingParams:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.all_stop_token_ids = set()


class FakeEngine:
    """Records what each request handed to vLLM; answers with one token."""

    def __init__(self):
        self.calls = []
        self.model_config = SimpleNamespace(max_model_len=1024)

    def get_tokenizer(self):
        return object()

    async def generate(self, prompt, params, **kwargs):
        self.calls.append((prompt, params, kwargs))
        token_ids = prompt["prompt_token_ids"] if isinstance(prompt, dict) else [1]
        yield SimpleNamespace(
            prompt_token_ids=token_ids,
            prompt_logprobs=None,
            num_cached_tokens=0,
            outputs=[
                SimpleNamespace(
                    index=0, text="ok", token_ids=[7], finish_reason="stop", logprobs=None
                )
            ],
        )

    async def abort(self, request_id):
        pass

    def get_num_unfinished_requests(self):
        return 0


@pytest.fixture
def fake_vllm(monkeypatch):
    """Just enough of vLLM for InferenceWorker.initialize and generate on CPU."""
    built = {}

    class FakeEngineArgs:
        def __init__(self, kwargs):
            self.kwargs = kwargs

        def create_engine_config(self):
            # vLLM builds the reasoning_parser kwarg's parser here
            # (config/reasoning.py:87).
            if self.kwargs.get("reasoning_parser") and built.get("tokenizer") == NO_THINK_TOKENS:
                raise RuntimeError(f"FakeReasoningParser {THINK_TOKENS_MISSING}")
            # vLLM copies reasoning_parser over structured_outputs_config's
            # (arg_utils.py:2712); gpt-oss sets one itself when neither does
            # (models/config.py:423).
            structured = self.kwargs.get("structured_outputs_config") or {}
            config = SimpleNamespace(
                parallel_config=SimpleNamespace(data_parallel_rank=0),
                structured_outputs_config=SimpleNamespace(
                    enable_in_reasoning=False,
                    reasoning_parser=self.kwargs.get("reasoning_parser")
                    or structured.get("reasoning_parser")
                    or built.get("engine_reasoner", ""),
                ),
                model_config=SimpleNamespace(
                    skip_tokenizer_init=True,
                    architecture=built.get("architecture", "MysteryForCausalLM"),
                ),
            )
            built["engine_kwargs"] = self.kwargs
            built["vllm_config"] = config
            return config

    class FakeAsyncLLM:
        @classmethod
        def from_vllm_config(cls, vllm_config, **_kwargs):
            built["reasoner_at_engine_start"] = vllm_config.structured_outputs_config.reasoning_parser
            engine = FakeEngine()
            engine.model_config.architecture = vllm_config.model_config.architecture
            return engine

    modules = {
        "vllm.plugins": {"load_general_plugins": lambda: None},
        "vllm.v1.engine": {},
        "vllm.v1.engine.async_llm": {"AsyncLLM": FakeAsyncLLM},
        "vllm.reasoning": {
            "ReasoningParserManager": SimpleNamespace(
                get_reasoning_parser=lambda name: FakeReasoningParser
            )
        },
        # None, as under skip_tokenizer_init, unless a test sets one.
        "vllm.tokenizers": {
            "cached_tokenizer_from_config": lambda model_config: built.get("tokenizer")
        },
        "vllm.sampling_params": {
            "StructuredOutputsParams": lambda **kwargs: kwargs,
            "RequestOutputKind": SimpleNamespace(DELTA="delta"),
        },
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        for key, value in attributes.items():
            setattr(module, key, value)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(sys.modules["vllm"], "plugins", sys.modules["vllm.plugins"], raising=False)
    monkeypatch.setattr(sys.modules["vllm"], "SamplingParams", FakeSamplingParams, raising=False)
    monkeypatch.setattr(worker_module, "arctic_inference_effective_enabled", lambda *a: False)
    monkeypatch.setattr(worker_module, "_ensure_router_replay_vllm_patches", lambda: None)
    monkeypatch.setattr(worker_module, "ensure_xgrammar_stop_mask_fix", lambda: None)
    monkeypatch.setattr(worker_module, "ensure_spec_decode_grammar_fix", lambda: None)
    monkeypatch.setattr(
        worker_module,
        "_create_async_engine_args",
        lambda kwargs, **_ignored: FakeEngineArgs(dict(kwargs)),
    )
    return built


def start_worker(**engine_kwargs):
    worker = worker_module.InferenceWorker.__ray_metadata__.modified_class()
    asyncio.run(worker.initialize({"model": "m", **engine_kwargs}))
    return worker


def replayed_masks(monkeypatch):
    calls = []
    module = types.ModuleType("arctic_platform.inference.server.action_mask_replay")
    module.build_action_masks_for_output = lambda **kwargs: calls.append(kwargs) or "masks"
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return calls


def generate(worker, prompt, **params):
    return asyncio.run(
        worker.generate(prompt, {"max_tokens": 4, "return_action_masks": True, **params})
    )


def test_chat_reasoning_parser_is_not_handed_to_vllm_as_an_engine_arg(fake_vllm):
    start_worker(chat_reasoning_parser="qwen3", tool_call_parser="hermes")

    assert "chat_reasoning_parser" not in fake_vllm["engine_kwargs"]
    assert "reasoning_parser" not in fake_vllm["engine_kwargs"]


def test_chat_reasoning_parser_gives_vllm_its_structured_output_reasoner(fake_vllm):
    # Chat grammars (tool_choice=required, JSON answers) wait for the end of
    # reasoning only if the engine has a reasoner (structured_output/__init__.py:88).
    start_worker(chat_reasoning_parser="qwen3")

    assert fake_vllm["reasoner_at_engine_start"] == "qwen3"


def test_chat_reasoning_parser_leaves_generate_without_a_parser(fake_vllm):
    worker = start_worker(chat_reasoning_parser="qwen3")

    assert worker._reasoning_parser is None


@pytest.mark.parametrize("enable_thinking", [None, True, False])
@pytest.mark.parametrize("prompt", [[1, 2], "hi"])
def test_generate_with_only_a_chat_parser_matches_a_worker_without_one(
    fake_vllm, monkeypatch, enable_thinking, prompt
):
    params = {} if enable_thinking is None else {"enable_thinking": enable_thinking}
    plain = start_worker()
    chat = start_worker(chat_reasoning_parser="qwen3")
    chat._return_reasoning_content = plain._return_reasoning_content = True

    plain_masks = replayed_masks(monkeypatch)
    plain_result = generate(plain, prompt, **params)
    chat_masks = replayed_masks(monkeypatch)
    chat_result = generate(chat, prompt, **params)

    [(plain_prompt, plain_params, plain_kwargs)] = plain.llm.calls
    [(chat_prompt, chat_params, chat_kwargs)] = chat.llm.calls
    # No <think> prefill and the same sampling parameters.
    assert chat_prompt == plain_prompt
    assert chat_params.kwargs == plain_params.kwargs
    # No parsed reasoning in the result.
    assert chat_result == plain_result
    # Action masks are replayed from the same inputs.
    for masks in (plain_masks, chat_masks):
        for call in masks:
            call.pop("tokenizer")
            call.pop("sampling_params")
    assert chat_masks == plain_masks
    assert chat_masks[0]["reasoning_parser"] is None


def test_generate_grammars_apply_from_the_first_token_despite_the_chat_reasoner(fake_vllm):
    # reasoning_ended=True makes vLLM constrain from token 0, exactly as with no
    # reasoner at all (structured_output/__init__.py:240 vs :243).
    plain = start_worker()
    chat = start_worker(chat_reasoning_parser="qwen3")

    generate(plain, [1, 2])
    generate(chat, [1, 2])

    assert plain.llm.calls[0][2]["reasoning_ended"] is None
    assert chat.llm.calls[0][2]["reasoning_ended"] is True


async def run_plain_stream(worker, prompt):
    worker.start_stream("attempt", prompt, {"n": 1}, 20, asdict(StreamLimits()))
    reader = worker.stream_events("attempt")
    async for batch in reader:
        if batch[-1]["type"] in ("completed", "terminal_error"):
            break
        worker.acknowledge_stream("attempt", batch[-1]["sequence"])
    await reader.aclose()


@pytest.mark.parametrize(
    ("engine_kwargs", "reasoning_ended"),
    [
        ({}, None),
        ({"chat_reasoning_parser": "qwen3"}, True),
        ({"reasoning_parser": "qwen3"}, None),
    ],
)
def test_plain_streams_keep_their_grammar_timing(fake_vllm, engine_kwargs, reasoning_ended):
    worker = start_worker(**engine_kwargs)
    worker._stream_sampling_params = lambda params: params

    asyncio.run(run_plain_stream(worker, [1, 2]))

    [(_, _, kwargs)] = worker.llm.calls
    assert kwargs.get("reasoning_ended") is reasoning_ended


def chat_parsers(monkeypatch, worker):
    """The (reasoning, tool-call) parser names the worker's ChatEngine is built with."""
    from arctic_platform.inference.server import chat as chat_module

    built = []
    monkeypatch.setattr(chat_module, "ChatEngine", lambda llm, model: built.append(model))
    worker._chat_engine()
    [model] = built
    return model.reasoning_parser, model.tool_call_parser


@pytest.mark.parametrize(
    ("engine_kwargs", "chat_parser"),
    [
        ({"chat_reasoning_parser": "qwen3"}, "qwen3"),
        ({"reasoning_parser": "qwen3"}, "qwen3"),
        ({"reasoning_parser": "qwen3", "chat_reasoning_parser": "qwen3"}, "qwen3"),
        # The engine gates grammars on this reasoner, so chat must parse with it.
        ({"structured_outputs_config": {"reasoning_parser": "qwen3"}}, "qwen3"),
        ({"model_default": "openai_gptoss"}, "openai_gptoss"),
        ({}, None),
    ],
)
def test_chat_streams_use_the_chat_parser_or_the_jobs_own(
    fake_vllm, monkeypatch, engine_kwargs, chat_parser
):
    engine_kwargs = dict(engine_kwargs)
    if "model_default" in engine_kwargs:
        fake_vllm["engine_reasoner"] = engine_kwargs.pop("model_default")
    worker = start_worker(tool_call_parser="hermes", **engine_kwargs)

    assert chat_parsers(monkeypatch, worker) == (chat_parser, "hermes")


def test_a_known_architecture_gets_its_chat_parsers_without_engine_kwargs(
    fake_vllm, monkeypatch
):
    fake_vllm["architecture"] = "Qwen3ForCausalLM"
    worker = start_worker()

    assert chat_parsers(monkeypatch, worker) == ("qwen3", "hermes")
    # For chat streams only: /generate keeps no parser and its grammar timing.
    assert fake_vllm["reasoner_at_engine_start"] == "qwen3"
    assert "reasoning_parser" not in fake_vllm["engine_kwargs"]
    assert "tokenizer_mode" not in fake_vllm["engine_kwargs"]
    assert worker._reasoning_parser is None
    generate(worker, [1, 2])
    assert worker.llm.calls[0][2]["reasoning_ended"] is True


def test_explicit_parsers_override_the_architectures(fake_vllm, monkeypatch):
    fake_vllm["architecture"] = "Qwen3ForCausalLM"
    worker = start_worker(chat_reasoning_parser="deepseek_r1", tool_call_parser="qwen3_xml")

    assert chat_parsers(monkeypatch, worker) == ("deepseek_r1", "qwen3_xml")


def test_a_jobs_own_reasoner_wins_over_the_architectures(fake_vllm, monkeypatch):
    # One reasoner per engine: the job chose it for /generate, so chat follows it
    # rather than failing a job that never asked for chat.
    fake_vllm["architecture"] = "Qwen3ForCausalLM"
    worker = start_worker(reasoning_parser="deepseek_r1")

    assert chat_parsers(monkeypatch, worker) == ("deepseek_r1", "hermes")
    assert worker._chat_only_reasoner is False


def test_a_family_without_a_reasoner_adds_none_to_the_engine(fake_vllm, monkeypatch):
    fake_vllm["architecture"] = "AfmoeForCausalLM"
    worker = start_worker()

    assert chat_parsers(monkeypatch, worker) == (None, "hermes")
    assert fake_vllm["reasoner_at_engine_start"] == ""
    assert worker._chat_only_reasoner is False


@pytest.mark.parametrize(
    ("architecture", "engine_kwargs", "support"),
    [
        ("Qwen3ForCausalLM", {}, {"chat_prompt": True, "thinking_optional": True}),
        ("GptOssForCausalLM", {}, {"chat_prompt": True, "thinking_optional": False}),
        # No reasoner: nothing to turn off.
        ("AfmoeForCausalLM", {}, {"chat_prompt": True, "thinking_optional": True}),
        ("MysteryForCausalLM", {}, {"chat_prompt": False, "thinking_optional": False}),
        (
            "MysteryForCausalLM",
            {"tool_call_parser": "hermes"},
            {"chat_prompt": True, "thinking_optional": True},
        ),
    ],
)
def test_chat_support_is_reported_per_model(
    fake_vllm, monkeypatch, architecture, engine_kwargs, support
):
    from arctic_platform.inference.server import chat as chat_module

    monkeypatch.setattr(
        chat_module, "ChatEngine", lambda llm, model: SimpleNamespace(has_chat_template=True)
    )
    fake_vllm["architecture"] = architecture
    worker = start_worker(**engine_kwargs)

    assert worker.get_chat_support() == support


def test_chat_support_is_false_without_a_chat_template(fake_vllm, monkeypatch):
    from arctic_platform.inference.server import chat as chat_module

    monkeypatch.setattr(
        chat_module, "ChatEngine", lambda llm, model: SimpleNamespace(has_chat_template=False)
    )
    fake_vllm["architecture"] = "Qwen3ForCausalLM"
    worker = start_worker()

    assert worker.get_chat_support() == {"chat_prompt": False, "thinking_optional": False}


def test_chat_support_is_false_when_the_chat_front_end_cannot_be_built(
    fake_vllm, monkeypatch
):
    from arctic_platform.inference.server import chat as chat_module

    def broken(llm, model):
        raise RuntimeError("no tokenizer")

    monkeypatch.setattr(chat_module, "ChatEngine", broken)
    fake_vllm["architecture"] = "Qwen3ForCausalLM"
    worker = start_worker()

    assert worker.get_chat_support() == {"chat_prompt": False, "thinking_optional": False}


def test_unknown_architecture_chat_is_unsupported_and_logged_once(fake_vllm, caplog):
    worker = start_worker()

    async def run_twice():
        events = []
        for attempt in ("one", "two"):
            worker.start_stream(
                attempt,
                ChatPrompt([{"role": "user", "content": "hi"}]),
                {"n": 1},
                20,
                asdict(StreamLimits()),
            )
            reader = worker.stream_events(attempt)
            async for batch in reader:
                events.extend(batch)
                break
            await reader.aclose()
        return events

    with caplog.at_level("INFO"):
        events = asyncio.run(run_twice())
    assert [(e["type"], e["code"]) for e in events] == [("terminal_error", "chat_unsupported")] * 2
    [record] = [r for r in caplog.records if "chat" in r.getMessage().lower()]
    assert "MysteryForCausalLM" in record.getMessage()


def test_a_jobs_own_reasoning_parser_still_drives_generate(fake_vllm, monkeypatch):
    worker = start_worker(reasoning_parser="qwen3", chat_reasoning_parser="qwen3")
    masks = replayed_masks(monkeypatch)

    generate(worker, [1, 2], enable_thinking=True)

    assert fake_vllm["engine_kwargs"]["reasoning_parser"] == "qwen3"
    assert isinstance(worker._reasoning_parser, FakeReasoningParser)
    [(prompt, _, kwargs)] = worker.llm.calls
    assert prompt == {"prompt_token_ids": [1, 2, THINK]}
    assert kwargs["reasoning_ended"] is False
    assert masks[0]["reasoning_parser"] is worker._reasoning_parser


@pytest.mark.parametrize(
    "engine_reasoner",
    [
        {"reasoning_parser": "qwen3"},
        {"structured_outputs_config": {"reasoning_parser": "qwen3"}},
        {"model_default": "qwen3"},
    ],
    ids=["reasoning_parser", "structured_outputs_config", "model_default"],
)
def test_a_different_chat_parser_than_the_engines_reasoner_is_refused(
    fake_vllm, engine_reasoner
):
    # vLLM has one structured-output reasoner per engine; it would not match
    # chat's, wherever the engine's came from.
    if "model_default" in engine_reasoner:
        fake_vllm["engine_reasoner"] = engine_reasoner.pop("model_default")
    with pytest.raises(
        ValueError,
        match="chat_reasoning_parser='deepseek_r1' differs from reasoning_parser='qwen3'",
    ):
        start_worker(chat_reasoning_parser="deepseek_r1", **engine_reasoner)


def test_a_chat_parser_matching_a_configured_reasoner_is_accepted(fake_vllm):
    worker = start_worker(
        chat_reasoning_parser="qwen3",
        structured_outputs_config={"reasoning_parser": "qwen3"},
    )

    assert fake_vllm["reasoner_at_engine_start"] == "qwen3"
    assert worker._chat_only_reasoner is False


def test_an_engine_that_already_has_a_reasoner_keeps_it_and_generate_is_unchanged(fake_vllm):
    # gpt-oss gets openai_gptoss without asking (models/config.py:423), so
    # /generate already waits for reasoning there; don't change that.
    fake_vllm["engine_reasoner"] = "openai_gptoss"
    worker = start_worker(chat_reasoning_parser="openai_gptoss")

    generate(worker, [1, 2])

    assert fake_vllm["reasoner_at_engine_start"] == "openai_gptoss"
    assert worker.llm.calls[0][2]["reasoning_ended"] is None


def test_a_tables_reasoner_the_tokenizer_cannot_run_is_left_out(fake_vllm, monkeypatch, caplog):
    # vLLM builds the structured-output reasoner on the first request with a
    # grammar, so a missing think token would fail that request, not engine start.
    fake_vllm["architecture"] = "Qwen3ForCausalLM"
    fake_vllm["tokenizer"] = NO_THINK_TOKENS
    with caplog.at_level("WARNING"):
        worker = start_worker()

    assert fake_vllm["reasoner_at_engine_start"] == ""
    assert worker._chat_only_reasoner is False
    assert chat_parsers(monkeypatch, worker) == (None, "hermes")
    assert any("qwen3" in r.getMessage() for r in caplog.records if r.levelname == "WARNING")


def test_a_tables_reasoner_the_tokenizer_can_run_is_kept(fake_vllm, monkeypatch):
    fake_vllm["architecture"] = "Qwen3ForCausalLM"
    fake_vllm["tokenizer"] = "tokenizer"
    worker = start_worker()

    assert fake_vllm["reasoner_at_engine_start"] == "qwen3"
    assert chat_parsers(monkeypatch, worker) == ("qwen3", "hermes")


@pytest.mark.parametrize("tokenizer", [THINK_IDS_NONE, REWORDED_ERROR])
def test_a_tables_reasoner_is_left_out_however_the_parser_reports_missing_tokens(
    fake_vllm, monkeypatch, caplog, tokenizer
):
    fake_vllm["architecture"] = "Qwen3ForCausalLM"
    fake_vllm["tokenizer"] = tokenizer
    with caplog.at_level("WARNING"):
        worker = start_worker()

    assert fake_vllm["reasoner_at_engine_start"] == ""
    assert chat_parsers(monkeypatch, worker) == (None, "hermes")
    assert any("qwen3" in r.getMessage() for r in caplog.records if r.levelname == "WARNING")


def test_a_dropped_job_reasoning_parser_leaves_chat_without_one_either(fake_vllm, monkeypatch):
    # The job's reasoning_parser is dropped when the tokenizer lacks its think
    # tokens; the table's reasoner for chat needs the same tokens.
    fake_vllm["architecture"] = "Qwen3ForCausalLM"
    fake_vllm["tokenizer"] = NO_THINK_TOKENS
    worker = start_worker(reasoning_parser="deepseek_r1")

    assert "reasoning_parser" not in fake_vllm["engine_kwargs"]
    assert worker._reasoning_parser is None
    assert fake_vllm["reasoner_at_engine_start"] == ""
    assert chat_parsers(monkeypatch, worker) == (None, "hermes")


def test_an_explicit_chat_reasoning_parser_the_tokenizer_cannot_run_fails_start(fake_vllm):
    fake_vllm["tokenizer"] = NO_THINK_TOKENS
    with pytest.raises(ValueError, match="chat_reasoning_parser='qwen3'.*cannot run"):
        start_worker(chat_reasoning_parser="qwen3")
