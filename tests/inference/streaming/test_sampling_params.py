"""Stream sampling parameters: request validation, engine arguments, error codes."""

import sys
import types

import pytest

from cpu_support import VLLMValidationError, load_library

load_library()
from arctic_platform.inference.server.streaming import (
    StreamingWorkerMixin,
    classify_engine_error,
    validate_request,
)


@pytest.fixture
def engine_params(monkeypatch):
    """Build params with stand-ins for the vLLM classes and return their kwargs."""

    class SamplingParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class StructuredOutputsParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    module = types.ModuleType("vllm.sampling_params")
    module.RequestOutputKind = types.SimpleNamespace(DELTA="delta")
    module.StructuredOutputsParams = StructuredOutputsParams
    monkeypatch.setitem(sys.modules, "vllm.sampling_params", module)
    monkeypatch.setattr(
        sys.modules["vllm"], "SamplingParams", SamplingParams, raising=False
    )

    def build(sampling_params):
        _, params = validate_request("prompt", sampling_params)
        return StreamingWorkerMixin()._stream_sampling_params(params).kwargs

    return build


def test_logit_bias_string_keys_become_token_ids():
    _, params = validate_request("prompt", {"logit_bias": {"50256": -100, 7: 2.5}})
    assert params["logit_bias"] == {50256: -100.0, 7: 2.5}


def test_logit_bias_accepts_the_boundaries():
    bias = {token: (100 if token % 2 else -100) for token in range(299)}
    bias[2**31 - 1] = 0
    _, params = validate_request("prompt", {"logit_bias": bias})
    assert len(params["logit_bias"]) == 300


@pytest.mark.parametrize(
    "logit_bias",
    [
        {token: 1 for token in range(301)},
        {1: 100.5},
        {1: -100.5},
        {1: True},
        {1: float("nan")},
        {1: float("inf")},
        {1: "1"},
        {-1: 1},
        {"-1": 1},
        {2**31: 1},
        {"1.0": 1},
        {" 1": 1},
        {"": 1},
        {"1" * 11: 1},
        {True: 1},
        {1: 1, "1": 2},
        [[1, 1]],
    ],
)
def test_logit_bias_rejection(logit_bias):
    with pytest.raises(ValueError, match="logit_bias"):
        validate_request("prompt", {"logit_bias": logit_bias})


def test_logit_bias_reaches_the_engine(engine_params):
    assert engine_params({"logit_bias": {"5": 1}})["logit_bias"] == {5: 1.0}


@pytest.mark.parametrize(
    "error",
    [
        VLLMValidationError(
            "token_id(s) [151936] in logit_bias contain out-of-vocab token ids. "
            "Vocabulary size: 151936",
            parameter="logit_bias",
            value=[151936],
        ),
        VLLMValidationError(
            "The min_p and logit_bias sampling parameters are not yet supported "
            "with speculative decoding."
        ),
    ],
)
def test_logit_bias_engine_rejection_is_typed(error):
    assert classify_engine_error(error) == ("invalid_sampling_params", None)


SCHEMA = {"type": "object", "properties": {"city": {"type": "string"}}}


@pytest.mark.parametrize(
    "structured_output", [{"json": SCHEMA}, {"json": {}}, {"json_object": True}]
)
def test_structured_output_shapes(structured_output):
    _, params = validate_request("prompt", {"structured_output": structured_output})
    assert params["structured_output"] == structured_output


def test_structured_output_schema_size_boundary():
    # {"description":"..."} is 18 bytes of framing around the padding.
    at_limit = {"description": "x" * (64 * 1024 - 18)}
    validate_request("prompt", {"structured_output": {"json": at_limit}})
    over = {"description": "x" * (64 * 1024 - 17)}
    with pytest.raises(ValueError, match="65536"):
        validate_request("prompt", {"structured_output": {"json": over}})


@pytest.mark.parametrize(
    "structured_output",
    [
        {},
        {"json": SCHEMA, "json_object": True},
        {"json": '{"type": "object"}'},
        {"json": None},
        {"json_object": False},
        {"json_object": 1},
        {"regex": "a+"},
        {"json": {"maximum": float("nan")}},
        {"json": {"enum": {1, 2}}},
        "json_object",
        ["json_object"],
    ],
)
def test_structured_output_rejection(structured_output):
    with pytest.raises(ValueError, match="structured_output"):
        validate_request("prompt", {"structured_output": structured_output})


def test_structured_output_becomes_the_vllm_type(engine_params):
    kwargs = engine_params({"structured_output": {"json": SCHEMA}})
    assert "structured_output" not in kwargs
    assert kwargs["structured_outputs"].kwargs == {"json": SCHEMA}
    kwargs = engine_params({"structured_output": {"json_object": True}})
    assert kwargs["structured_outputs"].kwargs == {"json_object": True}
    assert "structured_outputs" not in engine_params({})


@pytest.mark.parametrize(
    "message",
    [
        "Failed to transform json schema into a grammar: unsupported keyword",
        "The provided JSON schema contains features not supported by xgrammar.",
        "Grammar error: unsatisfiable schema",
        "Invalid grammar specification: bad key",
        "Failed to transform json schema into a regex: unsupported",
        "Regex uses unsupported feature for structured outputs: lookaround. "
        "Only basic matching constructs are supported",
    ],
)
def test_structured_output_engine_rejection_is_typed(message):
    assert classify_engine_error(VLLMValidationError(message)) == (
        "invalid_structured_output",
        None,
    )


def test_unrelated_validation_errors_stay_engine_errors():
    assert classify_engine_error(VLLMValidationError("something else")) == (
        "engine_error",
        None,
    )
    assert classify_engine_error(
        RuntimeError("Grammar error: not a validation error")
    ) == ("engine_error", None)


@pytest.mark.parametrize("budget", [1, 64])
def test_thinking_token_budget_accepts_one_to_max_tokens(budget, engine_params):
    sampling_params = {"max_tokens": 64, "thinking_token_budget": budget}
    assert engine_params(sampling_params)["thinking_token_budget"] == budget


@pytest.mark.parametrize("budget", [0, -1, 65, 1.0, True, "8"])
def test_thinking_token_budget_rejection(budget):
    with pytest.raises(ValueError, match="thinking_token_budget"):
        validate_request(
            "prompt", {"max_tokens": 64, "thinking_token_budget": budget}
        )


def test_thinking_token_budget_is_bounded_by_the_default_max_tokens():
    validate_request("prompt", {"thinking_token_budget": 4096})
    with pytest.raises(ValueError, match="thinking_token_budget"):
        validate_request("prompt", {"thinking_token_budget": 4097})


def test_thinking_budget_without_a_reasoning_parser_is_typed():
    error = VLLMValidationError(
        "thinking_token_budget is set but reasoning_config is not configured. "
        "Please set --reasoning-parser and/or --reasoning-config to use "
        "thinking_token_budget."
    )
    assert classify_engine_error(error) == ("invalid_sampling_params", None)
