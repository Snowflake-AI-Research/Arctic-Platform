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
