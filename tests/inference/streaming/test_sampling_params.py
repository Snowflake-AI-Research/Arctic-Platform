"""Stream sampling parameters: request validation, engine arguments, error codes."""

import asyncio
import sys
import types

import pytest

from cpu_support import VLLMValidationError, load_library

load_library()
from arctic_platform.inference.server.streaming import (
    ClientStream,
    StreamError,
    StreamingWorkerMixin,
    STREAM_CAPABILITIES,
    StreamLimits,
    classify_engine_error,
    delta_logprobs,
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
    "structured_outputs", [{"json": SCHEMA}, {"json": {}}, {"json_object": True}]
)
def test_structured_output_shapes(structured_outputs):
    _, params = validate_request("prompt", {"structured_outputs": structured_outputs})
    assert params["structured_outputs"] == structured_outputs


def test_structured_output_schema_size_boundary():
    # {"description":"..."} is 18 bytes of framing around the padding.
    at_limit = {"description": "x" * (64 * 1024 - 18)}
    validate_request("prompt", {"structured_outputs": {"json": at_limit}})
    over = {"description": "x" * (64 * 1024 - 17)}
    with pytest.raises(ValueError, match="65536"):
        validate_request("prompt", {"structured_outputs": {"json": over}})


@pytest.mark.parametrize(
    "structured_outputs",
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
def test_structured_output_rejection(structured_outputs):
    with pytest.raises(ValueError, match="structured_outputs"):
        validate_request("prompt", {"structured_outputs": structured_outputs})


def test_structured_output_becomes_the_vllm_type(engine_params):
    kwargs = engine_params({"structured_outputs": {"json": SCHEMA}})
    assert kwargs["structured_outputs"].kwargs == {"json": SCHEMA}
    kwargs = engine_params({"structured_outputs": {"json_object": True}})
    assert kwargs["structured_outputs"].kwargs == {"json_object": True}
    assert "structured_outputs" not in engine_params({})


def test_the_singular_structured_output_key_is_unknown():
    # The stream key is vLLM's name for it, structured_outputs.
    with pytest.raises(ValueError, match="Unsupported streaming parameters"):
        validate_request("prompt", {"structured_output": {"json_object": True}})


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


def test_omitted_max_tokens_stays_unset_for_the_worker():
    # The worker applies the default and lets it shrink to the model context.
    _, params = validate_request("prompt", {})
    assert "max_tokens" not in params
    _, params = validate_request("prompt", params)
    assert "max_tokens" not in params
    assert validate_request("prompt", {"max_tokens": 4096})[1]["max_tokens"] == 4096


def test_thinking_budget_without_a_reasoning_parser_is_typed():
    error = VLLMValidationError(
        "thinking_token_budget is set but reasoning_config is not configured. "
        "Please set --reasoning-parser and/or --reasoning-config to use "
        "thinking_token_budget."
    )
    assert classify_engine_error(error) == ("invalid_sampling_params", None)


@pytest.mark.parametrize("logprobs", [0, 20])
def test_logprobs_accepts_zero_to_twenty(logprobs, engine_params):
    assert engine_params({"logprobs": logprobs})["logprobs"] == logprobs


@pytest.mark.parametrize("logprobs", [-1, 21, True, 2.0, "2"])
def test_logprobs_rejection(logprobs):
    with pytest.raises(ValueError, match="logprobs"):
        validate_request("prompt", {"logprobs": logprobs})


def _logprob(logprob, rank, token):
    return types.SimpleNamespace(logprob=logprob, rank=rank, decoded_token=token)


def test_delta_logprobs_keep_the_chosen_token_out_of_a_top_k_it_missed():
    position = {
        7: _logprob(-6.0, 3, " Nice"),  # chosen, ranked third
        5: _logprob(-0.01, 1, " Paris"),
        9: _logprob(-5.2, 2, " Lyon"),
    }
    [entry] = delta_logprobs([7], [position], 2)
    assert entry == {
        "token_id": 7,
        "token": " Nice",
        "logprob": -6.0,
        "top": [
            {"token_id": 5, "token": " Paris", "logprob": -0.01},
            {"token_id": 9, "token": " Lyon", "logprob": -5.2},
        ],
    }


def test_delta_logprobs_encode_negative_infinity_and_zero_alternatives():
    [entry] = delta_logprobs([3], [{3: _logprob(float("-inf"), 4, None)}], 0)
    assert entry == {"token_id": 3, "token": "", "logprob": -9999.0, "top": []}


@pytest.mark.parametrize(
    "token_ids,positions",
    [
        ([1, 2], [{1: _logprob(-1.0, 1, "a")}]),
        ([1], None),
        ([1], [{2: _logprob(-1.0, 1, "a")}]),
    ],
)
def test_delta_logprobs_reject_misaligned_engine_output(token_ids, positions):
    with pytest.raises(StreamError, match="invalid_engine_output"):
        delta_logprobs(token_ids, positions, 1)


def _entry(token_id, top=()):
    return {"token_id": token_id, "token": "t", "logprob": -1.0, "top": list(top)}


def _alternative(token_id):
    return {"token_id": token_id, "token": "t", "logprob": -2.0}


@pytest.mark.parametrize(
    "logprobs,valid",
    [
        ([_entry(1, [_alternative(1), _alternative(4)]), _entry(2)], True),
        ([_entry(1)], False),  # one entry for two token IDs
        ([_entry(1), _entry(2), _entry(3)], False),
        ([_entry(2), _entry(1)], False),
        ([_entry(1, [_alternative(n) for n in range(21)]), _entry(2)], False),
        # More alternatives than the 2 requested.
        ([_entry(1, [_alternative(n) for n in range(3)]), _entry(2)], False),
        ([{**_entry(1), "logprob": float("nan")}, _entry(2)], False),
        ([{**_entry(1), "token": None}, _entry(2)], False),
        ([{**_entry(1), "token_id": True}, _entry(2)], False),
        ([{k: v for k, v in _entry(1).items() if k != "top"}, _entry(2)], False),
        ([_entry(1, [{"token_id": 4, "token": "t"}]), _entry(2)], False),
        ("logprobs", False),
    ],
)
def test_client_stream_validates_delta_logprobs(logprobs, valid, requested=2):
    async def check():
        stream = ClientStream(
            types.SimpleNamespace(),
            "request",
            types.SimpleNamespace(),
            {"n": 1} if requested is None else {"n": 1, "logprobs": requested},
            StreamLimits(),
        )
        event = {
            "type": "delta",
            "choice_index": 0,
            "text": "ab",
            "token_ids": [1, 2],
            "logprobs": logprobs,
            "sequence": 0,
            "version": 1,
        }
        try:
            if valid:
                assert (await stream._accept_event(event))["logprobs"] == logprobs
            else:
                with pytest.raises(StreamError, match="invalid_choice_event"):
                    await stream._accept_event(event)
        finally:
            stream.watchdog.cancel()

    asyncio.run(check())


def test_capability_is_advertised():
    # DSS enables these fields only when the installed version lists it.
    assert "sampling_params" in STREAM_CAPABILITIES


def test_streams_without_structured_output_never_import_it(engine_params):
    # A vLLM build without StructuredOutputsParams must still serve plain streams.
    del sys.modules["vllm.sampling_params"].StructuredOutputsParams
    assert engine_params({"max_tokens": 4})["max_tokens"] == 4


def test_client_stream_rejects_logprobs_nobody_requested():
    test_client_stream_validates_delta_logprobs([_entry(1), _entry(2)], False, requested=None)


def test_logprobs_above_the_engine_maximum_are_typed():
    error = VLLMValidationError(
        "Requested sample logprobs of 21, which is greater than max allowed: 20",
        parameter="logprobs",
        value=21,
    )
    assert classify_engine_error(error) == ("invalid_sampling_params", None)


def _logprob_delta(token_id):
    # A chosen token plus 20 alternatives, with full-precision float logprobs.
    position = {
        rank * 100_000 + token_id: _logprob(-rank / 7, rank, f" word{rank}")
        for rank in range(2, 21)
    }
    position[token_id] = _logprob(-0.123456789, 1, f" word{token_id}")
    return {
        "type": "delta",
        "choice_index": 0,
        "text": f" word{token_id}",
        "token_ids": [token_id],
        "logprobs": delta_logprobs([token_id], [position], 20),
    }


def _fill(buffer, tokens):
    for token_id in range(tokens):
        buffer.put(_logprob_delta(10_000 + token_id))


def test_logprob_stream_behind_a_slow_reader_overflows_the_default_buffer():
    # Each token with logprobs=20 serializes to over 1 KiB, so the default
    # 1 MiB buffer holds well under a thousand undelivered tokens.
    from arctic_platform.inference.server.streaming import EventBuffer, event_size

    assert event_size(_logprob_delta(10_000)) > 1024
    with pytest.raises(StreamError, match="buffer_overflow"):
        _fill(EventBuffer(StreamLimits()), 3000)


def test_logprob_stream_behind_a_slow_reader_fits_a_raised_buffer():
    from arctic_platform.inference.server.streaming import EventBuffer

    buffer = EventBuffer(StreamLimits(max_buffer_bytes=16 * 1024 * 1024))
    _fill(buffer, 3000)
    events = buffer.drain()
    assert [
        entry["token_id"] for event in events for entry in event["logprobs"]
    ] == list(range(10_000, 13_000))
    assert all(len(entry["top"]) == 20 for e in events for entry in e["logprobs"])
