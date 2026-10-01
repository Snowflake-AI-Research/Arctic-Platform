"""Opt-in dense prompt logprobs: same numbers, as tensors, only when asked.

These tests pin the two contracts the patch depends on:

1. vLLM hands us ``LogprobsTensors`` shaped ``[n - 1, k + 1]`` where row ``i``
   scores prompt position ``i + 1`` and column 0 is the observed token. Position
   0 has no score because nothing precedes it.
2. ``RequestOutput.prompt_logprobs`` is length ``n`` with ``None`` at index 0
   (``vllm.logprobs.create_prompt_logprobs`` seeds it). The dense arrays keep
   that alignment so existing callers index both by the same offset.

Nothing here imports vLLM: the patch keeps its vLLM imports inside
``ensure_dense_prompt_logprobs_patch`` so the pure logic stays testable on hosts
without it (same reason ``xgrammar_stop_mask`` does).
"""

from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from arctic_inference.vllm.dense_prompt_logprobs import (
    DENSE,
    FORMAT_KEY,
    RESULT_KEY,
    densify,
    stage_sampling_params,
    take_dense,
    wants_dense,
)

# prompt = [t0, t1, t2, t3]; only positions 1..3 can be scored.
PROMPT = [700, 701, 702, 703]
N, K = len(PROMPT), 2

# [n-1, k+1]. Column 0 is the observed token, columns 1..k are the top-k.
# Values are deliberately distinguishable so an off-by-one cannot pass.
TOKEN_IDS = torch.tensor(
    [
        [701, 910, 911],   # row 0 -> position 1, observed t1=701
        [702, 920, 921],   # row 1 -> position 2, observed t2=702
        [703, 930, 931],   # row 2 -> position 3, observed t3=703
    ],
    dtype=torch.int32,
)
LOGPROBS = torch.tensor(
    [
        [-1.5, -0.1, -0.2],
        [-2.5, -0.3, -0.4],
        [-3.5, -0.5, -0.6],
    ],
    dtype=torch.float32,
)
RANKS = torch.tensor([17, 1, 4], dtype=torch.int32)


class _Tensors(NamedTuple):
    """Stand-in for vllm.v1.outputs.LogprobsTensors (first three fields)."""

    logprob_token_ids: torch.Tensor
    logprobs: torch.Tensor
    selected_token_ranks: torch.Tensor


def _tensors():
    return _Tensors(TOKEN_IDS.clone(), LOGPROBS.clone(), RANKS.clone())


def _params(**kwargs):
    return SimpleNamespace(extra_args=kwargs or None)


# --------------------------------------------------------------------------
# wants_dense
# --------------------------------------------------------------------------


def test_wants_dense_only_when_the_request_asks_for_it():
    assert wants_dense(_params(**{FORMAT_KEY: DENSE})) is True
    assert wants_dense(_params()) is False
    assert wants_dense(SimpleNamespace(extra_args={})) is False
    assert wants_dense(SimpleNamespace()) is False
    assert wants_dense(None) is False


def test_wants_dense_refuses_a_value_it_cannot_honour():
    with pytest.raises(ValueError, match=FORMAT_KEY):
        wants_dense(_params(**{FORMAT_KEY: "sparse"}))


# --------------------------------------------------------------------------
# stage_sampling_params: what the DSS worker does before SamplingParams(**...)
# --------------------------------------------------------------------------


def test_stage_moves_the_key_into_extra_args_so_samplingparams_accepts_it():
    params = {"max_tokens": 1, "prompt_logprobs": K, FORMAT_KEY: DENSE}

    assert stage_sampling_params(params) is True
    assert FORMAT_KEY not in params, "SamplingParams would reject the unknown key"
    assert params["extra_args"] == {FORMAT_KEY: DENSE}
    assert params["max_tokens"] == 1 and params["prompt_logprobs"] == K


def test_stage_preserves_extra_args_other_passengers_already_put_there():
    params = {FORMAT_KEY: DENSE, "extra_args": {"dss_stop_token_sequences": [[1, 2]]}}

    stage_sampling_params(params)

    assert params["extra_args"] == {
        "dss_stop_token_sequences": [[1, 2]],
        FORMAT_KEY: DENSE,
    }


def test_stage_leaves_requests_that_did_not_ask_completely_untouched():
    params = {"max_tokens": 1, "prompt_logprobs": K}

    assert stage_sampling_params(params) is False
    assert params == {"max_tokens": 1, "prompt_logprobs": K}, "byte-for-byte unchanged"


def test_stage_refuses_a_value_it_cannot_honour():
    with pytest.raises(ValueError, match=FORMAT_KEY):
        stage_sampling_params({FORMAT_KEY: "sparse"})


# --------------------------------------------------------------------------
# take_dense: how the worker tells the two result shapes apart
# --------------------------------------------------------------------------


def test_take_dense_recognises_what_the_patch_produced():
    dense = densify(_tensors(), K)

    assert take_dense(dense) is dense


def test_take_dense_passes_on_every_stock_vllm_shape():
    assert take_dense([None, {701: object()}]) is None   # the list path
    assert take_dense(None) is None                      # prompt_logprobs disabled
    assert take_dense([]) is None
    assert take_dense({"logprobs": 1}) is None           # a dict, but not ours


# --------------------------------------------------------------------------
# densify: shape, dtype, alignment
# --------------------------------------------------------------------------


def test_densify_pads_to_prompt_length_keeping_the_vllm_alignment():
    dense = densify(_tensors(), K)

    # Length n, not n-1: index i lines up with prompt_token_ids[i], which is
    # what RequestOutput.prompt_logprobs does today.
    assert dense["logprobs"].shape == (N,)
    assert dense["topk_ids"].shape == (N, K)
    assert dense["topk_logprobs"].shape == (N, K)

    assert dense["logprobs"].dtype == torch.float32
    assert dense["topk_ids"].dtype == torch.int32
    assert dense["topk_logprobs"].dtype == torch.float32


def test_densify_marks_position_zero_as_unscored():
    dense = densify(_tensors(), K)

    assert torch.isnan(dense["logprobs"][0])
    assert torch.equal(dense["topk_ids"][0], torch.full((K,), -1, dtype=torch.int32))
    assert torch.all(torch.isneginf(dense["topk_logprobs"][0]))


def test_densify_row_i_of_the_tensor_lands_on_prompt_position_i_plus_one():
    dense = densify(_tensors(), K)

    # Column 0 of row i is the observed token at position i+1.
    torch.testing.assert_close(
        dense["logprobs"][1:], torch.tensor([-1.5, -2.5, -3.5])
    )
    # The observed column is dropped from topk_*; only the top-k survive.
    assert torch.equal(
        dense["topk_ids"][1:],
        torch.tensor([[910, 911], [920, 921], [930, 931]], dtype=torch.int32),
    )
    torch.testing.assert_close(
        dense["topk_logprobs"][1:],
        torch.tensor([[-0.1, -0.2], [-0.3, -0.4], [-0.5, -0.6]]),
    )


def test_densify_observed_logprob_matches_the_token_actually_in_the_prompt():
    dense = densify(_tensors(), K)

    for position in range(1, N):
        observed_token = int(TOKEN_IDS[position - 1, 0])
        assert observed_token == PROMPT[position], "column 0 must be the observed token"
        assert dense["logprobs"][position] == LOGPROBS[position - 1, 0]


# --------------------------------------------------------------------------
# densify: equivalence with the dict path vLLM builds today
# --------------------------------------------------------------------------


def _dict_path():
    """What append_logprobs_for_next_position produces, as a plain dict."""
    out = [None]
    for row in range(N - 1):
        ids = TOKEN_IDS[row].tolist()
        values = LOGPROBS[row].tolist()
        ranks = [int(RANKS[row])] + list(range(1, K + 1))
        out.append({t: {"logprob": v, "rank": r} for t, v, r in zip(ids, values, ranks)})
    return out


def test_densify_carries_the_same_numbers_as_the_dict_path():
    dense = densify(_tensors(), K)
    reference = _dict_path()

    assert reference[0] is None and torch.isnan(dense["logprobs"][0])
    for position in range(1, N):
        entries = reference[position]
        observed = PROMPT[position]
        assert dense["logprobs"][position] == pytest.approx(entries[observed]["logprob"])
        for slot in range(K):
            token = int(dense["topk_ids"][position, slot])
            assert dense["topk_logprobs"][position, slot] == pytest.approx(
                entries[token]["logprob"]
            )


# --------------------------------------------------------------------------
# densify: robustness against vLLM drift
# --------------------------------------------------------------------------


def test_densify_reads_fields_by_name_so_added_namedtuple_fields_are_harmless():
    """LogprobsTensors grew a field in v0.11.1 and again in v0.29.0."""

    class _Wider(NamedTuple):
        logprob_token_ids: torch.Tensor
        logprobs: torch.Tensor
        selected_token_ranks: torch.Tensor
        cu_num_generated_tokens: object = None
        cu_num_generated_tokens_tensor: object = None

    wider = _Wider(TOKEN_IDS.clone(), LOGPROBS.clone(), RANKS.clone())
    assert torch.equal(densify(wider, K)["topk_ids"], densify(_tensors(), K)["topk_ids"])


def test_densify_refuses_a_width_that_does_not_match_the_requested_topk():
    with pytest.raises(ValueError, match="k"):
        densify(_tensors(), K + 1)


def test_densify_accepts_topk_zero():
    """prompt_logprobs=0 is legal: the observed token only, no top-k."""
    tensors = _Tensors(
        TOKEN_IDS[:, :1].clone(), LOGPROBS[:, :1].clone(), RANKS.clone()
    )
    dense = densify(tensors, 0)

    assert dense["logprobs"].shape == (N,)
    assert dense["topk_ids"].shape == (N, 0)
    torch.testing.assert_close(dense["logprobs"][1:], torch.tensor([-1.5, -2.5, -3.5]))


def test_densify_does_not_alias_the_engine_tensors():
    """The engine reuses in_progress_prompt_logprobs_cpu buffers across requests."""
    tensors = _tensors()
    dense = densify(tensors, K)
    tensors.logprobs[0, 0] = -999.0

    assert dense["logprobs"][1] == pytest.approx(-1.5)


# --------------------------------------------------------------------------
# The patch itself. vLLM is not installed here, so stand in for the two
# attributes ensure_dense_prompt_logprobs_patch touches: the version string and
# LogprobsProcessor.
# --------------------------------------------------------------------------


class _StubProcessor:
    """Enough of vllm.v1.engine.logprobs.LogprobsProcessor to patch."""

    def __init__(self, num_prompt_logprobs):
        self.num_prompt_logprobs = num_prompt_logprobs
        self.prompt_logprobs = [None]
        self.updates = []

    @classmethod
    def from_new_request(cls, tokenizer, request):
        return cls(request.sampling_params.prompt_logprobs)

    def _update_prompt_logprobs(self, prompt_logprobs_tensors):
        # Stands in for vLLM's pythonization.
        self.updates.append(prompt_logprobs_tensors)
        self.prompt_logprobs = [None] + [{} for _ in range(N - 1)]


@pytest.fixture
def patched(monkeypatch):
    """Install stub vLLM modules, apply the patch, hand back the stub class."""
    import sys
    from types import ModuleType

    vllm = ModuleType("vllm")
    vllm.__version__ = "0.30.0"
    logprobs_mod = ModuleType("vllm.v1.engine.logprobs")
    logprobs_mod.LogprobsProcessor = _StubProcessor
    for name, module in (
        ("vllm", vllm),
        ("vllm.v1", ModuleType("vllm.v1")),
        ("vllm.v1.engine", ModuleType("vllm.v1.engine")),
        ("vllm.v1.engine.logprobs", logprobs_mod),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    original = (_StubProcessor.from_new_request, _StubProcessor._update_prompt_logprobs)
    from arctic_inference.vllm.dense_prompt_logprobs import (
        ensure_dense_prompt_logprobs_patch,
    )

    ensure_dense_prompt_logprobs_patch()
    yield _StubProcessor
    _StubProcessor.from_new_request, _StubProcessor._update_prompt_logprobs = original


def _request(dense: bool):
    extra_args = {FORMAT_KEY: DENSE} if dense else None
    return SimpleNamespace(
        sampling_params=SimpleNamespace(prompt_logprobs=K, extra_args=extra_args)
    )


def test_patched_from_new_request_still_behaves_like_a_classmethod(patched):
    processor = patched.from_new_request(None, _request(dense=True))

    assert isinstance(processor, _StubProcessor)
    assert processor.num_prompt_logprobs == K


def test_patch_sends_only_opted_in_requests_down_the_dense_path(patched):
    dense_proc = patched.from_new_request(None, _request(dense=True))
    plain_proc = patched.from_new_request(None, _request(dense=False))

    dense_proc._update_prompt_logprobs(_tensors())
    plain_proc._update_prompt_logprobs(_tensors())

    # Opted in: tensors, and vLLM's pythonization never ran.
    assert set(dense_proc.prompt_logprobs) == {"logprobs", "topk_ids", "topk_logprobs"}
    assert dense_proc.updates == []

    # Not opted in: stock behaviour, untouched.
    assert plain_proc.prompt_logprobs[0] is None
    assert len(plain_proc.updates) == 1


def test_patch_refuses_a_second_chunk_instead_of_overwriting(patched):
    """vLLM reassembles chunked prefill today; fail loudly if that changes."""
    processor = patched.from_new_request(None, _request(dense=True))
    processor._update_prompt_logprobs(_tensors())

    with pytest.raises(RuntimeError, match="second tensor chunk"):
        processor._update_prompt_logprobs(_tensors())


def test_patch_is_idempotent(patched):
    from arctic_inference.vllm.dense_prompt_logprobs import (
        ensure_dense_prompt_logprobs_patch,
    )

    before = patched._update_prompt_logprobs
    ensure_dense_prompt_logprobs_patch()

    assert patched._update_prompt_logprobs is before


# --------------------------------------------------------------------------
# densify is also what the CPU-only DummyWorker calls to build its dense
# answer, so the fake engine cannot drift from production. That test lives
# here rather than against DummyWorker itself: importing it drags in ray and
# vllm, and a stub chain that deep is more fragile than what it protects.
# --------------------------------------------------------------------------


def test_densify_output_satisfies_take_dense():
    """The round trip every caller relies on: densify -> take_dense -> arrays."""
    assert take_dense(densify(_tensors(), K)) is not None
