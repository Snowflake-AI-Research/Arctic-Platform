from dataclasses import dataclass
from dataclasses import field
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.inference.vllm.spec_decode_grammar import attach_spec_token_ids
from arctic_platform.inference.vllm.spec_decode_grammar import clear_sample_metadata
from arctic_platform.inference.vllm.spec_decode_grammar import metadata_for_sample
from arctic_platform.inference.vllm.spec_decode_grammar import reject_unvalidated_drafts
from arctic_platform.inference.vllm.spec_decode_grammar import stage_sample_metadata


@dataclass
class _Metadata:
    draft_token_ids: torch.Tensor
    num_draft_tokens: list


@dataclass
class _Grammar:
    spec_token_ids: dict = field(default_factory=dict)


def test_reject_unvalidated_drafts_keeps_checked_drafts_only():
    # Requests a, b, c (no drafts), d, e. e is not in the grammar output.
    draft_ids = torch.tensor([10, 11, 12, 20, 21, 22, 30, 31, 32, 40, 41], dtype=torch.int32)
    metadata = _Metadata(draft_ids.clone(), [3, 3, 0, 3, 2])
    grammar = _Grammar(
        {
            "a": [10, 11, 12],
            "b": [20, -1, -1],
            "d": [-1, -1, -1],
        }
    )

    rejected = reject_unvalidated_drafts(grammar, ["a", "b", "c", "d", "e"], metadata)

    assert rejected is not metadata
    assert rejected.draft_token_ids.tolist() == [10, 11, 12, 20, -1, -1, -1, -1, -1, 40, 41]
    assert metadata.draft_token_ids.tolist() == draft_ids.tolist()


def test_reject_unvalidated_drafts_returns_same_metadata_when_rows_match():
    metadata = _Metadata(torch.tensor([10, 11, 12], dtype=torch.int32), [3])

    rejected = reject_unvalidated_drafts(_Grammar({"a": [10, 11, 12]}), ["a"], metadata)

    assert rejected is metadata


def test_reject_from_first_placeholder_covers_the_runners_longer_draft_span():
    metadata = _Metadata(torch.tensor([8, 9, 10], dtype=torch.int32), [3])

    rejected = reject_unvalidated_drafts(_Grammar({"a": [-1]}), ["a"], metadata)

    assert rejected.draft_token_ids.tolist() == [-1, -1, -1]


def test_attach_spec_token_ids_snapshots_the_scheduled_drafts():
    scheduled = {"a": [-1, 4], "b": [7]}
    grammar = SimpleNamespace(structured_output_request_ids=["a", "c"])
    scheduler_output = SimpleNamespace(scheduled_spec_decode_tokens=scheduled)

    attached = attach_spec_token_ids(grammar, scheduler_output)
    scheduled["a"].append(9)

    assert attached is grammar
    assert grammar.spec_token_ids == {"a": [-1, 4]}
    assert attach_spec_token_ids(None, scheduler_output) is None


def test_sample_override_does_not_replace_the_metadata_used_for_drafting():
    draft_ids = torch.tensor([20, 21, 22], dtype=torch.int32)
    metadata = _Metadata(draft_ids.clone(), [3])
    runner = SimpleNamespace(
        execute_model_state=SimpleNamespace(spec_decode_metadata=metadata),
        input_batch=SimpleNamespace(req_ids=["b"]),
    )
    grammar = _Grammar({"b": [20, -1, -1]})

    stage_sample_metadata(runner, grammar)
    try:
        verified = metadata_for_sample(runner, metadata)
        assert verified.draft_token_ids.tolist() == [20, -1, -1]
        assert runner.execute_model_state.spec_decode_metadata is metadata
        assert metadata.draft_token_ids.tolist() == draft_ids.tolist()
    finally:
        clear_sample_metadata(runner)

    assert metadata_for_sample(runner, metadata) is metadata


def test_grammar_output_keeps_spec_token_ids_across_pickle():
    pickle = pytest.importorskip("pickle")
    np = pytest.importorskip("numpy")
    pytest.importorskip("vllm")
    from vllm.v1.core.sched.output import GrammarOutput

    grammar = GrammarOutput(["a"], np.zeros((1, 1), dtype=np.int32))
    scheduler_output = SimpleNamespace(scheduled_spec_decode_tokens={"a": [-1, 5]})
    attach_spec_token_ids(grammar, scheduler_output)

    restored = pickle.loads(pickle.dumps(grammar))

    assert restored.spec_token_ids == {"a": [-1, 5]}


def test_ensure_spec_decode_grammar_fix_is_idempotent():
    pytest.importorskip("vllm")
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    from arctic_platform.inference.vllm.spec_decode_grammar import ensure_spec_decode_grammar_fix

    ensure_spec_decode_grammar_fix()
    get_grammar_bitmask = Scheduler.get_grammar_bitmask
    sample_tokens = GPUModelRunner.sample_tokens
    sample = GPUModelRunner._sample

    ensure_spec_decode_grammar_fix()

    assert Scheduler.get_grammar_bitmask is get_grammar_bitmask
    assert GPUModelRunner.sample_tokens is sample_tokens
    assert GPUModelRunner._sample is sample
    assert get_grammar_bitmask._arctic_spec_grammar is True
    assert sample_tokens._arctic_spec_grammar is True
    assert sample._arctic_spec_grammar is True
