from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from arctic_platform.inference.vllm.spec_decode_grammar import (
    attach_spec_token_ids,
    mask_unvalidated_drafts_v2,
)


@dataclass
class _Batch:
    req_ids: list[str]
    num_draft_tokens_per_req: np.ndarray
    cu_num_logits_np: np.ndarray
    logits_indices: torch.Tensor
    input_ids: torch.Tensor
    num_draft_tokens: int


@dataclass
class _Grammar:
    spec_token_ids: dict = field(default_factory=dict)


def _batch():
    # Sparse logits indices include a bonus row before each request's drafts.
    return _Batch(
        ['a', 'b', 'c', 'd', 'e'], np.array([3, 3, 0, 3, 2]),
        np.array([0, 4, 8, 9, 13, 16]),
        torch.arange(16) * 2, torch.arange(32), 11,
    )


def test_mask_preserves_checked_drafts_and_original_model_inputs():
    batch = _batch()
    grammar = _Grammar({'a': [10, 11, 12], 'b': [20, -1, -1], 'd': [-1, -1, -1]})
    masked = mask_unvalidated_drafts_v2(grammar, batch)
    expected = batch.input_ids.clone()
    expected[[12, 14, 20, 22, 24]] = -1
    assert masked is not batch
    assert torch.equal(masked.input_ids, expected)
    assert torch.equal(batch.input_ids, torch.arange(32))
    assert masked.logits_indices is batch.logits_indices


def test_mask_returns_original_batch_when_every_draft_is_checked():
    batch = _batch()
    assert mask_unvalidated_drafts_v2(_Grammar({'a': [1, 2, 3]}), batch) is batch


def test_mask_rejects_entire_tail_after_short_placeholder_list():
    batch = _batch()
    masked = mask_unvalidated_drafts_v2(_Grammar({'b': [-1]}), batch)
    assert masked.input_ids[[10, 12, 14]].tolist() == [-1, -1, -1]


def test_attach_snapshots_scheduled_drafts():
    scheduled = {'a': [-1, 4], 'b': [7]}
    grammar = SimpleNamespace(structured_output_request_ids=['a', 'c'])
    attach_spec_token_ids(grammar, SimpleNamespace(scheduled_spec_decode_tokens=scheduled))
    scheduled['a'].append(9)
    assert grammar.spec_token_ids == {'a': [-1, 4]}
    assert attach_spec_token_ids(None, None) is None


def test_grammar_output_keeps_spec_token_ids_across_pickle():
    import pickle
    from vllm.v1.core.sched.output import GrammarOutput

    grammar = GrammarOutput(['a'], np.zeros((1, 1), dtype=np.int32))
    attach_spec_token_ids(grammar, SimpleNamespace(scheduled_spec_decode_tokens={'a': [-1, 5]}))
    assert pickle.loads(pickle.dumps(grammar)).spec_token_ids == {'a': [-1, 5]}


def test_installer_masks_only_the_sampling_batch_and_is_idempotent(monkeypatch):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from arctic_platform.inference.vllm import spec_decode_grammar as patch

    seen = []
    monkeypatch.setattr(GPUModelRunner, 'sample', lambda self, hidden, batch, grammar: seen.append(batch))
    monkeypatch.setattr(patch, '_APPLIED', False)
    patch.ensure_spec_decode_grammar_fix()
    wrapper = GPUModelRunner.sample
    patch.ensure_spec_decode_grammar_fix()
    assert GPUModelRunner.sample is wrapper
    batch = _batch()
    wrapper(None, None, batch, _Grammar({'b': [-1]}))
    assert seen[0].input_ids[[10, 12, 14]].tolist() == [-1, -1, -1]
    assert torch.equal(batch.input_ids, torch.arange(32))
