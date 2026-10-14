# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``logits_to_keep`` handling on the forward installed by ``enable_chunked_lm_head_logprobs``.

Hugging Face accepts two forms of ``logits_to_keep``: an ``int`` count of trailing positions, where ``0`` means
keep every position, and a 1-D tensor of the column indices to keep. Only the ``int`` form has a value that means
"nothing was asked for", so only the ``int`` form can be decided by a truth test.

These tests cover the delegated branch -- ``dss_compute_logprobs=False``, where the wrapper hands the call to the
model's own forward. Nothing downstream of that branch re-derives the selection, so whatever the branch decides
about ``logits_to_keep`` is what the caller gets: a dropped selector is a full-width result with no shape error
to signal it.

One row per call: the contract is per-column, and a second row only obscures which column an assertion is about.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

VOCAB_SIZE = 7
HIDDEN_SIZE = 4
NUM_EMBEDDINGS = 8
SEQ_LEN = 4

# Distinguishes "the wrapper passed ``logits_to_keep=0``" from "the wrapper did not pass it at all", which a
# recorded value alone cannot: the model's own default for the argument is ``0``.
NOT_PASSED = object()


class _RecordingCausalLM(torch.nn.Module):
    """A causal LM that slices by ``logits_to_keep`` as Hugging Face does and records what it was handed.

    ``slice(-logits_to_keep, None)`` is the ``int`` convention, and it covers ``0`` without a special case
    because ``-0`` is ``0``; a tensor indexes the column axis directly.
    """

    def __init__(self) -> None:
        super().__init__()

        class _Backbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.Embedding(NUM_EMBEDDINGS, HIDDEN_SIZE)

            def forward(self, input_ids=None, **_kwargs):
                return SimpleNamespace(last_hidden_state=self.emb(input_ids))

        self.model = _Backbone()
        self.lm_head = torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=True)
        self.received: list = []

    def forward(self, input_ids=None, **kwargs):
        selector = kwargs.get("logits_to_keep", NOT_PASSED)
        self.received.append(selector)
        hidden = self.model(input_ids=input_ids).last_hidden_state
        if selector is NOT_PASSED:
            columns = slice(None)
        elif isinstance(selector, int):
            columns = slice(-selector, None)
        else:
            columns = selector
        return SimpleNamespace(logits=self.lm_head(hidden[:, columns, :]))


def _patched_model() -> _RecordingCausalLM:
    from arctic_platform.model.implementations.gpu.lm_head import enable_chunked_lm_head_logprobs

    torch.manual_seed(17)
    model = _RecordingCausalLM()
    enable_chunked_lm_head_logprobs(model, token_chunk_size=2, vocab_chunk_size=3, fp32_lm_head=True)
    return model


def _input_ids() -> torch.Tensor:
    return torch.tensor([[1, 2, 3, 4]])


def test_delegated_forward_passes_a_multi_column_tensor_selector_through():
    """A tensor of column indices must reach the wrapped forward and select exactly those columns.

    A truth test on this value raises ``RuntimeError: Boolean value of Tensor is ambiguous`` before the wrapped
    forward is ever called, which closes the delegated path to every caller holding a tensor selector.
    """
    model = _patched_model()
    input_ids = _input_ids()
    kept = torch.tensor([1, 3])

    out = model(input_ids=input_ids, logits_to_keep=kept)

    assert len(model.received) == 1
    assert torch.equal(model.received[0], kept)
    expected = model.lm_head(model.model.emb(input_ids)[:, kept, :])
    assert tuple(out.logits.shape) == (1, 2, VOCAB_SIZE)
    torch.testing.assert_close(out.logits, expected)


def test_delegated_forward_passes_a_single_column_tensor_selector_through():
    """A one-element selector naming column ``0`` must select that column, not the whole sequence.

    This is the form a truth test answers rather than raising on: ``bool(tensor([0]))`` is the emptiness of the
    column index, not of the selection, so the selector is dropped and the call returns all ``SEQ_LEN`` columns.
    The result is well-formed and finite, and a caller that indexes it by rollout position reads the wrong
    tokens.
    """
    model = _patched_model()
    input_ids = _input_ids()
    kept = torch.tensor([0])

    out = model(input_ids=input_ids, logits_to_keep=kept)

    assert len(model.received) == 1
    assert model.received[0] is not NOT_PASSED, "the selector never reached the wrapped forward"
    expected = model.lm_head(model.model.emb(input_ids)[:, kept, :])
    assert tuple(out.logits.shape) == (1, 1, VOCAB_SIZE)
    torch.testing.assert_close(out.logits, expected)


@pytest.mark.parametrize("count", [1, 2, SEQ_LEN])
def test_delegated_forward_passes_an_int_count_through(count):
    """A positive ``int`` count keeps the last ``count`` columns."""
    model = _patched_model()
    input_ids = _input_ids()

    out = model(input_ids=input_ids, logits_to_keep=count)

    assert model.received == [count]
    expected = model.lm_head(model.model.emb(input_ids)[:, -count:, :])
    assert tuple(out.logits.shape) == (1, count, VOCAB_SIZE)
    torch.testing.assert_close(out.logits, expected)


def test_delegated_forward_omits_a_zero_int_count():
    """``logits_to_keep=0`` is the "no selection" value and is not forwarded.

    Dropping it keeps the wrapped forward on its own default, which matters for a model whose signature does not
    accept the argument at all; the kept width is the full sequence either way.
    """
    model = _patched_model()
    input_ids = _input_ids()

    out = model(input_ids=input_ids, logits_to_keep=0)

    assert model.received == [NOT_PASSED]
    assert tuple(out.logits.shape) == (1, SEQ_LEN, VOCAB_SIZE)
