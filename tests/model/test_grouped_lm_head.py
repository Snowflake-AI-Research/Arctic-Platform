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
"""Grouped outputs from both chunked LM heads against dense math."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from arctic_platform.model.implementations.gpu.lm_head import chunked_lm_head_logprobs
from arctic_platform.model.implementations.gpu.lm_head import enable_chunked_lm_head_logprobs
from arctic_platform.model.implementations.moe.layers.lm_head import FusedOutputLinear
from arctic_platform.model.implementations.moe.layers.lm_head import cast_float_and_contiguous
from arctic_platform.model.implementations.moe.layers.lm_head import inject_prime_lm_head
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal

VOCAB = 2 * 8192 + 37
FLOAT32_ATOL = 3e-5


def _dense_groups(logits, token_ids):
    member = token_ids >= 0
    safe_ids = token_ids.clamp(min=0).long()
    in_group = torch.zeros(logits.shape, dtype=torch.int32).scatter_add_(-1, safe_ids, member.int()).gt(0)
    complement = logits.masked_fill(in_group, float("-inf"))
    has_tail = torch.isfinite(complement).any(-1, keepdim=True)
    tail = complement.masked_fill(~has_tail, 0.0).logsumexp(-1, keepdim=True)
    tail = tail.masked_fill(~has_tail, float("-inf"))
    candidates = logits.gather(-1, safe_ids).masked_fill(~member, float("-inf"))
    return torch.cat((candidates, tail), -1) - logits.logsumexp(-1, keepdim=True)


def _head(kind, hidden, weight, labels, token_ids, chunks, **kwargs):
    token_chunk, vocab_chunk = chunks
    if kind == "moe":
        head = FusedOutputLinear(
            hidden.shape[-1],
            weight.shape[0],
            token_chunk,
            fp32_lm_head=True,
        )
        if token_ids is not None:
            kwargs["group_token_ids"] = token_ids
        return torch.func.functional_call(head, {"weight": weight}, (hidden, labels), kwargs)
    return chunked_lm_head_logprobs(
        hidden,
        weight,
        labels,
        token_chunk_size=token_chunk,
        vocab_chunk_size=vocab_chunk,
        fp32_lm_head=True,
        group_token_ids=token_ids,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("kind", "chunks"),
    [
        ("moe", (3, None)),
        ("moe", (16, None)),
        ("dense", (4, 5000)),
        ("dense", (3, VOCAB)),
    ],
)
def test_both_heads_match_dense_group_values_and_gradients(kind, chunks):
    set_seed(0)
    hidden = torch.randn(1, 10, 4)
    weight = 0.5 * torch.randn(VOCAB, 4)
    labels = torch.randint(0, VOCAB, (1, 10))
    token_ids = torch.randint(0, VOCAB, (1, 10, 4), dtype=torch.int32)
    token_ids[0, 0] = torch.tensor([8191, 8192, 16383, 16384])
    token_ids[0, 1, 0] = labels[0, 1]
    token_ids[0, 1, 1] = labels[0, 1]
    token_ids[0, 2, 2:] = -1
    token_ids[0, 3] = -1
    labels[0, 4] = -100
    hidden[0, 5] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    weight[:, 3] = 0.0
    weight[token_ids[0, 5].long(), 3] = torch.tensor([30.0, 29.0, 28.0, 27.0])
    token_ids[0, 7, 0] = labels[0, 7]

    allowed = [
        sorted({*token_ids[0, 6, :2].tolist(), int(labels[0, 6]), 5}),
        sorted(set(token_ids[0, 7].tolist())),
    ]
    masks = {
        "seq_len": 10,
        "vocab_size": VOCAB,
        "positions": [7, 8],
        "set_indices": [0, 1],
        "set_modes_allow": [True, True],
        "set_offsets": [0, len(allowed[0]), len(allowed[0]) + len(allowed[1])],
        "token_ids": allowed[0] + allowed[1],
    }
    disallowed = torch.ones(1, 10, VOCAB, dtype=torch.bool)
    disallowed[0, :6] = False
    disallowed[0, 8:] = False
    for position, tokens in zip((6, 7), allowed):
        disallowed[0, position, tokens] = False
    output_gradients = (
        torch.randn(1, 10),
        torch.randn(1, 10, 5),
    )

    actual_hidden = hidden.clone().requires_grad_(True)
    actual_weight = weight.clone().requires_grad_(True)
    output = _head(
        kind,
        actual_hidden,
        actual_weight,
        labels,
        token_ids,
        chunks,
        action_masks=masks,
    )
    actual = (output["logprobs"], output["group_log_probs"]) if kind == "moe" else output
    torch.autograd.backward(actual, output_gradients)

    reference_hidden = hidden.double().requires_grad_(True)
    reference_weight = weight.double().requires_grad_(True)
    logits = (reference_hidden @ reference_weight.T).masked_fill(disallowed, float("-inf"))
    expected = (
        logits.log_softmax(-1).gather(-1, labels.clamp(min=0)[..., None]).squeeze(-1),
        _dense_groups(logits, token_ids),
    )
    sum(
        (gradient * torch.where(torch.isfinite(reference), reference, 0.0)).sum()
        for reference, gradient in zip(expected, output_gradients)
    ).backward()

    assert float(expected[1][0, 5, -1].detach()) < -20.0
    assert torch.isneginf(expected[1][0, 7, -1])
    for observed, reference in zip(actual, expected):
        torch_assert_close(observed.double(), reference.detach(), rtol=0, atol=FLOAT32_ATOL)
    torch_assert_close(actual_hidden.grad.double(), reference_hidden.grad, rtol=0, atol=FLOAT32_ATOL)
    torch_assert_close(actual_weight.grad.double(), reference_weight.grad, rtol=0, atol=FLOAT32_ATOL)

    default = _head(kind, hidden, weight, labels, None, chunks, action_masks=masks)
    if kind == "moe":
        assert set(default) == {"logprobs", "entropy"}
        default_logprobs = default["logprobs"]
    else:
        assert torch.is_tensor(default)
        default_logprobs = default
    torch_assert_equal(default_logprobs, actual[0].detach())


@pytest.mark.parametrize("kind,chunks", [("moe", (2, None)), ("dense", (2, 5))])
def test_grouped_heads_validate_temperature_ids_and_shape(kind, chunks):
    hidden = torch.randn(1, 2, 3)
    weight = torch.randn(7, 3)
    labels = torch.tensor([[1, 2]])
    token_ids = torch.tensor([[[1, 2], [2, 3]]], dtype=torch.int32)

    with pytest.raises(ValueError, match=r"received temperature values \[0\.699"):
        _head(
            kind,
            hidden,
            weight,
            labels,
            token_ids,
            chunks,
            temperature=torch.full((1, 2), 0.7),
        )
    with pytest.raises(ValueError, match=r"token id 7 outside the vocabulary \[0, 7\)"):
        _head(kind, hidden, weight, labels, torch.tensor([[[1, 7], [2, 3]]]), chunks)
    with pytest.raises(ValueError, match="must contain integer token ids"):
        _head(kind, hidden, weight, labels, token_ids.float(), chunks)
    with pytest.raises(ValueError, match="same candidate width"):
        _head(kind, hidden, weight, labels, torch.tensor([1, 2, 3]), chunks)


@pytest.mark.parametrize("kind,chunks", [("moe", (2, None)), ("dense", (2, 5))])
@pytest.mark.parametrize("temperature", [1, torch.tensor(1.0), torch.ones(1, 2, dtype=torch.bfloat16)])
def test_grouped_heads_preserve_neutral_temperature_math(kind, chunks, temperature):
    set_seed(9)
    hidden = torch.randn(1, 2, 3)
    weight = torch.randn(7, 3)
    labels = torch.tensor([[1, 2]])
    token_ids = torch.tensor([[[1, 5], [2, 6]]], dtype=torch.int32)

    expected = _head(kind, hidden, weight, labels, token_ids, chunks)
    observed = _head(
        kind,
        hidden,
        weight,
        labels,
        token_ids,
        chunks,
        temperature=temperature,
    )
    if kind == "moe":
        expected = expected["logprobs"], expected["group_log_probs"]
        observed = observed["logprobs"], observed["group_log_probs"]
    for actual, reference in zip(observed, expected):
        torch_assert_equal(actual, reference)
    torch_assert_close(observed[1].exp().sum(-1), torch.ones_like(observed[1][..., 0]), rtol=0, atol=3e-7)


class _Backbone(nn.Module):
    def __init__(self, hidden_states):
        super().__init__()
        self.hidden_states = hidden_states

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, **kwargs):
        return SimpleNamespace(last_hidden_state=self.hidden_states)


class _CausalLM(nn.Module):
    def __init__(self, hidden_states, weight):
        super().__init__()
        self.model = _Backbone(hidden_states)
        self.lm_head = nn.Linear(weight.shape[1], weight.shape[0], bias=False)
        self.lm_head.weight = nn.Parameter(weight.clone())
        self.config = SimpleNamespace(final_logit_softcapping=None)

    def forward(self, **kwargs):
        return kwargs


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_patched_model_forwards_slice_group_ids_with_logits_to_keep(kind):
    set_seed(17)
    hidden = torch.randn(1, 4, 3)
    weight = torch.randn(11, 3)
    labels = torch.tensor([[1, 2, 3, 4]])
    token_ids = torch.tensor([[[1, 5], [2, 6], [3, 7], [4, 8]]], dtype=torch.int32)
    keep = torch.tensor([1, 3])
    model = _CausalLM(hidden, weight)
    if kind == "dense":
        enable_chunked_lm_head_logprobs(
            model,
            token_chunk_size=2,
            vocab_chunk_size=5,
            fp32_lm_head=True,
        )
        output = model(
            input_ids=torch.ones(1, 4, dtype=torch.long),
            labels=labels,
            logits_to_keep=keep,
            dss_compute_logprobs=True,
            group_token_ids=token_ids,
        )
    else:
        inject_prime_lm_head(model, chunk_size=2, fp32_lm_head=True)
        output = model(
            input_ids=torch.ones(1, 4, dtype=torch.long),
            labels=labels,
            logits_to_keep=keep,
            dss_compute_logprobs=True,
            group_token_ids=token_ids,
        )

    logits = hidden[:, keep] @ weight.T
    expected_logprobs = logits.log_softmax(-1).gather(-1, labels[:, keep, None]).squeeze(-1)
    expected_groups = _dense_groups(logits.double(), token_ids[:, keep]).float()
    torch_assert_close(output["logprobs"], expected_logprobs, rtol=0, atol=1e-6)
    torch_assert_close(output["group_log_probs"], expected_groups, rtol=0, atol=1e-6)


def test_moe_output_cast_preserves_group_log_probs():
    output = cast_float_and_contiguous(
        {
            "logprobs": torch.ones(2, dtype=torch.float64),
            "group_log_probs": torch.ones(2, 3, dtype=torch.float64),
        }
    )
    assert output["group_log_probs"].dtype == torch.float32
    assert output["group_log_probs"].is_contiguous()
