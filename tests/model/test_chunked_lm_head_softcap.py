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

"""The chunked LM head applies Gemma's ``final_logit_softcapping`` as the model's own forward does.

Gemma 2 and Gemma 4 compute ``softcap * tanh(logits / softcap)`` in their forward. The chunked branch skips that
forward's projection, so without the cap it scores a different distribution: on a 6-layer Gemma 4 31B fixture the
SFT loss is 921.6 uncapped against 40.8 capped.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from arctic_platform.model.implementations.gpu.lm_head import chunked_lm_head_logprobs
from arctic_platform.model.implementations.gpu.lm_head import enable_chunked_lm_head_logprobs

RTOL = 1e-5
ATOL = 1e-5
VOCAB_SIZE = 7
HIDDEN_SIZE = 4
SOFTCAP = 5.0


def _dense_logprobs(hidden, weight, labels, *, softcap, temperature):
    logits = torch.einsum("bsh,vh->bsv", hidden.float(), weight.float())
    logits = softcap * torch.tanh(logits / softcap)
    return torch.log_softmax(logits / temperature, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def test_softcap_value_and_gradient_match_the_dense_oracle():
    """Logits well past the cap, so an uncapped or wrongly differentiated head misses by whole nats."""
    torch.manual_seed(5)
    hidden = (4.0 * torch.randn(1, 5, HIDDEN_SIZE)).requires_grad_(True)
    weight = (4.0 * torch.randn(VOCAB_SIZE, HIDDEN_SIZE)).requires_grad_(True)
    labels = torch.tensor([[0, 3, 6, 2, 5]])
    temperature = 0.7
    grad_weights = torch.tensor([[1.0, 0.5, 0.25, 2.0, 1.5]])

    actual = chunked_lm_head_logprobs(
        hidden,
        weight,
        labels,
        temperature=temperature,
        token_chunk_size=2,
        vocab_chunk_size=3,
        fp32_lm_head=True,
        softcap=SOFTCAP,
    )
    (actual * grad_weights).sum().backward()

    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    expected = _dense_logprobs(hidden_ref, weight_ref, labels, softcap=SOFTCAP, temperature=temperature)
    (expected * grad_weights).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(hidden.grad, hidden_ref.grad, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=RTOL, atol=ATOL)


class _SoftcappedCausalLM(torch.nn.Module):
    """A composite-config causal LM whose own forward caps its logits, as Gemma 4's does."""

    def __init__(self) -> None:
        super().__init__()

        class _Backbone(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.emb = torch.nn.Embedding(8, HIDDEN_SIZE)

            def forward(self, input_ids=None, **_kwargs):
                return SimpleNamespace(last_hidden_state=self.emb(input_ids))

        self.model = _Backbone()
        self.lm_head = torch.nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)
        text_config = SimpleNamespace(final_logit_softcapping=SOFTCAP)
        self.config = SimpleNamespace(get_text_config=lambda: text_config)

    def forward(self, input_ids=None, **_kwargs):
        logits = self.lm_head(self.model(input_ids=input_ids).last_hidden_state)
        return SimpleNamespace(logits=SOFTCAP * torch.tanh(logits / SOFTCAP))


def test_chunked_branch_scores_what_the_model_forward_scores():
    torch.manual_seed(11)
    model = _SoftcappedCausalLM()
    with torch.no_grad():
        model.lm_head.weight.mul_(8.0)
    enable_chunked_lm_head_logprobs(model, token_chunk_size=2, vocab_chunk_size=3, fp32_lm_head=True)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    labels = torch.tensor([[2, 3, 4, 5, 6]])

    chunked = model(input_ids=input_ids, labels=labels, dss_compute_logprobs=True)["logprobs"]
    logits = model(input_ids=input_ids).logits

    expected = torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(chunked, expected, rtol=RTOL, atol=ATOL)
