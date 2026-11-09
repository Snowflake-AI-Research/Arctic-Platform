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

"""PrimeRL model forwards preserve recoverable LM-head target validation."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from arctic_platform.model.implementations.glm52.models.glm_moe_dsa.modeling_glm_moe_dsa import GlmMoeDsaForCausalLM
from arctic_platform.model.implementations.gpu.lm_head import mark_lm_head_targets_validated
from arctic_platform.model.implementations.gpu.lm_head import validate_lm_head_targets
from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM


class _Backbone(nn.Module):
    def forward(self, input_ids=None, **_kwargs):
        return SimpleNamespace(
            last_hidden_state=torch.zeros(input_ids.shape[0], input_ids.shape[1], 4, device=input_ids.device)
        )


class _RecordingHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.validated_vocab_size = None

    def forward(self, _hidden_states, labels, **_kwargs):
        self.validated_vocab_size = getattr(labels, "_ap_validated_lm_head_vocab_size", None)
        validate_lm_head_targets(labels, vocab_size=7)
        return SimpleNamespace(logprobs=torch.zeros_like(labels, dtype=torch.float32))


@pytest.mark.parametrize(
    ("model_class", "extra_attributes"),
    [
        (Qwen3_5MoeForCausalLM, {"_is_vlm": False}),
        (GlmMoeDsaForCausalLM, {}),
    ],
)
def test_prime_rl_forward_preserves_label_validation_across_logits_slice(model_class, extra_attributes):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    head = _RecordingHead()
    model = SimpleNamespace(model=_Backbone(), lm_head=head, **extra_attributes)
    labels = mark_lm_head_targets_validated(torch.tensor([[1, 2, 3, 4]], device=device), vocab_size=7)

    model_class.forward(
        model,
        input_ids=torch.tensor([[0, 1, 2, 3]], device=device),
        labels=labels,
        logits_to_keep=2,
    )

    assert head.validated_vocab_size == 7


def test_injected_prime_rl_forward_preserves_label_validation_across_logits_slice():
    from arctic_platform.model.implementations.moe.layers.lm_head import _patch_model_forward

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _Backbone()
            self.lm_head = _RecordingHead()

        def forward(self, input_ids=None, **_kwargs):
            return self.model(input_ids=input_ids)

    model = Model()
    _patch_model_forward(model)
    labels = mark_lm_head_targets_validated(torch.tensor([[1, 2, 3, 4]]), vocab_size=7)

    model(
        input_ids=torch.tensor([[0, 1, 2, 3]]),
        labels=labels,
        logits_to_keep=2,
    )

    assert model.lm_head.validated_vocab_size == 7
