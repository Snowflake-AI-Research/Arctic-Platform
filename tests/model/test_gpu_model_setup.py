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

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from arctic_platform.model import ActivationOffloadConfig
from arctic_platform.model.implementations.gpu.lm_head import chunked_lm_head_logprobs
from arctic_platform.model.implementations.gpu.lm_head import enable_fp32_lm_head


def test_chunked_lm_head_matches_full_projection_and_gradients():
    torch.manual_seed(7)
    hidden = torch.randn(2, 5, 6, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(11, 6, dtype=torch.float64, requires_grad=True)
    labels = torch.randint(0, 11, (2, 5))
    temperature = torch.rand(2, 5, dtype=torch.float64) + 0.5

    actual = chunked_lm_head_logprobs(
        hidden,
        weight,
        labels,
        temperature=temperature,
        token_chunk_size=3,
        vocab_chunk_size=4,
        fp32_lm_head=False,
    )
    expected = (
        torch.log_softmax(
            (hidden @ weight.t()).float() / temperature.float().unsqueeze(-1),
            dim=-1,
        )
        .gather(-1, labels.unsqueeze(-1))
        .squeeze(-1)
    )
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    actual.sum().backward(retain_graph=True)
    actual_hidden_grad = hidden.grad.detach().clone()
    actual_weight_grad = weight.grad.detach().clone()
    hidden.grad = None
    weight.grad = None
    expected.sum().backward()
    torch.testing.assert_close(hidden.grad, actual_hidden_grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(weight.grad, actual_weight_grad, rtol=1e-6, atol=1e-6)


def test_fp32_lm_head_projection():
    model = nn.Module()
    model.lm_head = nn.Linear(4, 7, bias=False, dtype=torch.bfloat16)
    enable_fp32_lm_head(model)

    output = model.lm_head(torch.randn(3, 4, dtype=torch.bfloat16))

    assert output.dtype == torch.float32
    assert model.lm_head._dss_fp32_lm_head is True


def test_activation_offload_config_rejects_negative_pin_memory_limit():
    with pytest.raises(ValueError, match="non-negative"):
        ActivationOffloadConfig(pin_memory_max_size_gib=-1)
