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

"""The chunked LM head's bf16 hidden-state gradient is summed across vocab tiles in fp32.

Each vocab tile adds one ``grad_logits @ weight`` product to the hidden-state gradient. Summing 64 of them in bf16
puts this fixture 1.1% from the fp32 oracle; summing in fp32 leaves only each product's own rounding, 0.22%.
"""

from __future__ import annotations

import torch

from arctic_platform.model.implementations.gpu.lm_head import chunked_lm_head_logprobs

TOKENS = 64
HIDDEN_SIZE = 256
VOCAB_SIZE = 8192
VOCAB_CHUNK = 128
MAX_RELATIVE_ERROR = 5e-3


def test_bf16_hidden_gradient_is_accumulated_in_fp32():
    torch.manual_seed(0)
    hidden = torch.randn(1, TOKENS, HIDDEN_SIZE).bfloat16()
    weight = (torch.randn(VOCAB_SIZE, HIDDEN_SIZE) / HIDDEN_SIZE**0.5).bfloat16()
    labels = torch.randint(0, VOCAB_SIZE, (1, TOKENS))

    actual = hidden.clone().requires_grad_(True)
    chunked_lm_head_logprobs(
        actual, weight, labels, token_chunk_size=TOKENS, vocab_chunk_size=VOCAB_CHUNK, fp32_lm_head=False
    ).sum().backward()

    expected = hidden.float().requires_grad_(True)
    torch.log_softmax(expected @ weight.float().t(), dim=-1).gather(-1, labels.unsqueeze(-1)).sum().backward()

    error = (actual.grad.float() - expected.grad).norm() / expected.grad.norm()
    assert error < MAX_RELATIVE_ERROR, f"hidden gradient is {error:.2e} from the fp32 oracle"
