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

"""A near-certain target keeps its log-probability and its gradient in the chunked LM head.

When the target logit leads the logsumexp of the other logits by a margin ``m``, its log-probability is
``-log(1 + exp(-m))``. A logsumexp that sums the target's own ``exp(0) = 1`` with the rest rounds that to exactly 0
once ``exp(-m)`` falls below float32 resolution (``m`` past ~16.6, sooner beside a large max logit), and a backward
that forms ``1 - softmax`` at the target then reports exactly 0 there while every other entry keeps its gradient,
so the logit gradient no longer sums to zero (https://jonbarron.info/softmaxfix/).
"""

import pytest
import torch

from arctic_platform.model.implementations.gpu.lm_head import chunked_lm_head_logprobs

VOCAB_SIZE = 64
LABELS = torch.tensor([5, 37, 60])
TARGET_LOGIT = 30.0


@pytest.mark.parametrize("margin", [4.0, 10.0, 14.0, 17.0, 20.0])
def test_near_certain_target_matches_float64_log_softmax(margin):
    """Value and gradient match a float64 ``log_softmax`` to float32 relative precision at every margin.

    Float64's own rounding is ``2.2e-16 * exp(margin)`` relative, 1.1e-7 at the largest margin here, so the
    reference holds well inside the bound. The pre-fix head reports 0 from a margin of 14 on.
    """
    torch.manual_seed(0)
    logits = torch.randn(len(LABELS), VOCAB_SIZE)
    rows = torch.arange(len(LABELS))
    others = logits.clone()
    others[rows, LABELS] = float("-inf")
    logits += TARGET_LOGIT - margin - others.logsumexp(-1, keepdim=True)
    logits[rows, LABELS] = TARGET_LOGIT

    hidden = logits.reshape(1, len(LABELS), VOCAB_SIZE).requires_grad_(True)
    actual = chunked_lm_head_logprobs(
        hidden,
        torch.eye(VOCAB_SIZE),
        LABELS.reshape(1, -1),
        token_chunk_size=2,
        vocab_chunk_size=16,
        fp32_lm_head=True,
    )
    actual.sum().backward()

    reference = logits.double().requires_grad_(True)
    expected = torch.log_softmax(reference, -1)[rows, LABELS]
    expected.sum().backward()

    torch.testing.assert_close(actual.reshape(-1).double(), expected.detach(), rtol=1e-5, atol=0.0)
    torch.testing.assert_close(hidden.grad.reshape(len(LABELS), -1).double(), reference.grad, rtol=1e-5, atol=1e-30)
