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

"""The unchunked compute_logprobs path keeps a near-certain label's log-probability and gradient.

torch.log_softmax folds the argmax's own exp(0) = 1 into its normalizer, so once the label leads the other logits by
more than ~16.6 nats in float32 its log-probability rounds to exactly 0 and its logit gradient to 0 while the others keep
theirs (https://jonbarron.info/softmaxfix/). Writing the normalizer as softplus(logsumexp(others)) fixes that but turns
the gradient into NaN when every other logit is -inf (arXiv:2610.36427); log1p of the others' sum avoids both.
"""

import pytest
import torch

from arctic_platform.rl.processors.compute_logprobs import compute_logprobs_post
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close

VOCAB_SIZE = 64


def _logprobs(logits, labels):
    logits = logits.clone().requires_grad_(True)
    out = compute_logprobs_post({"logits": logits}, {}, {"labels": labels}, "cpu")["logprobs"]
    out.sum().backward()
    return out.detach(), logits.grad


@pytest.mark.parametrize("margin", [4.0, 10.0, 14.0, 17.0, 20.0])
def test_near_certain_label_matches_float64_log_softmax(margin):
    """Value and gradient match float64 at every margin, for a near-certain label and for one beside it."""
    set_seed(0)
    logits = torch.randn(1, 2, VOCAB_SIZE)
    logits[..., 5] = 0.0
    others = logits.clone()
    others[..., 5] = float("-inf")
    logits += 30.0 - margin - others.logsumexp(-1, keepdim=True)
    logits[..., 5] = 30.0
    labels = torch.tensor([[5, int(others[0, 1].argmax())]])

    actual, grad = _logprobs(logits, labels)

    reference = logits.double().requires_grad_(True)
    expected = torch.log_softmax(reference, -1).gather(-1, labels[..., None]).squeeze(-1)
    expected.sum().backward()
    torch_assert_close(actual.double(), expected.detach(), rtol=1e-5, atol=0.0)
    torch_assert_close(grad.double(), reference.grad, rtol=1e-5, atol=1e-30)


def test_label_alone_against_minus_infinity_or_overflow_keeps_a_finite_gradient():
    """Every other logit -inf, or pushed to -inf by the max shift: log-probability 0 and an all-zero, finite gradient."""
    logits = torch.full((1, 2, VOCAB_SIZE), float("-inf"))
    logits[0, 0, 3] = 2.0
    logits[0, 1] = -3e38
    logits[0, 1, 3] = 3e38

    actual, grad = _logprobs(logits, torch.tensor([[3, 3]]))

    torch_assert_close(actual, torch.zeros(1, 2))
    torch_assert_close(grad, torch.zeros_like(grad))
