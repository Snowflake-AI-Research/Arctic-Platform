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

"""A near-certain target keeps its log-probability and its gradient in both chunked LM heads.

When the target logit leads the logsumexp of the other logits by a margin ``m``, its log-probability is
``-log(1 + exp(-m))``. A logsumexp that sums the target's own ``exp(0) = 1`` with the rest rounds that to exactly 0
once ``exp(-m)`` falls below float32 resolution (``m`` past ~16.6, sooner beside a large max logit), and a backward
that forms ``1 - softmax`` at the target then reports exactly 0 there while every other entry keeps its gradient,
so the logit gradient no longer sums to zero (https://jonbarron.info/softmaxfix/).
"""

import pytest
import torch

from arctic_platform.model.implementations.gpu.lm_head import chunked_lm_head_logprobs
from arctic_platform.model.implementations.moe.layers.lm_head import FusedOutputLinear
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close

VOCAB_SIZE = 64
LABELS = torch.tensor([5, 37, 60])
TARGET_LOGIT = 30.0


def _head(kind, hidden, labels):
    if kind == "moe":
        head = FusedOutputLinear(VOCAB_SIZE, VOCAB_SIZE, 2, fp32_lm_head=True)
        output = torch.func.functional_call(head, {"weight": torch.eye(VOCAB_SIZE)}, (hidden, labels))
        return output["logprobs"], output["entropy"]
    logprobs = chunked_lm_head_logprobs(
        hidden, torch.eye(VOCAB_SIZE), labels, token_chunk_size=2, vocab_chunk_size=16, fp32_lm_head=True
    )
    return logprobs, None


@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("margin", [4.0, 10.0, 14.0, 17.0, 20.0])
def test_near_certain_target_matches_float64_log_softmax(kind, margin):
    """Value and gradient match a float64 ``log_softmax`` to float32 relative precision at every margin.

    Float64's own rounding is ``2.2e-16 * exp(margin)`` relative, 1.1e-7 at the largest margin here, so the
    reference holds well inside the bound. The pre-fix heads report 0 from a margin of 14 on.
    """
    set_seed(0)
    logits = torch.randn(len(LABELS), VOCAB_SIZE)
    rows = torch.arange(len(LABELS))
    others = logits.clone()
    others[rows, LABELS] = float("-inf")
    logits += TARGET_LOGIT - margin - others.logsumexp(-1, keepdim=True)
    logits[rows, LABELS] = TARGET_LOGIT

    hidden = logits.reshape(1, len(LABELS), VOCAB_SIZE).requires_grad_(True)
    actual, entropy = _head(kind, hidden, LABELS.reshape(1, -1))
    actual.sum().backward()

    reference = logits.double().requires_grad_(True)
    log_probs = torch.log_softmax(reference, -1)
    expected = log_probs[rows, LABELS]
    expected.sum().backward()

    torch_assert_close(actual.reshape(-1).double(), expected.detach(), rtol=1e-5, atol=0.0)
    torch_assert_close(hidden.grad.reshape(len(LABELS), -1).double(), reference.grad, rtol=1e-5, atol=1e-30)
    if entropy is not None:
        expected_entropy = -(log_probs.exp() * log_probs).sum(-1).detach()
        torch_assert_close(entropy.reshape(-1).double(), expected_entropy, rtol=0.0, atol=1e-5)


def _near_certain_logits(margin):
    set_seed(0)
    logits = torch.randn(len(LABELS), VOCAB_SIZE)
    rows = torch.arange(len(LABELS))
    others = logits.clone()
    others[rows, LABELS] = float("-inf")
    logits += TARGET_LOGIT - margin - others.logsumexp(-1, keepdim=True)
    logits[rows, LABELS] = TARGET_LOGIT
    return logits


@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("margin", [14.0, 20.0])
def test_near_certain_target_gradient_with_grouped_outputs(kind, margin):
    """With grouped outputs the normalizer gradient differs from the logprob gradient, and the target entry still
    matches float64.

    The candidates exclude the target and the tail gradient is 0: the target entry is then exactly
    ``g * (1 - p_t) - p_t * sum(candidate gradients)``, the rewritten expression. A nonzero tail gradient adds
    ``p_t / p_tail - p_t`` at the target, which these heads do not compute without cancellation.
    """
    logits = _near_certain_logits(margin)
    rows = torch.arange(len(LABELS))
    candidates = torch.stack(((LABELS + 1) % VOCAB_SIZE, (LABELS + 2) % VOCAB_SIZE), -1)
    group_token_ids = candidates.reshape(1, len(LABELS), 2).to(torch.int32)
    logprob_gradients = torch.randn(1, len(LABELS))
    group_gradients = torch.randn(1, len(LABELS), 3)
    group_gradients[..., -1] = 0.0

    hidden = logits.reshape(1, len(LABELS), VOCAB_SIZE).requires_grad_(True)
    labels = LABELS.reshape(1, -1)
    if kind == "moe":
        head = FusedOutputLinear(VOCAB_SIZE, VOCAB_SIZE, 2, fp32_lm_head=True)
        output = torch.func.functional_call(
            head, {"weight": torch.eye(VOCAB_SIZE)}, (hidden, labels), {"group_token_ids": group_token_ids}
        )
        actual, actual_groups = output["logprobs"], output["group_log_probs"]
    else:
        actual, actual_groups = chunked_lm_head_logprobs(
            hidden,
            torch.eye(VOCAB_SIZE),
            labels,
            token_chunk_size=2,
            vocab_chunk_size=16,
            fp32_lm_head=True,
            group_token_ids=group_token_ids,
        )
    torch.autograd.backward((actual, actual_groups), (logprob_gradients, group_gradients))

    reference = logits.double().requires_grad_(True)
    log_probs = torch.log_softmax(reference, -1)
    expected = log_probs[rows, LABELS]
    expected_candidates = log_probs.gather(-1, candidates)
    (
        (logprob_gradients.reshape(-1).double() * expected).sum()
        + (group_gradients[0, :, :2].double() * expected_candidates).sum()
    ).backward()

    torch_assert_close(actual.reshape(-1).double(), expected.detach(), rtol=1e-5, atol=0.0)
    torch_assert_close(actual_groups[0, :, :2].double(), expected_candidates.detach(), rtol=1e-5, atol=0.0)
    torch_assert_close(hidden.grad.reshape(len(LABELS), -1).double(), reference.grad, rtol=1e-5, atol=1e-30)


@pytest.mark.parametrize("kind", ["dense", "moe"])
@pytest.mark.parametrize("margin", [14.0, 20.0])
def test_near_certain_target_with_bfloat16_lm_head(kind, margin):
    """``fp32_lm_head=False`` keeps the matmuls in bfloat16 and the target's value and gradient nonzero.

    The logits are bfloat16 inputs times an identity weight, so the head sees them exactly; the reference is
    float64 on those values. The gradient comes back in bfloat16, so it is compared to bfloat16 precision.
    """
    logits = _near_certain_logits(margin).to(torch.bfloat16)
    rows = torch.arange(len(LABELS))
    hidden = logits.reshape(1, len(LABELS), VOCAB_SIZE).requires_grad_(True)
    weight = torch.eye(VOCAB_SIZE, dtype=torch.bfloat16)
    labels = LABELS.reshape(1, -1)
    if kind == "moe":
        head = FusedOutputLinear(VOCAB_SIZE, VOCAB_SIZE, 2, fp32_lm_head=False)
        actual = torch.func.functional_call(head, {"weight": weight}, (hidden, labels))["logprobs"]
    else:
        actual = chunked_lm_head_logprobs(
            hidden, weight, labels, token_chunk_size=2, vocab_chunk_size=16, fp32_lm_head=False
        )
    actual.sum().backward()

    reference = logits.double().requires_grad_(True)
    expected = torch.log_softmax(reference, -1)[rows, LABELS]
    expected.sum().backward()

    assert (actual != 0).all()
    torch_assert_close(actual.reshape(-1).double(), expected.detach(), rtol=1e-5, atol=0.0)
    target_gradients = hidden.grad.reshape(len(LABELS), -1)[rows, LABELS]
    assert (target_gradients != 0).all()
    torch_assert_close(hidden.grad.reshape(len(LABELS), -1).double(), reference.grad, rtol=1e-2, atol=1e-30)
