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

"""``FusedOutputLinear`` never reports a score for a target it did not gather.

The Qwen3.5 chunked head runs the same tiled scheme as ``chunked_lm_head_logprobs``: a position's target logit
is written only while iterating the vocab tile whose index range contains that position's label, so a label
outside ``[0, vocab_size)`` leaves the entry at its initialized zero and the head reports
``0 - logsumexp(logits)``. The ignore-index sentinel ``-100`` is exactly such a label, and it is the one that
arrives in normal use, from padded rows and from the last position of a split row.

This head is bias-free and returns entropy alongside the logprobs. Entropy is a function of the whole logit row
and not of the label, so the no-regression case below pins it as well.
"""

from __future__ import annotations

import pytest
import torch

from arctic_platform.model.implementations.qwen35.models.layers.lm_head import FusedOutputLinear

# The oracle is a dense fp32 ``log_softmax``; the tiled path differs from it only in log-sum-exp accumulation
# order, and on these shapes (fp32, CPU) that floor measures 0.02e-05 across values, entropy, and gradients.
# The asserted absolute bound of 1.00e-05 is fifty times the floor and still far below the error of a target
# logit taken from no vocab tile, which is 0.54 nats here -- five decades above the bound.
RTOL = 1e-5
ATOL = 1e-5

IGNORE_INDEX = -100
VOCAB_SIZE = 6
HIDDEN_SIZE = 3
CHUNK_SIZE = 2


def _dense_logprobs_and_entropy(hidden, weight, labels, *, temperature=None):
    """Oracle over materialized ``[B, S, V]`` logits. ``labels`` must be in ``[0, vocab_size)``."""
    logits = torch.einsum("bsh,vh->bsv", hidden.float(), weight.float())
    if temperature is not None:
        safe = temperature.masked_fill(temperature == 0, 1.0)
        logits = logits / safe.unsqueeze(-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1), entropy


def _substituted(labels):
    """``labels`` with the sentinel replaced by the vocabulary index the head scores it at."""
    return labels.masked_fill(labels == IGNORE_INDEX, 0)


def test_ignored_target_is_scored_at_the_substitute_vocabulary_entry():
    """Value and gradient at an ignored position are those of the substitute entry, at every position.

    The gradient weight at the ignored position is non-zero so the backward pass is held to the same
    substitution as the forward, and the whole tensor is asserted because the supervised positions are already
    correct without the guard.
    """
    torch.manual_seed(3)
    hidden = torch.randn(1, 4, HIDDEN_SIZE, requires_grad=True)
    head = FusedOutputLinear(HIDDEN_SIZE, VOCAB_SIZE, CHUNK_SIZE, fp32_lm_head=True)
    labels = torch.tensor([[1, IGNORE_INDEX, 3, 4]])
    grad_weights = torch.tensor([[1.0, 0.75, 0.5, 0.25]])

    actual = head(hidden, labels)["logprobs"]
    (actual * grad_weights).sum().backward()

    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = head.weight.detach().clone().requires_grad_(True)
    expected, _ = _dense_logprobs_and_entropy(hidden_ref, weight_ref, _substituted(labels))
    (expected * grad_weights).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(hidden.grad, hidden_ref.grad, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(head.weight.grad, weight_ref.grad, rtol=RTOL, atol=ATOL)


def test_ignored_target_never_reports_a_positive_log_probability():
    """No position may score above zero, whatever the labels.

    An ungathered target logit is reported as ``0 - logsumexp(logits)``, which exceeds zero exactly when the
    partition function is below one. Every logit here is near ``-9``, putting the ignored position's reported
    score near ``+7.2``. The bound is the definition of a log-probability, not a tolerance.
    """
    torch.manual_seed(17)
    hidden = 1.0 + 0.1 * torch.randn(1, 3, HIDDEN_SIZE)
    head = FusedOutputLinear(HIDDEN_SIZE, VOCAB_SIZE, CHUNK_SIZE, fp32_lm_head=True)
    with torch.no_grad():
        head.weight.copy_(-3.0 + 0.1 * torch.randn(VOCAB_SIZE, HIDDEN_SIZE))
    labels = torch.tensor([[2, IGNORE_INDEX, 4]])

    with torch.no_grad():
        actual = head(hidden, labels)["logprobs"]

    assert torch.isfinite(actual).all()
    assert float(actual.max()) <= 0.0, f"reported a log-probability above zero: {actual.tolist()}"


@pytest.mark.parametrize("bad_label", [-1, -99, -101, VOCAB_SIZE, VOCAB_SIZE + 1])
def test_labels_outside_the_vocabulary_are_rejected(bad_label):
    """A label that is neither a vocabulary index nor the sentinel is a caller error, not a score.

    Both sides of the tile-membership comparison are covered, and both neighbours of the sentinel, since a
    check written as ``labels < IGNORE_INDEX`` or as equality against the wrong sign would admit one of them.
    """
    hidden = torch.randn(1, 4, HIDDEN_SIZE)
    head = FusedOutputLinear(HIDDEN_SIZE, VOCAB_SIZE, CHUNK_SIZE, fp32_lm_head=True)
    labels = torch.tensor([[1, bad_label, 3, 4]])

    with pytest.raises(ValueError, match="outside the vocabulary"):
        head(hidden, labels)


def test_a_constraint_at_an_ignored_position_is_dropped_and_the_others_are_kept():
    """Scoring the sentinel at a substitute entry must not put that entry under a constraint.

    An allow-set excluding the substitute drives its logit to ``-inf``, which makes the reported score
    meaningless and lets the sampled-token check reject a position the caller asked to ignore. A constraint at
    a supervised position is present too, so discarding every constraint would fail the same assertion.
    """
    torch.manual_seed(5)
    hidden = torch.randn(1, 4, HIDDEN_SIZE)
    head = FusedOutputLinear(HIDDEN_SIZE, VOCAB_SIZE, CHUNK_SIZE, fp32_lm_head=True)
    labels = torch.tensor([[1, IGNORE_INDEX, 3, 4]])
    # Target position 2 constrains source position 1, whose label is the sentinel: allow token 1 only, which
    # excludes the substitute entry. Target position 3 constrains source position 2, whose label is 3: allow
    # tokens 3 and 5, which keeps its own label reachable.
    action_masks = {
        "seq_len": 4,
        "vocab_size": VOCAB_SIZE,
        "positions": [2, 3],
        "set_indices": [0, 1],
        "set_modes_allow": [True, True],
        "set_offsets": [0, 1, 3],
        "token_ids": [1, 3, 5],
    }

    actual = head(hidden, labels, action_masks=action_masks)["logprobs"]

    logits = torch.einsum("bsh,vh->bsv", hidden.float(), head.weight.detach().float())
    # The surviving constraint at source position 2 alone; source position 1 is scored unconstrained.
    logits[0, 2, [token for token in range(VOCAB_SIZE) if token not in (3, 5)]] = float("-inf")
    expected = torch.log_softmax(logits, dim=-1)
    expected = expected.gather(-1, _substituted(labels).unsqueeze(-1)).squeeze(-1)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_labels_all_inside_the_vocabulary_reproduce_the_dense_log_softmax():
    """No-regression: with nothing to substitute, logprobs, entropy, and gradient are unchanged.

    ``chunk_size=3`` over 5 positions leaves a two-position trailing tile, and per-position temperature is
    supplied because the domain check runs on the flattened labels before the tile loop and must not disturb
    the bounds the temperature is sliced with.
    """
    torch.manual_seed(11)
    hidden = torch.randn(1, 5, 4, requires_grad=True)
    head = FusedOutputLinear(4, 7, 3, fp32_lm_head=True)
    labels = torch.tensor([[0, 1, 2, 3, 4]])
    temperature = torch.tensor([[0.7, 1.1, 0.9, 1.3, 0.8]], dtype=torch.float32)
    grad_weights = torch.linspace(0.1, 1.0, steps=5).reshape(1, 5)

    out = head(hidden, labels, temperature=temperature)
    actual = out["logprobs"]
    (actual * grad_weights).sum().backward()

    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = head.weight.detach().clone().requires_grad_(True)
    expected, expected_entropy = _dense_logprobs_and_entropy(hidden_ref, weight_ref, labels, temperature=temperature)
    (expected * grad_weights).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(out["entropy"], expected_entropy, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(hidden.grad, hidden_ref.grad, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(head.weight.grad, weight_ref.grad, rtol=RTOL, atol=ATOL)
