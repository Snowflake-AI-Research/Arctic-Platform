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

"""The chunked LM head never reports a score for a target it did not gather.

``chunked_lm_head_logprobs`` streams the vocabulary in tiles and gathers each position's target logit from the
one tile whose index range contains that position's label, leaving ``target_logits`` at the zero it was
initialized to when no tile does. The ignore-index sentinel ``-100`` belongs to no tile, so an unguarded head
reports ``0 - logsumexp(logits)`` at an ignored position: finite, not the log-probability of any token, and
positive whenever every logit at that position is negative.

These tests pin what the head reports at an ignored position instead, pin that any *other* out-of-range label
is refused rather than folded into the same path, and pin that an action-mask constraint recorded at an ignored
position is dropped rather than applied to the substitute entry.

Every fixture is a single row, which is the shape a packed call hands the head and which keeps each assertion
about one identifiable position.
"""

from __future__ import annotations

import pytest
import torch

from arctic_platform.model.implementations.gpu.lm_head import chunked_lm_head_logprobs

# Every comparison below is against a dense fp32 ``log_softmax`` over the whole vocabulary, which differs from
# the tiled path only in the order the log-sum-exp is accumulated. On these shapes (fp32, CPU) that
# accumulation-order floor measures 0.10e-05 in the value and 0.14e-05 in the gradient, so the asserted
# absolute bound of 1.00e-05 is seven times the floor. It leaves no room for a wrong target logit: a target
# logit taken from no vocab tile is off by 0.32 nats here, five decades above the bound.
RTOL = 1e-5
ATOL = 1e-5

IGNORE_INDEX = -100
VOCAB_SIZE = 6
HIDDEN_SIZE = 3


def _dense_logprobs(hidden, weight, labels, *, temperature=None):
    """Oracle: materialize ``[B, S, V]`` logits, ``log_softmax`` them, and gather at ``labels``.

    ``labels`` must already be in ``[0, vocab_size)``; the caller substitutes for the sentinel so that the
    quantity the oracle is asked for is the same one the head is asked for.
    """
    logits = torch.einsum("bsh,vh->bsv", hidden.float(), weight.float())
    if temperature is not None:
        logits = logits / temperature.unsqueeze(-1)
    return torch.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def _substituted(labels):
    """``labels`` with the sentinel replaced by the vocabulary index the head scores it at."""
    return labels.masked_fill(labels == IGNORE_INDEX, 0)


def test_ignored_target_is_scored_at_the_substitute_vocabulary_entry():
    """Value and gradient at an ignored position are those of the substitute entry, at every position.

    Asserting the whole tensor rather than only the supervised positions is the point: the supervised positions
    are already correct without the guard, and the ignored one is where an ungathered target logit shows up.
    The gradient weight at the ignored position is non-zero so that the backward pass is held to the same
    substitution as the forward.
    """
    torch.manual_seed(3)
    hidden = torch.randn(1, 4, HIDDEN_SIZE, requires_grad=True)
    weight = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, requires_grad=True)
    labels = torch.tensor([[1, IGNORE_INDEX, 3, 4]])
    grad_weights = torch.tensor([[1.0, 0.75, 0.5, 0.25]])

    actual = chunked_lm_head_logprobs(
        hidden, weight, labels, token_chunk_size=2, vocab_chunk_size=3, fp32_lm_head=True
    )
    (actual * grad_weights).sum().backward()

    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    expected = _dense_logprobs(hidden_ref, weight_ref, _substituted(labels))
    (expected * grad_weights).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(hidden.grad, hidden_ref.grad, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=RTOL, atol=ATOL)


def test_ignored_target_never_reports_a_positive_log_probability():
    """No position may score above zero, whatever the labels.

    An ungathered target logit is reported as ``0 - logsumexp(logits)``, so it exceeds zero exactly when the
    partition function is below one -- here every logit is near ``-9``, which puts ``logsumexp`` near ``-7.2``
    and the ignored position's reported score near ``+7.2``. The bound is the definition of a log-probability
    rather than a tolerance, so it holds for any vocabulary, any tiling, and any hidden state.
    """
    torch.manual_seed(17)
    hidden = 1.0 + 0.1 * torch.randn(1, 3, HIDDEN_SIZE)
    weight = -3.0 + 0.1 * torch.randn(VOCAB_SIZE, HIDDEN_SIZE)
    labels = torch.tensor([[2, IGNORE_INDEX, 4]])

    actual = chunked_lm_head_logprobs(
        hidden, weight, labels, token_chunk_size=2, vocab_chunk_size=3, fp32_lm_head=True
    )

    assert torch.isfinite(actual).all()
    assert float(actual.max()) <= 0.0, f"reported a log-probability above zero: {actual.tolist()}"


@pytest.mark.parametrize("bad_label", [-1, -99, -101, VOCAB_SIZE, VOCAB_SIZE + 1])
def test_labels_outside_the_vocabulary_are_rejected(bad_label):
    """A label that is neither a vocabulary index nor the sentinel is a caller error, not a score.

    Such a label falls in no vocab tile for the same reason the sentinel does, so without a domain check it is
    reported through the identical silent path. Both sides of the tile-membership comparison are covered, and
    both neighbours of the sentinel, since a check written as ``labels < IGNORE_INDEX`` or as equality against
    the wrong sign would let one of them through.
    """
    hidden = torch.randn(1, 4, HIDDEN_SIZE)
    weight = torch.randn(VOCAB_SIZE, HIDDEN_SIZE)
    labels = torch.tensor([[1, bad_label, 3, 4]])

    with pytest.raises(ValueError, match="outside the vocabulary"):
        chunked_lm_head_logprobs(hidden, weight, labels, token_chunk_size=2, vocab_chunk_size=3, fp32_lm_head=True)


def test_a_constraint_at_an_ignored_position_is_dropped_and_the_others_are_kept():
    """Scoring the sentinel at a substitute entry must not put that entry under a constraint.

    An allow-set that excludes the substitute drives its logit to ``-inf``, which both makes the reported score
    meaningless and lets ``validate_action_mask_targets`` reject a position the caller asked to ignore. A
    constraint at a supervised position is present too, so a head that discarded every constraint would fail
    the same assertion.
    """
    torch.manual_seed(5)
    hidden = torch.randn(1, 4, HIDDEN_SIZE)
    weight = torch.randn(VOCAB_SIZE, HIDDEN_SIZE)
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

    actual = chunked_lm_head_logprobs(
        hidden,
        weight,
        labels,
        token_chunk_size=2,
        vocab_chunk_size=3,
        fp32_lm_head=True,
        action_masks=action_masks,
    )

    logits = torch.einsum("bsh,vh->bsv", hidden.float(), weight.float())
    # The surviving constraint at source position 2 alone; source position 1 is scored unconstrained.
    logits[0, 2, [token for token in range(VOCAB_SIZE) if token not in (3, 5)]] = float("-inf")
    expected = torch.log_softmax(logits, dim=-1)
    expected = expected.gather(-1, _substituted(labels).unsqueeze(-1)).squeeze(-1)

    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_dropping_every_constraint_leaves_an_unconstrained_scoring_pass():
    """When every constrained source position is ignored, the head scores as if no mask had been supplied.

    Filtering can empty the constraint set, and an empty set has to fall through the ``has_constraints``
    guards in ``apply_lm_head_action_masks_`` and ``validate_action_mask_targets`` rather than mask a row of
    logits or index into an empty position tensor.
    """
    torch.manual_seed(5)
    hidden = torch.randn(1, 4, HIDDEN_SIZE)
    weight = torch.randn(VOCAB_SIZE, HIDDEN_SIZE)
    labels = torch.tensor([[1, IGNORE_INDEX, IGNORE_INDEX, 4]])
    # Target positions 2 and 3 constrain source positions 1 and 2, both of which carry the sentinel.
    action_masks = {
        "seq_len": 4,
        "vocab_size": VOCAB_SIZE,
        "positions": [2, 3],
        "set_indices": [0, 0],
        "set_modes_allow": [True],
        "set_offsets": [0, 1],
        "token_ids": [1],
    }

    actual = chunked_lm_head_logprobs(
        hidden,
        weight,
        labels,
        token_chunk_size=2,
        vocab_chunk_size=3,
        fp32_lm_head=True,
        action_masks=action_masks,
    )

    expected = _dense_logprobs(hidden, weight, _substituted(labels))

    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)


def test_a_window_whose_targets_are_all_ignored_is_scored_at_the_substitute():
    """A window with no supervised position is a real state for a padded or shard-tail batch.

    It exercises the branch where the substitution applies to every position at once, and where a head that
    only special-cased a mixed window would still report ``-logsumexp`` throughout.
    """
    torch.manual_seed(7)
    hidden = torch.randn(1, 3, HIDDEN_SIZE, requires_grad=True)
    weight = torch.randn(VOCAB_SIZE, HIDDEN_SIZE, requires_grad=True)
    labels = torch.full((1, 3), IGNORE_INDEX)

    actual = chunked_lm_head_logprobs(
        hidden, weight, labels, token_chunk_size=2, vocab_chunk_size=3, fp32_lm_head=True
    )
    # A caller drops ignored positions from the loss, so the gradient reaching the head here is exactly zero.
    (actual * torch.zeros_like(actual)).sum().backward()

    expected = _dense_logprobs(hidden.detach(), weight.detach(), _substituted(labels))

    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(hidden.grad, torch.zeros_like(hidden), rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(weight.grad, torch.zeros_like(weight), rtol=RTOL, atol=ATOL)


def test_labels_all_inside_the_vocabulary_reproduce_the_dense_log_softmax():
    """No-regression: with nothing to substitute, value and gradient are unchanged.

    Ragged tiles on both axes -- ``token_chunk_size=3`` over 5 positions, ``vocab_chunk_size=4`` over 7 entries
    -- and per-position temperature, because the domain check runs on the flattened labels before the tile loop
    and must not disturb its bounds.
    """
    torch.manual_seed(11)
    hidden = torch.randn(1, 5, 4, requires_grad=True)
    weight = torch.randn(7, 4, requires_grad=True)
    labels = torch.tensor([[0, 1, 2, 3, 4]])
    temperature = torch.tensor([[0.7, 1.1, 0.9, 1.3, 0.8]], dtype=torch.float32)
    grad_weights = torch.linspace(0.1, 1.0, steps=5).reshape(1, 5)

    actual = chunked_lm_head_logprobs(
        hidden,
        weight,
        labels,
        temperature=temperature,
        token_chunk_size=3,
        vocab_chunk_size=4,
        fp32_lm_head=True,
    )
    (actual * grad_weights).sum().backward()

    hidden_ref = hidden.detach().clone().requires_grad_(True)
    weight_ref = weight.detach().clone().requires_grad_(True)
    expected = _dense_logprobs(hidden_ref, weight_ref, labels, temperature=temperature)
    (expected * grad_weights).sum().backward()

    torch.testing.assert_close(actual, expected, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(hidden.grad, hidden_ref.grad, rtol=RTOL, atol=ATOL)
    torch.testing.assert_close(weight.grad, weight_ref.grad, rtol=RTOL, atol=ATOL)
