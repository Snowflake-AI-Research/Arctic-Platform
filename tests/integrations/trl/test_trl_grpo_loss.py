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

"""CPU tests for the server-side ``trl_grpo`` loss math + normalization."""

from __future__ import annotations

import pytest
import torch

trl_loss = pytest.importorskip("arctic_platform.integrations.trl.loss")
trl_grpo = trl_loss.trl_grpo


def _ref_trl_loss(logprobs, old, adv, mask, eps_low, eps_high, batch_num_tokens, dp_size, grad_accum):
    """Direct reimpl of TRL's compute_loss.loss_fn math, plus the server ``* dp_size``."""
    ratio = torch.exp(logprobs - old)
    per_token = -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * adv)
    loss = (per_token * mask).sum() / batch_num_tokens
    return loss / grad_accum * dp_size


def _call(logprobs, old, adv, mask, *, eps_low=0.2, eps_high=0.2, batch_num_tokens=None, dp_size=1, grad_accum=1):
    if batch_num_tokens is None:
        batch_num_tokens = int(mask.sum().item())
    meta = {
        "epsilon_low": eps_low,
        "epsilon_high": eps_high,
        "batch_num_tokens": batch_num_tokens,
        "dp_size": dp_size,
        "grad_accum_steps": grad_accum,
    }
    return trl_grpo(
        {"logprobs": logprobs},
        {"old_log_probs": old, "advantages": adv, "loss_mask": mask},
        meta,
        {},
        "cpu",
    )


def test_matches_trl_math_dp1_gas1():
    torch.manual_seed(0)
    b, s = 3, 5
    logprobs = torch.randn(b, s)
    old = torch.randn(b, s)
    adv = torch.randn(b, s)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], dtype=torch.float32)
    tokens = int(mask.sum().item())

    loss, metrics = _call(logprobs, old, adv, mask, batch_num_tokens=tokens)
    ref = _ref_trl_loss(logprobs, old, adv, mask, 0.2, 0.2, tokens, 1, 1)

    assert loss.item() == pytest.approx(ref.item(), abs=1e-6)
    assert metrics["loss"].item() == pytest.approx(ref.item(), abs=1e-6)


def test_masked_positions_are_ignored():
    # Poison a masked slot; the loss must be unchanged (mask zeroes it, incl. NaN).
    logprobs = torch.zeros(1, 4)
    old = torch.zeros(1, 4)
    adv = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    base, _ = _call(logprobs, old, adv, mask, batch_num_tokens=int(mask.sum()))
    adv_poison = adv.clone()
    adv_poison[0, 2] = float("nan")
    poisoned, _ = _call(logprobs, old, adv_poison, mask, batch_num_tokens=int(mask.sum()))

    assert torch.isfinite(poisoned)
    assert poisoned.item() == pytest.approx(base.item(), abs=1e-6)


def test_normalization_scales_with_dp_gas_and_tokens():
    torch.manual_seed(1)
    b, s = 2, 4
    logprobs = torch.randn(b, s)
    old = torch.randn(b, s)
    adv = torch.randn(b, s)
    mask = torch.ones(b, s)
    tokens = int(mask.sum().item())

    base, _ = _call(logprobs, old, adv, mask, batch_num_tokens=tokens, dp_size=1, grad_accum=1)

    dp4, _ = _call(logprobs, old, adv, mask, batch_num_tokens=tokens, dp_size=4, grad_accum=1)
    assert dp4.item() == pytest.approx(base.item() * 4, abs=1e-6)

    gas2, _ = _call(logprobs, old, adv, mask, batch_num_tokens=tokens, dp_size=1, grad_accum=2)
    assert gas2.item() == pytest.approx(base.item() / 2, abs=1e-6)

    tok2, _ = _call(logprobs, old, adv, mask, batch_num_tokens=tokens * 2, dp_size=1, grad_accum=1)
    assert tok2.item() == pytest.approx(base.item() / 2, abs=1e-6)


def test_epsilon_high_distinct_from_low_is_honored():
    # ratio = 3 (>1+eps_high), adv > 0 -> the high clamp binds, so loss depends on eps_high, not eps_low.
    logprobs = torch.tensor([[torch.log(torch.tensor(3.0))]])
    old = torch.zeros(1, 1)
    adv = torch.tensor([[2.0]])
    mask = torch.ones(1, 1)

    loss_hi5, _ = _call(logprobs, old, adv, mask, eps_low=0.1, eps_high=0.5, batch_num_tokens=1)
    loss_hi2, _ = _call(logprobs, old, adv, mask, eps_low=0.1, eps_high=0.2, batch_num_tokens=1)

    # per_token = -min(ratio*adv, (1+eps_high)*adv) = -(1+eps_high)*adv (clamped).
    assert loss_hi5.item() == pytest.approx(-(1 + 0.5) * 2.0, abs=1e-6)
    assert loss_hi2.item() == pytest.approx(-(1 + 0.2) * 2.0, abs=1e-6)

    # eps_low must not affect this positive-advantage, high-clamped case.
    loss_lo_changed, _ = _call(logprobs, old, adv, mask, eps_low=0.4, eps_high=0.5, batch_num_tokens=1)
    assert loss_lo_changed.item() == pytest.approx(loss_hi5.item(), abs=1e-6)
