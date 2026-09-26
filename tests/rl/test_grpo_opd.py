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

"""Sampled-token on-policy distillation and GRPO DP-normalization regressions."""

from __future__ import annotations

import math

import pytest
import torch

from arctic_platform.rl.processors import resolve_loss

_TAU = 0.5
_CLIP = 1.0


def _grpo(
    logprobs: torch.Tensor,
    *,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    teacher: torch.Tensor | None = None,
    sequence_loss_weights: torch.Tensor | None = None,
    config: dict | None = None,
) -> tuple[torch.Tensor, dict]:
    batch = {
        "input_ids": torch.zeros_like(logprobs, dtype=torch.long),
        "old_log_probs_shifted": logprobs.detach().clone(),
        "advantages": advantages,
        "loss_mask": loss_mask,
    }
    if teacher is not None:
        batch["teacher_log_probs_shifted"] = teacher
    if sequence_loss_weights is not None:
        batch["sequence_loss_weights"] = sequence_loss_weights
    return resolve_loss("grpo").loss(
        {"logprobs": logprobs},
        batch,
        {},
        {
            "use_cispo_loss": True,
            "is_weight_clip_max": 5.0,
            **(config or {}),
        },
        "cpu",
    )


@pytest.mark.parametrize("teacher_clip_negative", [None, 0.0])
def test_sampled_teacher_matches_preblended_advantages_value_gradient_and_metrics(teacher_clip_negative):
    student_values = torch.tensor([[-1.2, -0.8, -1.5, -0.4, -0.9]])
    deltas = torch.tensor([[0.4, 3.0, math.nan, 1.0e6, -2.5]])
    mask = torch.tensor([[1, 1, 1, 0, 1]], dtype=torch.bool)
    advantages = torch.tensor([[0.7, -0.4, 0.2, 9.0, 0.5]])
    low = -(_CLIP if teacher_clip_negative is None else teacher_clip_negative)
    scored = mask & torch.isfinite(deltas)
    teacher_term = torch.where(scored, deltas.clamp(low, _CLIP), torch.zeros_like(deltas))
    blended = advantages + _TAU * teacher_term

    expected_student = student_values.clone().requires_grad_()
    expected_loss, _ = _grpo(
        expected_student,
        advantages=blended,
        loss_mask=mask,
    )
    expected_grad = torch.autograd.grad(expected_loss, expected_student)[0]

    actual_student = student_values.clone().requires_grad_()
    teacher = (student_values + deltas).requires_grad_()
    teacher_config = {
        "teacher_tau": _TAU,
        "teacher_clip": _CLIP,
    }
    if teacher_clip_negative is not None:
        teacher_config["teacher_clip_negative"] = teacher_clip_negative
    actual_loss, metrics = _grpo(
        actual_student,
        advantages=advantages,
        loss_mask=mask,
        teacher=teacher,
        config=teacher_config,
    )
    actual_grad = torch.autograd.grad(actual_loss, actual_student)[0]

    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=1e-7)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=1e-7)
    assert teacher.grad is None
    assert metrics["teacher_tau"] == _TAU
    assert metrics["teacher_term_token_count"] == 3.0
    assert metrics["teacher_log_ratio_sum"] == pytest.approx(0.4 + 3.0 - 2.5, rel=0, abs=1e-6)
    assert metrics["teacher_clipped_log_ratio_sum"] == pytest.approx(
        float(teacher_term.sum()),
        rel=0,
        abs=1e-6,
    )


def test_sampled_teacher_tau_zero_is_exactly_legacy_grpo():
    values = torch.tensor([[-1.0, -0.5, -1.5]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    advantages = torch.tensor([[0.2, -0.4, 9.0]])
    teacher = torch.tensor([[-0.1, -3.0, math.nan]])

    baseline_student = values.clone().requires_grad_()
    baseline_loss, baseline_metrics = _grpo(
        baseline_student,
        advantages=advantages,
        loss_mask=mask,
    )
    baseline_grad = torch.autograd.grad(baseline_loss, baseline_student)[0]

    zero_student = values.clone().requires_grad_()
    zero_loss, zero_metrics = _grpo(
        zero_student,
        advantages=advantages,
        loss_mask=mask,
        teacher=teacher,
        config={"teacher_tau": 0.0, "teacher_clip": _CLIP},
    )
    zero_grad = torch.autograd.grad(zero_loss, zero_student)[0]

    assert zero_loss.item() == baseline_loss.item()
    assert zero_metrics == baseline_metrics
    assert not any(key.startswith("teacher") for key in zero_metrics)
    assert torch.equal(zero_grad, baseline_grad)


@pytest.mark.parametrize(
    ("context_extra", "config_extra", "match"),
    [
        ({}, {"teacher_tau": 0.5, "teacher_clip": 1.0}, "teacher_log_probs_shifted"),
        (
            {"teacher_log_probs_shifted": torch.zeros(1, 2)},
            {"teacher_tau": True, "teacher_clip": 1.0},
            "teacher_tau",
        ),
        (
            {"teacher_log_probs_shifted": torch.zeros(1, 2)},
            {"teacher_tau": -0.5, "teacher_clip": 1.0},
            "teacher_tau",
        ),
        (
            {"teacher_log_probs_shifted": torch.zeros(1, 2)},
            {"teacher_tau": 0.5},
            "teacher_clip",
        ),
        (
            {"teacher_log_probs_shifted": torch.zeros(1, 2)},
            {"teacher_tau": 0.5, "teacher_clip": math.inf},
            "teacher_clip",
        ),
        (
            {"teacher_log_probs_shifted": torch.zeros(1, 2)},
            {"teacher_tau": 0.5, "teacher_clip": 1.0, "teacher_clip_negative": -0.1},
            "teacher_clip_negative",
        ),
        (
            {"teacher_log_probs_shifted": torch.zeros(1, 2)},
            {"teacher_tau": 0.5, "teacher_clip": 1.0, "importance_sampling_level": "sequence"},
            "importance_sampling_level",
        ),
        (
            {
                "teacher_log_probs_shifted": torch.zeros(1, 2),
                "sequence_loss_weights": torch.ones(1),
            },
            {"teacher_tau": 0.5, "teacher_clip": 1.0},
            "prompt-mean",
        ),
    ],
)
def test_sampled_teacher_invalid_requests_fail_during_validation(context_extra, config_extra, match):
    context = {
        "input_ids": torch.zeros(1, 2, dtype=torch.long),
        "loss_mask": torch.ones(1, 2, dtype=torch.bool),
        **context_extra,
    }
    with pytest.raises(ValueError, match=match):
        resolve_loss("grpo").validation_callback(context, config_extra)


def test_weighted_prompt_mean_grpo_dp2_matches_dp1_loss_and_gradient():
    values = torch.tensor([[-1.2, -0.8, -0.4], [-0.7, -1.1, -1.5]])
    advantages = torch.tensor([[0.5, -0.2, 0.0], [-0.3, 0.7, 0.2]])
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool)
    weights = torch.tensor([0.25, 0.75])

    whole = values.clone().requires_grad_()
    expected_loss, _ = _grpo(
        whole,
        advantages=advantages,
        loss_mask=mask,
        sequence_loss_weights=weights,
        config={"loss_agg_mode": "prompt-mean", "dp_size": 1},
    )
    expected_grad = torch.autograd.grad(expected_loss, whole)[0]

    local_losses = []
    local_grads = []
    for row in range(2):
        local = values[row : row + 1].clone().requires_grad_()
        local_loss, _ = _grpo(
            local,
            advantages=advantages[row : row + 1],
            loss_mask=mask[row : row + 1],
            sequence_loss_weights=weights[row : row + 1],
            config={"loss_agg_mode": "prompt-mean", "dp_size": 2},
        )
        local_losses.append(local_loss.detach())
        local_grads.append(torch.autograd.grad(local_loss, local)[0])

    torch.testing.assert_close(sum(local_losses) / 2, expected_loss, rtol=0, atol=1e-7)
    torch.testing.assert_close(torch.cat(local_grads) / 2, expected_grad, rtol=0, atol=1e-7)
