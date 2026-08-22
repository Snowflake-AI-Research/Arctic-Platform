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

from arctic_platform.rl.processors import LOSS_FNS
from arctic_platform.rl.processors.on_policy_distill import on_policy_distill_loss


def _loss(student, teacher, mask):
    return on_policy_distill_loss(
        {"logprobs": student},
        {"teacher_log_probs_shifted": teacher, "loss_mask": mask},
        {},
        {"distill_estimator": "low_var_kl", "loss_agg_mode": "token-mean"},
        "cpu",
    )


def test_equal_logprobs_have_zero_loss():
    logprobs = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    loss, metrics = _loss(logprobs, logprobs.detach().clone(), torch.ones_like(logprobs, dtype=torch.bool))
    assert loss.item() == 0.0
    assert metrics["distill_kl_count"] == 2


def test_loss_is_registered():
    assert LOSS_FNS["on_policy_distill"] is on_policy_distill_loss


def test_masked_values_do_not_change_loss():
    student = torch.tensor([[-1.0, -2.0, -400.0]], requires_grad=True)
    teacher = torch.tensor([[-1.1, -1.8, 99.0]])
    mask = torch.tensor([[True, True, False]])
    loss_a, _ = _loss(student, teacher, mask)
    teacher[-1, -1] = -999.0
    student_b = student.detach().clone()
    student_b[-1, -1] = 123.0
    student_b.requires_grad_(True)
    loss_b, _ = _loss(student_b, teacher, mask)
    torch.testing.assert_close(loss_a, loss_b)


def test_teacher_is_detached_and_student_receives_gradient():
    student = torch.tensor([[-1.0]], requires_grad=True)
    teacher = torch.tensor([[-2.0]], requires_grad=True)
    loss, _ = _loss(student, teacher, torch.tensor([[True]]))
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_kl_coefficient_scales_loss():
    student = torch.tensor([[-1.0]], requires_grad=True)
    teacher = torch.tensor([[-2.0]])
    mask = torch.tensor([[True]])
    base, _ = _loss(student, teacher, mask)
    doubled, _ = on_policy_distill_loss(
        {"logprobs": student},
        {"teacher_log_probs_shifted": teacher, "loss_mask": mask},
        {},
        {"distill_estimator": "low_var_kl", "kl_coef": 2.0},
        "cpu",
    )
    torch.testing.assert_close(doubled, 2 * base)


def test_shape_mismatch_and_empty_mask_fail():
    with pytest.raises(ValueError, match="identical shapes"):
        _loss(torch.zeros(1, 2), torch.zeros(1, 1), torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="empty"):
        _loss(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))
