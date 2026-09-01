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

import math

import pytest
import torch

from arctic_platform.rl.processors import LOSS_FNS
from arctic_platform.rl.processors.on_policy_distill import _DEFAULT_DELTA_CLAMP
from arctic_platform.rl.processors.on_policy_distill import _distill_kl_per_token
from arctic_platform.rl.processors.on_policy_distill import on_policy_distill_loss


def _make_call(
    *,
    student: torch.Tensor,
    teacher: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
    config: dict | None = None,
    extra_batch: dict | None = None,
):
    if loss_mask is None:
        loss_mask = torch.ones_like(student, dtype=torch.bool)
    batch = {
        "teacher_log_probs_shifted": teacher,
        "loss_mask": loss_mask,
    }
    if extra_batch:
        batch.update(extra_batch)
    return {"logprobs": student}, batch, {}, dict(config or {})


def _run(**kwargs):
    model_outputs, batch, meta, config = _make_call(**kwargs)
    return on_policy_distill_loss(model_outputs, batch, meta, config, "cpu")


def _loss(student, teacher, mask):
    return _run(
        student=student,
        teacher=teacher,
        loss_mask=mask,
        config={"distill_estimator": "low_var_kl", "loss_agg_mode": "token-mean"},
    )


def test_equal_logprobs_have_zero_loss():
    logprobs = torch.tensor([[-1.0, -2.0]], requires_grad=True)
    loss, metrics = _loss(logprobs, logprobs.detach().clone(), torch.ones_like(logprobs, dtype=torch.bool))
    assert loss.item() == 0.0
    assert metrics["distill_kl_count"] == 2.0
    assert metrics["distill_k1"] == 0.0
    assert metrics["distill_kl_max"] == 0.0


def test_k1_metric_is_masked_mean_logprob_gap():
    student = torch.tensor([[-1.0, -2.0, -9.0]], requires_grad=True)
    teacher = torch.tensor([[-1.5, -1.0, 3.0]])
    mask = torch.tensor([[True, True, False]])
    _, metrics = _loss(student, teacher, mask)
    expected = ((-1.0 - -1.5) + (-2.0 - -1.0)) / 2
    assert metrics["distill_k1"] == pytest.approx(expected)


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
    doubled, metrics = on_policy_distill_loss(
        {"logprobs": student},
        {"teacher_log_probs_shifted": teacher, "loss_mask": mask},
        {},
        {"distill_estimator": "low_var_kl", "kl_coef": 2.0},
        "cpu",
    )
    torch.testing.assert_close(doubled, 2 * base)
    assert metrics["distill_kl_coef"] == 2.0


def test_shape_mismatch_and_empty_mask_fail():
    with pytest.raises(ValueError, match="identical shapes"):
        _loss(torch.zeros(1, 2), torch.zeros(1, 1), torch.ones(1, 2, dtype=torch.bool))
    with pytest.raises(ValueError, match="empty"):
        _loss(torch.zeros(1, 2), torch.zeros(1, 2), torch.zeros(1, 2, dtype=torch.bool))


def test_k3_is_non_negative_over_a_wide_delta_range():
    delta = torch.linspace(-15.0, 15.0, 601)
    student = torch.zeros_like(delta)
    per_token, _ = _distill_kl_per_token(
        student, delta, estimator="low_var_kl", delta_clamp=_DEFAULT_DELTA_CLAMP
    )
    assert bool((per_token >= 0.0).all())


def test_k3_matches_closed_form():
    student = torch.tensor([[-2.0, -0.5]])
    teacher = torch.tensor([[-1.0, -3.0]])
    per_token, delta = _distill_kl_per_token(
        student, teacher, estimator="low_var_kl", delta_clamp=_DEFAULT_DELTA_CLAMP
    )
    for got, d in zip(per_token.flatten().tolist(), delta.flatten().tolist()):
        assert got == pytest.approx(math.exp(d) - d - 1.0, rel=1e-6)


@pytest.mark.parametrize(
    "teacher_lp, student_lp, expect_student_logprob_rises",
    [
        (-0.5, -2.0, True),
        (-3.0, -1.0, False),
    ],
)
def test_gradient_moves_student_toward_teacher(teacher_lp, student_lp, expect_student_logprob_rises):
    student = torch.full((1, 3), student_lp, requires_grad=True)
    teacher = torch.full((1, 3), teacher_lp)
    loss, _ = _run(student=student, teacher=teacher)
    loss.backward()
    grad = student.grad
    assert grad is not None
    if expect_student_logprob_rises:
        assert bool((grad < 0).all()), grad
    else:
        assert bool((grad > 0).all()), grad
    delta = teacher_lp - student_lp
    expected = (1.0 - math.exp(delta)) / student.numel()
    assert grad.flatten()[0].item() == pytest.approx(expected, rel=1e-5)


def test_output_clamp_is_off_by_default_so_gradients_survive_large_delta():
    student = torch.full((1, 2), -8.0, requires_grad=True)
    teacher = torch.full((1, 2), -1.0)
    loss, metrics = _run(student=student, teacher=teacher)
    assert loss.item() > 10.0, "default path must not cap the objective at 10"
    loss.backward()
    assert bool((student.grad.abs() > 1.0).all()), student.grad
    assert metrics["distill_kl_output_clamped_count"] == 0.0


def test_output_clamp_when_explicitly_requested():
    student = torch.full((1, 2), -8.0, requires_grad=True)
    teacher = torch.full((1, 2), -1.0)
    loss, metrics = _run(student=student, teacher=teacher, config={"kl_clamp_max": 10.0})
    assert loss.item() == pytest.approx(10.0, rel=1e-6)
    assert metrics["distill_kl_output_clamped_count"] == 2.0
    loss.backward()
    assert bool((student.grad == 0).all())


def test_k1_is_rejected_as_an_objective():
    student = torch.full((1, 4), -2.0, requires_grad=True)
    teacher = torch.full((1, 4), -1.0)
    with pytest.raises(ValueError, match="value.*estimator, not a trainable objective"):
        _run(student=student, teacher=teacher, config={"distill_estimator": "k1"})


def test_unknown_estimator_is_rejected():
    student = torch.full((1, 4), -2.0)
    teacher = torch.full((1, 4), -1.0)
    with pytest.raises(ValueError, match="Unknown distill_estimator"):
        _run(student=student, teacher=teacher, config={"distill_estimator": "reverse_kl_full"})


def test_unknown_config_key_is_rejected():
    student = torch.full((1, 4), -2.0)
    teacher = torch.full((1, 4), -1.0)
    with pytest.raises(ValueError, match="Unknown config keys"):
        _run(student=student, teacher=teacher, config={"kl_ceof": 0.5})


@pytest.mark.parametrize("estimator", ["low_var_kl", "k3", "mse", "k2", "abs"])
def test_trainable_estimators_have_correct_gradient_direction(estimator):
    student = torch.full((1, 4), -3.0, requires_grad=True)
    teacher = torch.full((1, 4), -1.0)
    loss, _ = _run(student=student, teacher=teacher, config={"distill_estimator": estimator})
    loss.backward()
    assert bool((student.grad < 0).all()), (estimator, student.grad)


def test_packed_layout_matches_padded_layout():
    student_rows = [[-1.0, -2.0, -1.5], [-2.5, -0.5]]
    teacher_rows = [[-1.5, -1.0, -2.0], [-1.0, -1.5]]
    flat_student = torch.tensor([[v for row in student_rows for v in row]])
    flat_teacher = torch.tensor([[v for row in teacher_rows for v in row]])
    flat_mask = torch.ones_like(flat_student, dtype=torch.bool)
    cu = torch.tensor([0, 3, 5], dtype=torch.int32)
    packed_loss, packed_metrics = _run(
        student=flat_student,
        teacher=flat_teacher,
        loss_mask=flat_mask,
        extra_batch={"cu_seqlens": cu},
    )
    padded_loss, padded_metrics = _run(student=flat_student, teacher=flat_teacher)
    assert packed_loss.item() == pytest.approx(padded_loss.item(), rel=1e-9)
    assert packed_metrics["distill_kl_count"] == padded_metrics["distill_kl_count"] == 5.0


def test_delta_clamp_keeps_loss_finite_on_absurd_input():
    student = torch.full((1, 4), -400.0)
    teacher = torch.full((1, 4), -0.001)
    loss, metrics = _run(student=student, teacher=teacher)
    assert torch.isfinite(loss), loss
    assert metrics["distill_delta_clamped_count"] == 4.0


def test_missing_teacher_logprobs_raises_a_useful_error():
    student = torch.full((1, 4), -2.0)
    with pytest.raises(ValueError, match="teacher_log_probs_shifted"):
        on_policy_distill_loss(
            {"logprobs": student},
            {"loss_mask": torch.ones_like(student, dtype=torch.bool)},
            {},
            {},
            "cpu",
        )


def test_k1_metric_is_signed_and_detects_bias_direction():
    student = torch.full((1, 4), -3.0)
    teacher = torch.full((1, 4), -1.0)
    _, under = _run(student=student, teacher=teacher)
    _, over = _run(student=teacher, teacher=student)
    assert under["distill_k1_sum"] < 0.0
    assert over["distill_k1_sum"] > 0.0
    assert under["distill_kl_sum"] > 0.0 and over["distill_kl_sum"] > 0.0


def test_sampler_train_kl_is_zero_when_sampler_matches_trainer():
    student = torch.full((1, 4), -2.0)
    teacher = torch.full((1, 4), -1.0)
    _, metrics = _run(
        student=student,
        teacher=teacher,
        extra_batch={"old_log_probs_shifted": student.clone()},
    )
    assert metrics["sampler_train_kl_sum"] == 0.0
    assert metrics["sampler_train_abs_delta_max"] == 0.0


def test_sampler_train_kl_is_positive_on_disagreement():
    student = torch.full((1, 4), -2.0)
    teacher = torch.full((1, 4), -1.0)
    _, metrics = _run(
        student=student,
        teacher=teacher,
        extra_batch={"old_log_probs_shifted": torch.full((1, 4), -2.5)},
    )
    assert metrics["sampler_train_kl_sum"] > 0.0
    assert metrics["sampler_train_abs_delta_max"] == pytest.approx(0.5, rel=1e-6)


def test_sampler_metrics_absent_when_sampler_logprobs_absent():
    student = torch.full((1, 4), -2.0)
    teacher = torch.full((1, 4), -1.0)
    _, metrics = _run(student=student, teacher=teacher)
    assert "sampler_train_kl_sum" not in metrics


def test_all_metrics_are_plain_floats():
    student = torch.full((1, 4), -2.0, requires_grad=True)
    teacher = torch.full((1, 4), -1.0)
    _, metrics = _run(student=student, teacher=teacher)
    for key, value in metrics.items():
        assert isinstance(value, float), (key, type(value))
        assert math.isfinite(value), (key, value)


def test_logits_fallback_when_no_logprobs_post_processor():
    torch.manual_seed(0)
    B, S, V = 1, 5, 7
    logits = torch.randn(B, S, V, requires_grad=True)
    input_ids = torch.randint(0, V, (B, S))
    expected = (
        torch.log_softmax(logits.float(), dim=-1)
        .gather(-1, torch.roll(input_ids, -1, dims=-1).unsqueeze(-1))
        .squeeze(-1)
    )
    loss, _ = on_policy_distill_loss(
        {"logits": logits},
        {
            "input_ids": input_ids,
            "teacher_log_probs_shifted": expected.detach(),
            "loss_mask": torch.ones(B, S, dtype=torch.bool),
        },
        {},
        {},
        "cpu",
    )
    assert loss.item() == pytest.approx(0.0, abs=1e-6)


def test_count_opd_loss_tokens_dict_and_gas_list():
    from arctic_platform.rl.processors.on_policy_distill import count_opd_loss_tokens

    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
    assert count_opd_loss_tokens({"loss_mask": mask}) == (3, 2)
    assert count_opd_loss_tokens([{"loss_mask": mask}, {"loss_mask": mask}]) == (6, 4)
    assert count_opd_loss_tokens({"input_ids": torch.zeros(1, 2)}) == (0, 0)


def test_count_opd_loss_tokens_packed_cu_seqlens():
    from arctic_platform.rl.processors.on_policy_distill import count_opd_loss_tokens

    mask = torch.ones(1, 5, dtype=torch.long)
    cu = torch.tensor([0, 3, 5], dtype=torch.int32)
    assert count_opd_loss_tokens({"loss_mask": mask, "cu_seqlens": cu}) == (5, 2)


def test_global_token_norm_downweights_short_microbatch():
    student_short = torch.full((1, 1), -2.0, requires_grad=True)
    teacher_short = torch.full((1, 1), -1.0)
    student_long = torch.full((1, 9), -2.0, requires_grad=True)
    teacher_long = torch.full((1, 9), -1.0)
    cfg = {"distill_estimator": "low_var_kl", "loss_agg_mode": "token-mean", "dp_size": 8, "batch_num_tokens": 10}
    loss_short, short_metrics = _run(
        student=student_short, teacher=teacher_short, config=cfg, loss_mask=torch.ones(1, 1, dtype=torch.bool)
    )
    loss_long, long_metrics = _run(
        student=student_long, teacher=teacher_long, config=cfg, loss_mask=torch.ones(1, 9, dtype=torch.bool)
    )
    local_short, _ = _run(student=student_short, teacher=teacher_short, loss_mask=torch.ones(1, 1, dtype=torch.bool))
    assert local_short.item() == pytest.approx(loss_short.item() * 10 / 8, rel=1e-5)
    assert loss_long.item() == pytest.approx(loss_short.item() * 9, rel=1e-5)
    assert short_metrics["distill_dp_size"] == 8.0
    assert short_metrics["distill_batch_num_tokens"] == 10.0
    assert long_metrics["distill_batch_num_tokens"] == 10.0
    assert "loss.sum" in short_metrics
    assert "loss.tokens" in short_metrics
    assert "distill_kl.sum" in short_metrics
    assert "distill_kl.tokens" in short_metrics
    assert short_metrics["distill_kl.tokens"] == 1.0
    assert long_metrics["distill_kl.tokens"] == 9.0


def test_meta_supplies_norm_when_config_omits_it():
    student = torch.full((1, 2), -2.0, requires_grad=True)
    teacher = torch.full((1, 2), -1.0)
    model_outputs, batch, _, config = _make_call(student=student, teacher=teacher)
    via_config, _ = on_policy_distill_loss(
        model_outputs, batch, {}, {**config, "dp_size": 4, "batch_num_tokens": 20}, "cpu"
    )
    via_meta, _ = on_policy_distill_loss(
        model_outputs, batch, {"dp_size": 4, "batch_num_tokens": 20}, config, "cpu"
    )
    assert via_meta.item() == pytest.approx(via_config.item(), rel=1e-9)


def test_apply_opd_global_token_config_writes_config_and_meta():
    from arctic_platform.rl.processors.on_policy_distill import apply_opd_global_token_config
    from arctic_platform.rl.processors.on_policy_distill import count_opd_loss_tokens

    mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]])
    tokens, seqs = count_opd_loss_tokens({"loss_mask": mask})
    processing = {"config": {"distill_estimator": "low_var_kl"}}
    meta: dict = {}
    apply_opd_global_token_config(
        processing, meta, dp_size=8, batch_num_tokens=tokens, global_batch_size=seqs
    )
    assert processing["config"]["dp_size"] == 8
    assert processing["config"]["batch_num_tokens"] == 4
    assert processing["config"]["global_batch_size"] == 2
    assert meta["batch_num_tokens"] == 4
    assert meta["dp_size"] == 8
