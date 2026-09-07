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
# See the License for the specific language governing terms and
# limitations under the License.

"""Bit-level JSD gate: Arctic generalized_jsd vs TRL's sparse async formula.

TRL's trainer module pulls Accelerate/transformers, so the comparison copies
``_add_tail_bucket`` / ``_jsd_divergence`` from
``trl.experimental.async_distillation.async_distillation_trainer`` (beta=0
path: full top-k + tail).
"""

from __future__ import annotations

import torch

from arctic_platform.integrations.trl_distill.jsd import generalized_jsd


def _trl_add_tail_bucket(log_probs, valid_mask):
    log_sum = torch.logsumexp(torch.where(valid_mask, log_probs, torch.full_like(log_probs, float("-inf"))), dim=-1)
    log_sum = torch.clamp(log_sum, max=-1e-7)
    tail = torch.log(-torch.expm1(log_sum)).unsqueeze(-1)
    tail_mask = torch.ones_like(valid_mask[..., :1], dtype=torch.bool)
    return torch.cat([log_probs, tail], dim=-1), torch.cat([valid_mask, tail_mask], dim=-1)


def _trl_jsd_divergence(student_log_probs, teacher_log_probs, beta, support_mask):
    safe_student = torch.where(support_mask, student_log_probs, torch.zeros_like(student_log_probs))
    safe_teacher = torch.where(support_mask, teacher_log_probs, torch.zeros_like(teacher_log_probs))
    student_probs = torch.where(support_mask, student_log_probs.exp(), torch.zeros_like(student_log_probs))
    teacher_probs = torch.where(support_mask, teacher_log_probs.exp(), torch.zeros_like(teacher_log_probs))
    if beta == 0:
        return torch.nan_to_num(teacher_probs * (safe_teacher - safe_student), nan=0.0)
    if beta == 1:
        return torch.nan_to_num(student_probs * (safe_student - safe_teacher), nan=0.0)
    beta_t = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
    tiny = torch.finfo(student_probs.dtype).tiny
    mixture_probs = (1 - beta_t) * student_probs + beta_t * teacher_probs
    safe_mixture = torch.where(support_mask, torch.log(mixture_probs.clamp_min(tiny)), torch.zeros_like(student_log_probs))
    kl_teacher = torch.nan_to_num(teacher_probs * (safe_teacher - safe_mixture), nan=0.0)
    kl_student = torch.nan_to_num(student_probs * (safe_student - safe_mixture), nan=0.0)
    return beta_t * kl_teacher + (1 - beta_t) * kl_student


def _trl_token_mean_jsd(student_logits_k, teacher_logprobs_k, loss_mask, *, student_logit_logsumexp, add_tail_bucket):
    valid = torch.isfinite(teacher_logprobs_k)
    student_support = (student_logits_k.float() - student_logit_logsumexp.unsqueeze(-1).float()).masked_fill(
        ~valid, float("-inf")
    )
    teacher_support = teacher_logprobs_k.masked_fill(~valid, float("-inf"))
    if add_tail_bucket:
        student_logp, support = _trl_add_tail_bucket(student_support, valid)
        teacher_logp, _ = _trl_add_tail_bucket(teacher_support, valid)
    else:
        student_logp = student_support - torch.logsumexp(torch.where(valid, student_support, torch.full_like(student_support, float("-inf"))), dim=-1, keepdim=True)
        teacher_logp = teacher_support - torch.logsumexp(torch.where(valid, teacher_support, torch.full_like(teacher_support, float("-inf"))), dim=-1, keepdim=True)
        support = valid
    per_elem = _trl_jsd_divergence(student_logp, teacher_logp, beta=0.0, support_mask=support)
    per_token = per_elem.sum(dim=-1)
    mask = loss_mask.to(dtype=per_token.dtype)
    return (per_token * mask).sum() / mask.sum().clamp(min=1.0)


def test_arctic_jsd_matches_trl_formula_beta0():
    torch.manual_seed(0)
    vocab = 32
    hidden = torch.randn(2, 4, 8)
    weight = torch.randn(vocab, 8)
    logits = hidden @ weight.T
    ids = torch.tensor(
        [
            [[0, 3, 5], [1, 2, 7], [4, 8, 9], [0, 1, 2]],
            [[6, 7, 8], [10, 11, 12], [0, 15, 16], [20, 21, 22]],
        ]
    )
    gathered = torch.gather(logits, dim=-1, index=ids)
    lse = torch.logsumexp(logits.float(), dim=-1)
    teacher = torch.log_softmax(torch.randn(2, 4, 3), dim=-1)
    teacher[0, 3, 2] = float("-inf")
    mask = torch.tensor([[True, True, True, False], [True, False, True, True]])

    arctic = generalized_jsd(
        gathered,
        teacher,
        mask,
        beta=0.0,
        add_tail_bucket=True,
        student_logit_logsumexp=lse,
    )
    trl = _trl_token_mean_jsd(
        gathered,
        teacher,
        mask,
        student_logit_logsumexp=lse,
        add_tail_bucket=True,
    )
    rel = (arctic - trl).abs() / trl.abs().clamp(min=1e-8)
    assert torch.isfinite(arctic) and torch.isfinite(trl)
    assert float(rel) < 1e-4, f"arctic={float(arctic)} trl={float(trl)} rel={float(rel)}"


def test_arctic_jsd_matches_trl_formula_no_tail():
    torch.manual_seed(1)
    logits_k = torch.tensor([[[1.2, 0.1, -0.4], [0.3, 0.3, 0.3]]])
    lse = torch.logsumexp(torch.cat([logits_k, torch.zeros(1, 2, 5)], dim=-1), dim=-1)
    teacher = torch.log_softmax(torch.tensor([[[0.5, 0.4, 0.1], [1.0, 0.0, -1.0]]]), dim=-1)
    mask = torch.tensor([[True, True]])
    arctic = generalized_jsd(
        logits_k, teacher, mask, beta=0.0, add_tail_bucket=False, student_logit_logsumexp=lse
    )
    trl = _trl_token_mean_jsd(
        logits_k, teacher, mask, student_logit_logsumexp=lse, add_tail_bucket=False
    )
    rel = (arctic - trl).abs() / trl.abs().clamp(min=1e-8)
    assert float(rel) < 1e-4
