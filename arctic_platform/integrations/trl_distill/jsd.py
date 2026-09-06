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

"""Generalized JSD on a sparse teacher support (CPU)."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _add_tail_bucket(log_probs: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    log_sum = torch.logsumexp(torch.where(valid_mask, log_probs, torch.full_like(log_probs, float("-inf"))), dim=-1, keepdim=True)
    log_sum = torch.clamp(log_sum, max=-1e-7)
    tail = torch.log(-torch.expm1(log_sum))
    tail_mask = torch.ones((*valid_mask.shape[:-1], 1), dtype=torch.bool, device=valid_mask.device)
    return torch.cat([log_probs, tail], dim=-1), torch.cat([valid_mask, tail_mask], dim=-1)


def generalized_jsd(
    student_logits_k: torch.Tensor,
    teacher_logprobs_k: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    beta: float = 0.0,
    add_tail_bucket: bool = True,
) -> torch.Tensor:
    """Token-mean generalized JSD. ``beta=0`` forward KL, ``beta=1`` reverse KL."""
    valid = torch.isfinite(teacher_logprobs_k)
    student_logits = student_logits_k.masked_fill(~valid, float("-inf"))
    teacher_logp = teacher_logprobs_k.masked_fill(~valid, float("-inf"))
    if add_tail_bucket:
        student_logp = torch.log_softmax(student_logits, dim=-1)
        student_logp, student_valid = _add_tail_bucket(student_logp, valid)
        teacher_logp, teacher_valid = _add_tail_bucket(teacher_logp, valid)
        support = student_valid & teacher_valid
    else:
        student_logp = torch.log_softmax(student_logits, dim=-1)
        support = valid

    if beta == 0.0:
        per_token = F.kl_div(student_logp, teacher_logp, reduction="none", log_target=True)
    elif beta == 1.0:
        per_token = F.kl_div(teacher_logp, student_logp, reduction="none", log_target=True)
    else:
        beta_t = student_logp.new_tensor(beta)
        mixture = torch.logsumexp(
            torch.stack([student_logp + torch.log1p(-beta_t), teacher_logp + torch.log(beta_t)]),
            dim=0,
        )
        per_token = beta_t * F.kl_div(mixture, teacher_logp, reduction="none", log_target=True)
        per_token = per_token + (1 - beta_t) * F.kl_div(mixture, student_logp, reduction="none", log_target=True)

    per_token = torch.where(support, per_token, torch.zeros_like(per_token)).sum(dim=-1)
    mask = loss_mask.to(dtype=per_token.dtype)
    denom = mask.sum().clamp(min=1.0)
    return (per_token * mask).sum() / denom
