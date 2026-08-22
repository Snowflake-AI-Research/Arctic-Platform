# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Single-logit on-policy distillation loss."""

from __future__ import annotations

import torch

from .functional import agg_loss
from .functional import kl_penalty
from .pipeline import register_loss_fn


@register_loss_fn("on_policy_distill")
def on_policy_distill_loss(
    model_outputs: dict,
    batch: dict,
    meta: dict,
    config: dict,
    device: str,
) -> tuple[torch.Tensor, dict]:
    """Reverse-KL estimate on completion tokens sampled by the student.

    ``compute_entropy_and_logprobs`` supplies the student's causal token
    log-probabilities. The fixed teacher values and masks travel with the batch,
    and the teacher is always detached.
    """
    student_logprobs = model_outputs.get("logprobs")
    if student_logprobs is None:
        raise ValueError("on_policy_distill requires post=['compute_entropy_and_logprobs']")

    teacher = batch["teacher_log_probs_shifted"].to(student_logprobs.device).detach()
    loss_mask = batch["loss_mask"].to(student_logprobs.device).bool()
    if teacher.shape != student_logprobs.shape or loss_mask.shape != student_logprobs.shape:
        raise ValueError(
            "student logprobs, teacher_log_probs_shifted, and loss_mask must have identical shapes"
        )
    if not loss_mask.any():
        raise ValueError("on_policy_distill loss_mask is empty")

    estimator = config.get("distill_estimator", "low_var_kl")
    per_token_kl = kl_penalty(student_logprobs, teacher, method=estimator)
    loss = float(config.get("kl_coef", 1.0)) * agg_loss(
        per_token_kl,
        loss_mask,
        loss_agg_mode=config.get("loss_agg_mode", "token-mean"),
        dp_size=config.get("dp_size", 1),
        batch_num_tokens=config.get("batch_num_tokens"),
        global_batch_size=config.get("global_batch_size"),
    )

    masked_kl = per_token_kl.detach()[loss_mask]
    metrics = {
        "distill_kl": float(masked_kl.mean().cpu()),
        "distill_kl_sum": float(masked_kl.sum().cpu()),
        "distill_kl_count": int(masked_kl.numel()),
        "loss": float(loss.detach().cpu()),
    }
    old_logprobs = batch.get("old_log_probs_shifted")
    if old_logprobs is not None:
        old = old_logprobs.to(student_logprobs.device)
        delta = (student_logprobs.detach() - old).abs()[loss_mask]
        metrics.update(
            {
                "sampler_train_abs_delta_max": float(delta.max().cpu()),
                "sampler_train_abs_delta_mean": float(delta.mean().cpu()),
            }
        )
    return loss, metrics
