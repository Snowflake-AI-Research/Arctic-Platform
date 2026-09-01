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

"""Single-logit on-policy distillation loss.

The student generates on-policy, a frozen teacher scores those exact same token
ids, and the student minimises per-token reverse KL ``KL(pi_student || pi_teacher)``
estimated from the sampled token alone.

``functional.kl_penalty(method="low_var_kl")`` computes the same k3 estimator but
clamps its *output* to ``[-10, 10]``. That zeroes the gradient for
``|delta| ≳ 2.63`` — exactly where student and teacher disagree most. This module
clamps only the *input* ``delta`` (to keep ``exp`` finite) and leaves the output
cap opt-in via ``kl_clamp_max``.
"""

from __future__ import annotations

from typing import Optional

import torch

from .functional import agg_loss
from .pipeline import register_loss_fn

_DEFAULT_DELTA_CLAMP = 20.0

_ESTIMATORS_K3 = frozenset({"low_var_kl", "k3"})
_ESTIMATORS_K2 = frozenset({"mse", "k2"})
_ESTIMATORS_ABS = frozenset({"abs"})
_ESTIMATORS_VALUE_ONLY = frozenset({"kl", "k1"})
_ESTIMATORS_TRAINABLE = _ESTIMATORS_K3 | _ESTIMATORS_K2 | _ESTIMATORS_ABS

_DISTILL_CONFIG_KEYS = frozenset(
    {
        "distill_estimator",
        "kl_coef",
        "delta_clamp",
        "kl_clamp_max",
        "loss_agg_mode",
        "dp_size",
        "batch_num_tokens",
        "global_batch_size",
    }
)


def _packed_singleton_to_1d(tensor: torch.Tensor) -> torch.Tensor:
    """``pack_sequences`` emits ``[1, T]``; canonicalise to ``[T]`` for the loss."""
    if torch.is_tensor(tensor) and tensor.ndim == 2 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def count_opd_loss_tokens(batch_data) -> tuple[int, int]:
    """Local ``(tokens, sequences)`` from ``loss_mask`` on a rank shard or GAS list.

    The DeepSpeed worker all-reduces these into ``batch_num_tokens`` / ``dp_size``
    so each microbatch is ``sum(kl) / T_global * dp_size`` instead of a local
    per-rollout mean.
    """
    if isinstance(batch_data, list):
        tokens = seqs = 0
        for micro_batch in batch_data:
            mb_tokens, mb_seqs = count_opd_loss_tokens(micro_batch)
            tokens += mb_tokens
            seqs += mb_seqs
        return tokens, seqs
    if not isinstance(batch_data, dict):
        return 0, 0
    loss_mask = batch_data.get("loss_mask")
    if not torch.is_tensor(loss_mask):
        return 0, 0
    tokens = int(loss_mask.count_nonzero().item())
    if loss_mask.ndim >= 2 and loss_mask.shape[0] > 1:
        seqs = int(loss_mask.reshape(loss_mask.shape[0], -1).any(dim=-1).sum().item())
    elif torch.is_tensor(batch_data.get("cu_seqlens")) and batch_data["cu_seqlens"].numel() >= 2:
        seqs = int(batch_data["cu_seqlens"].numel()) - 1
    else:
        seqs = 1 if tokens else 0
    return tokens, seqs


def _positive_int(value) -> Optional[int]:
    if value is None:
        return None
    parsed = int(value)
    return parsed if parsed > 0 else None


def apply_opd_global_token_config(
    processing: dict,
    meta_data: dict,
    *,
    dp_size: int,
    batch_num_tokens: int,
    global_batch_size: int,
) -> None:
    """Write all-reduced token counts into OPD ``processing['config']`` and ``meta``."""
    config = processing.setdefault("config", {})
    if isinstance(config, dict):
        config["dp_size"] = int(dp_size)
        config["batch_num_tokens"] = int(batch_num_tokens)
        config["global_batch_size"] = int(global_batch_size)
    meta_data["dp_size"] = int(dp_size)
    meta_data["batch_num_tokens"] = int(batch_num_tokens)
    meta_data["global_num_tokens"] = int(batch_num_tokens)
    meta_data["global_batch_size"] = int(global_batch_size)


def _resolve_distill_norm(config: dict, meta: dict) -> tuple[int, Optional[int], Optional[int]]:
    """``(dp_size, batch_num_tokens, global_batch_size)`` from config, then meta."""
    meta = meta if isinstance(meta, dict) else {}
    dp_size = int(config.get("dp_size") or meta.get("dp_size") or 1)
    batch_num_tokens = _positive_int(config.get("batch_num_tokens"))
    if batch_num_tokens is None:
        batch_num_tokens = _positive_int(meta.get("batch_num_tokens"))
    if batch_num_tokens is None:
        batch_num_tokens = _positive_int(meta.get("global_num_tokens"))
    global_batch_size = _positive_int(config.get("global_batch_size"))
    if global_batch_size is None:
        global_batch_size = _positive_int(meta.get("global_batch_size"))
    return dp_size, batch_num_tokens, global_batch_size


def _distill_kl_per_token(
    student_logprobs: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    *,
    estimator: str,
    delta_clamp: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token reverse-KL objective, differentiable through ``student_logprobs``.

    With ``delta = teacher - student`` (teacher detached), k3 is
    ``exp(delta) - delta - 1``, whose gradient ``1 - exp(delta)`` pushes the
    student toward the teacher. Returns ``(per_token_loss, unclamped delta)``.

    ``k1`` is rejected: ``student - teacher`` has derivative ``+1``, so minimising
    it drives every sampled logprob down regardless of the teacher.
    """
    if estimator in _ESTIMATORS_VALUE_ONLY:
        raise ValueError(
            f"distill_estimator='{estimator}' is a KL *value* estimator, not a trainable "
            "objective: its gradient w.r.t. the student logprob is the constant +1, so "
            "minimising it lowers every sampled logprob irrespective of the teacher. It is "
            "reported as the 'distill_k1_sum' metric instead. Use 'low_var_kl' (k3, default), "
            "'mse' (k2), or 'abs'."
        )
    if estimator not in _ESTIMATORS_TRAINABLE:
        raise ValueError(
            f"Unknown distill_estimator '{estimator}'. Expected one of "
            f"{sorted(_ESTIMATORS_TRAINABLE)} (or 'k1'/'kl', which are value-only)."
        )

    delta = teacher_logprobs.float() - student_logprobs.float()
    delta_clamped = torch.clamp(delta, min=-delta_clamp, max=delta_clamp)

    if estimator in _ESTIMATORS_K3:
        per_token = delta_clamped.exp() - delta_clamped - 1.0
    elif estimator in _ESTIMATORS_K2:
        per_token = 0.5 * delta_clamped.square()
    else:
        per_token = delta_clamped.abs()
    return per_token, delta


def _masked_sum(values: torch.Tensor, mask: torch.Tensor) -> float:
    values = values.detach().float()
    return float(torch.where(mask.bool(), values, torch.zeros_like(values)).sum().item())


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask_bool = mask.bool()
    if not bool(mask_bool.any()):
        return 0.0
    values = values.detach().float()
    return float(values[mask_bool].max().item())


def _require_batch_tensor(batch: dict, key: str, *, produced_by: str) -> torch.Tensor:
    value = batch.get(key)
    if value is None:
        raise ValueError(
            f"loss_fn 'on_policy_distill' requires batch key '{key}' — {produced_by}. "
            f"Present batch keys: {sorted(batch)}"
        )
    if not torch.is_tensor(value):
        raise TypeError(f"batch['{key}'] must be a tensor, got {type(value).__name__}")
    return value


def _student_logprobs(model_outputs: dict, batch: dict) -> torch.Tensor:
    logprobs = model_outputs.get("logprobs")
    if logprobs is not None:
        return logprobs
    if "logits" not in model_outputs:
        raise ValueError("on_policy_distill requires post=['compute_entropy_and_logprobs']")
    logits = model_outputs["logits"]
    input_ids = batch["input_ids"].to(logits.device)
    if input_ids.ndim < logits.ndim:
        input_ids = input_ids.view(logits.shape[:-1])
    labels = torch.roll(input_ids, shifts=-1, dims=-1)
    return torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)


@register_loss_fn("on_policy_distill")
def on_policy_distill_loss(
    model_outputs: dict,
    batch: dict,
    meta: dict,
    config: dict,
    device: str,
) -> tuple[torch.Tensor, dict]:
    """Single-logit on-policy distillation: reverse KL against a frozen teacher.

    The config schema is strict — an unrecognised key raises rather than being
    ignored, so a typo cannot silently fall back to a different objective.
    """
    unknown_keys = set(config) - _DISTILL_CONFIG_KEYS
    if unknown_keys:
        raise ValueError(
            f"Unknown config keys for loss_fn 'on_policy_distill': {sorted(unknown_keys)} — this "
            "contract fails loudly on unrecognized keys so a typo cannot silently change the "
            f"objective. Known keys: {sorted(_DISTILL_CONFIG_KEYS)}"
        )

    logprobs = _student_logprobs(model_outputs, batch)
    if not torch.isfinite(logprobs).all():
        logprobs = torch.nan_to_num(logprobs, nan=0.0, posinf=0.0, neginf=0.0)

    teacher_logprobs = _require_batch_tensor(
        batch,
        "teacher_log_probs_shifted",
        produced_by="the teacher's per-token logprobs for the same token ids, roll(-1)-aligned",
    ).to(logprobs.device)
    loss_mask = _require_batch_tensor(
        batch,
        "loss_mask",
        produced_by="True on the completion-predicting positions the student is trained on",
    ).to(logprobs.device)

    cu_seqlens = batch.get("cu_seqlens")
    if cu_seqlens is None and isinstance(meta, dict):
        cu_seqlens = meta.get("cu_seqlens")
    if cu_seqlens is not None:
        logprobs = _packed_singleton_to_1d(logprobs)
        teacher_logprobs = _packed_singleton_to_1d(teacher_logprobs)
        loss_mask = _packed_singleton_to_1d(loss_mask)

    if teacher_logprobs.shape != logprobs.shape:
        raise ValueError(
            "student logprobs, teacher_log_probs_shifted, and loss_mask must have identical shapes; "
            f"got teacher {tuple(teacher_logprobs.shape)} vs student {tuple(logprobs.shape)}"
        )
    if loss_mask.shape != logprobs.shape:
        raise ValueError(
            "student logprobs, teacher_log_probs_shifted, and loss_mask must have identical shapes; "
            f"got loss_mask {tuple(loss_mask.shape)} vs student {tuple(logprobs.shape)}"
        )
    if not loss_mask.any():
        raise ValueError("on_policy_distill loss_mask is empty")

    estimator = config.get("distill_estimator", "low_var_kl")
    delta_clamp = float(config.get("delta_clamp", _DEFAULT_DELTA_CLAMP))
    kl_clamp_max: Optional[float] = config.get("kl_clamp_max")

    per_token_kl, delta = _distill_kl_per_token(
        logprobs,
        teacher_logprobs.detach(),
        estimator=estimator,
        delta_clamp=delta_clamp,
    )

    kl_output_clamped = torch.zeros((), device=logprobs.device)
    if kl_clamp_max is not None:
        kl_output_clamped = (per_token_kl.detach() > float(kl_clamp_max)).to(per_token_kl.dtype)
        per_token_kl = torch.clamp(per_token_kl, max=float(kl_clamp_max))

    dp_size, batch_num_tokens, global_batch_size = _resolve_distill_norm(config, meta)
    loss = agg_loss(
        per_token_kl,
        loss_mask,
        loss_agg_mode=config.get("loss_agg_mode", "token-mean"),
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
    )
    kl_coef = float(config.get("kl_coef", 1.0))
    loss = kl_coef * loss

    mask_count = float(loss_mask.count_nonzero().item())
    masked_kl_mean = _masked_sum(per_token_kl, loss_mask) / max(mask_count, 1.0)
    masked_k1_mean = _masked_sum(-delta, loss_mask) / max(mask_count, 1.0)
    kl_sum = _masked_sum(per_token_kl, loss_mask)
    metrics = {
        "loss": float(loss.detach().cpu()),
        "loss.sum": kl_coef * kl_sum,
        "loss.tokens": mask_count,
        "distill_kl": masked_kl_mean,
        "distill_kl.sum": kl_sum,
        "distill_kl.tokens": mask_count,
        "distill_k1": masked_k1_mean,
        "distill_kl_sum": _masked_sum(per_token_kl, loss_mask),
        "distill_kl_count": mask_count,
        "distill_k1_sum": _masked_sum(-delta, loss_mask),
        "distill_abs_delta_sum": _masked_sum(delta.abs(), loss_mask),
        "teacher_logprob_sum": _masked_sum(teacher_logprobs, loss_mask),
        "student_logprob_sum": _masked_sum(logprobs, loss_mask),
        "distill_delta_clamped_count": _masked_sum((delta.abs() > delta_clamp).to(delta.dtype), loss_mask),
        "distill_kl_output_clamped_count": (
            _masked_sum(kl_output_clamped, loss_mask) if kl_clamp_max is not None else 0.0
        ),
        "loss_term_distill": float(loss.detach().item()),
        "distill_kl_max": _masked_max(per_token_kl, loss_mask),
        "distill_delta_max": _masked_max(delta, loss_mask),
        "distill_delta_min": -_masked_max(-delta, loss_mask),
        "distill_kl_coef": kl_coef,
        "distill_delta_clamp": delta_clamp,
        "distill_estimator_is_k3": float(estimator in _ESTIMATORS_K3),
        "distill_dp_size": float(dp_size),
        "distill_batch_num_tokens": float(batch_num_tokens if batch_num_tokens is not None else mask_count),
    }

    sampler_logprobs = batch.get("old_log_probs_shifted")
    if sampler_logprobs is not None:
        sampler_logprobs = sampler_logprobs.to(logprobs.device)
        if cu_seqlens is not None:
            sampler_logprobs = _packed_singleton_to_1d(sampler_logprobs)
        if sampler_logprobs.shape == logprobs.shape:
            sampler_delta = sampler_logprobs.detach().float() - logprobs.detach().float()
            sampler_delta_clamped = torch.clamp(sampler_delta, min=-delta_clamp, max=delta_clamp)
            metrics["sampler_train_kl_sum"] = _masked_sum(
                sampler_delta_clamped.exp() - sampler_delta_clamped - 1.0, loss_mask
            )
            metrics["sampler_train_abs_delta_sum"] = _masked_sum(sampler_delta.abs(), loss_mask)
            metrics["sampler_train_abs_delta_max"] = _masked_max(sampler_delta.abs(), loss_mask)
            abs_delta = sampler_delta.abs()[loss_mask.bool()]
            metrics["sampler_train_abs_delta_mean"] = (
                float(abs_delta.mean().item()) if abs_delta.numel() else 0.0
            )

    return loss, metrics
