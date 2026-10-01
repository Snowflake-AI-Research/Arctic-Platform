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

"""GRPO loss function and proximal logp utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum
from itertools import pairwise
from typing import Any
from typing import Tuple

import torch

from arctic_platform.common.registry import declare_loss_capabilities
from arctic_platform.common.registry import register_loss_fn

from .base_loss import PRESERVES_EXPLICIT_LOSS_SCALE_CONFIG
from .base_loss import REQUIRES_ALIGNED_TOKEN_LOGPROBS
from .functional import RATIO_MASK_CONFIG_KEYS
from .functional import EchoBatchDenominator
from .functional import RatioMasks
from .functional import _full_observation_denominator
from .functional import _get_sequence_parallel_group
from .functional import _packed_per_sequence_sums
from .functional import _resolve_dp_size
from .functional import _validate_loss_denominators
from .functional import agg_loss
from .functional import canonicalize_loss_mask
from .functional import cispo_actor_loss_fn
from .functional import dp_loss_multiplier
from .functional import echo_env_prediction_loss_fn
from .functional import kl_penalty
from .functional import ppo_actor_loss_fn
from .functional import sapo_loss_fn
from .packed_reduction import PackedLossReduction
from .packed_reduction import additive_packed_loss_reduction
from .packed_reduction import local_mean_packed_loss_reduction

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_EPSILON = 1e-8


class ProxLogpMethod(str, Enum):
    """Method for computing proximal policy log-probabilities in decoupled PPO."""

    RECOMPUTE = "recompute"
    LOGLINEAR = "loglinear"
    METRICS = "metrics"

    def skips_forward_pass(self) -> bool:
        return self == ProxLogpMethod.LOGLINEAR


class ProxApproxMethod(str, Enum):
    """Approximation method for proximal policy log-probabilities."""

    LOGLINEAR = "loglinear"
    LINEAR = "linear"
    ROLLOUT = "rollout"


PROX_LOGP_METHOD_RECOMPUTE = ProxLogpMethod.RECOMPUTE.value
PROX_LOGP_METHOD_LOGLINEAR = ProxLogpMethod.LOGLINEAR.value
PROX_LOGP_METHOD_METRICS = ProxLogpMethod.METRICS.value
PROX_APPROX_METHOD_LOGLINEAR = ProxApproxMethod.LOGLINEAR.value
PROX_APPROX_METHOD_LINEAR = ProxApproxMethod.LINEAR.value
PROX_APPROX_METHOD_ROLLOUT = ProxApproxMethod.ROLLOUT.value
PROX_APPROX_METHODS_ALL = [m.value for m in ProxApproxMethod]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_importance_weight():
    pass  # placeholder — actual logic is inline in loss fns


def _compute_approximation_errors():
    pass  # placeholder


def _tensor_scalar_stats():
    pass  # placeholder


def _packed_singleton_to_1d(tensor: torch.Tensor | None) -> torch.Tensor | None:
    """Squeeze ``pack_sequences``' singleton [1, T] layout to canonical 1-D [T]."""
    if tensor is not None and tensor.ndim == 2 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def _masked_mean_float(values: torch.Tensor, mask: torch.Tensor) -> float:
    mask = mask.bool()
    safe_values = torch.where(mask, values, torch.zeros_like(values))
    return float(safe_values.sum() / mask.sum().clamp(min=1))


def _validate_nll_mask(nll_mask: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    mask = canonicalize_loss_mask(nll_mask, loss_mask, objective="nll_mask", binary=True)
    if (mask & ~loss_mask).any().item():
        raise ValueError("nll_mask must be a subset of loss_mask")
    return mask


def _source_statistics(stat: dict, mask: torch.Tensor) -> dict[str, float]:
    def total(values):
        return float(torch.where(mask, values.detach().double(), 0.0).sum())

    count = float(mask.sum())
    metrics = {
        "grpo_stats_token_count": count,
        "grpo_importance_weight_sum": total(stat["importance_weight"]),
        "grpo_log_ratio_sum": total(stat["approx_kl"]),
        "grpo_clipped_token_count": total(stat["clip_mask"]),
    }
    if "clipped_is_weight" in stat:
        weight = stat["clipped_is_weight"].detach().double()
        metrics.update(
            cispo_is_weight_sum=total(weight),
            cispo_is_weight_sq_sum=total(weight.square()),
            cispo_trainable_token_count=count,
            cispo_lower_tail_token_count=total(stat["lower_tail_mask"]),
        )
    return metrics


def _grpo_metrics_callback(_worker_metrics: Sequence[dict], metrics: dict) -> None:
    totals = {
        "grpo_stats_token_count",
        "grpo_importance_weight_sum",
        "grpo_log_ratio_sum",
        "grpo_clipped_token_count",
        "grpo_entropy_sum",
    }
    if not totals.issubset(metrics):
        return
    count = max(float(metrics["grpo_stats_token_count"]), 1.0)
    metrics.update(
        {
            name: float(metrics[total]) / count
            for name, total in (
                ("importance_weight", "grpo_importance_weight_sum"),
                ("approx_kl", "grpo_log_ratio_sum"),
                ("clip_ratio", "grpo_clipped_token_count"),
                ("entropy", "grpo_entropy_sum"),
            )
        }
    )


def _resolve_teacher_tau(config: dict, context: dict) -> float:
    teacher_tau = config.get("teacher_tau")
    teacher_clip = config.get("teacher_clip")
    teacher_clip_negative = config.get("teacher_clip_negative")
    if teacher_tau is None:
        return 0.0
    if isinstance(teacher_tau, bool) or not isinstance(teacher_tau, (int, float)):
        raise ValueError(f"teacher_tau must be a finite non-negative number, got {teacher_tau!r}")
    tau = float(teacher_tau)
    if not math.isfinite(tau) or tau < 0.0:
        raise ValueError(f"teacher_tau must be a finite non-negative number, got {teacher_tau!r}")
    if tau == 0.0:
        return 0.0
    if (
        teacher_clip is None
        or isinstance(teacher_clip, bool)
        or not isinstance(teacher_clip, (int, float))
        or not math.isfinite(teacher_clip)
        or teacher_clip <= 0.0
    ):
        raise ValueError(f"teacher_tau={tau} requires a finite positive config 'teacher_clip', got {teacher_clip!r}")
    if teacher_clip_negative is not None and (
        isinstance(teacher_clip_negative, bool)
        or not isinstance(teacher_clip_negative, (int, float))
        or not math.isfinite(teacher_clip_negative)
        or teacher_clip_negative < 0.0
    ):
        raise ValueError(f"teacher_clip_negative must be a finite non-negative number, got {teacher_clip_negative!r}")
    if context.get("teacher_log_probs_shifted") is None:
        raise ValueError("teacher_tau > 0 requires context 'teacher_log_probs_shifted'")
    if config.get("importance_sampling_level", "token") != "token":
        raise ValueError("teacher_tau > 0 requires importance_sampling_level='token'")
    if config.get("loss_agg_mode", "token-mean") == "token-mean" and context.get("sequence_loss_weights") is not None:
        raise ValueError("teacher_tau > 0 with token-mean cannot carry sequence_loss_weights to the teacher term")
    return tau


def compute_prox_logp_approximations(
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor,
    current_version: int,
    method: str | None = None,
) -> dict[str, torch.Tensor]:
    v_proximal = current_version - 1
    v_behave = versions.float()
    v_theta = float(current_version)
    generated_tokens_mask = versions >= 0
    version_diff = v_theta - v_behave
    version_gap = v_proximal - v_behave
    alpha = torch.where(
        (version_diff > 0) & generated_tokens_mask, version_gap / version_diff, torch.zeros_like(v_behave)
    )
    alpha = torch.clamp(alpha, 0.0, 1.0)
    approximations = {}
    methods_to_compute = [method] if method else PROX_APPROX_METHODS_ALL
    for m in methods_to_compute:
        if m == PROX_APPROX_METHOD_LOGLINEAR:
            approximations[PROX_APPROX_METHOD_LOGLINEAR] = old_logp + alpha * (logprobs - old_logp)
        elif m == PROX_APPROX_METHOD_LINEAR:
            p_arithmetic = (1 - alpha) * torch.exp(old_logp) + alpha * torch.exp(logprobs)
            approximations[PROX_APPROX_METHOD_LINEAR] = torch.log(p_arithmetic + 1e-10)
        elif m == PROX_APPROX_METHOD_ROLLOUT:
            approximations[PROX_APPROX_METHOD_ROLLOUT] = old_logp.clone()
    return approximations


def _resolve_proximal_logp(
    prox_logp_gt: torch.Tensor | None,
    prox_logp_method: str,
    old_logp: torch.Tensor,
    logprobs: torch.Tensor,
    versions: torch.Tensor | None,
    current_version: int | None,
) -> torch.Tensor:
    prox_logp_is_none = prox_logp_gt is None
    if prox_logp_is_none:
        if prox_logp_method == PROX_LOGP_METHOD_RECOMPUTE:
            # On-policy default: proximal policy = behavioral policy.
            # Frameworks that don't have a separate prox_logp (VERL, SkyRL) can
            # omit the field; AReaL passes it explicitly for async off-policy training.
            return old_logp
        if not ProxLogpMethod(prox_logp_method).skips_forward_pass():
            raise ValueError(f"prox_logp is None but prox_logp_method='{prox_logp_method}'.")
        if versions is None:
            raise ValueError(
                f"prox_logp is None with prox_logp_method='{prox_logp_method}' but versions not available."
            )
    prox_logp = prox_logp_gt
    if prox_logp_method == PROX_LOGP_METHOD_LOGLINEAR:
        if prox_logp_is_none and versions is not None and current_version is not None:
            approximations = compute_prox_logp_approximations(
                old_logp=old_logp,
                logprobs=logprobs,
                versions=versions,
                current_version=current_version,
                method=PROX_APPROX_METHOD_LOGLINEAR,
            )
            prox_logp = approximations[PROX_APPROX_METHOD_LOGLINEAR]
    if prox_logp is None:
        raise RuntimeError(f"prox_logp is None after handling prox_logp_method='{prox_logp_method}'.")
    if torch.isnan(prox_logp).any() or torch.isinf(prox_logp).any():
        raise RuntimeError(f"prox_logp contains NaN or Inf with prox_logp_method='{prox_logp_method}'.")
    return prox_logp


def _get_m2po_loss_mask(old_logp, prox_logp, loss_mask, m2_threshold):
    return _apply_m2po_masking(old_logp, prox_logp, loss_mask, m2_threshold)


def _apply_m2po_masking(old_logp, prox_logp, loss_mask, m2_threshold):
    delta = old_logp - prox_logp
    m2 = delta * delta
    mask_flat = loss_mask.view(-1)
    m2_selected = m2.view(-1)[mask_flat]
    if m2_selected.numel() == 0:
        return loss_mask
    sorted_m2, indices = torch.sort(m2_selected, descending=True)
    restored_indices = torch.argsort(indices)
    n = sorted_m2.numel()
    suffix_sums = sorted_m2.flip(0).cumsum(0).flip(0)
    counts = torch.arange(n, 0, -1, device=sorted_m2.device, dtype=sorted_m2.dtype)
    avg_m2_suffix = suffix_sums / counts
    below = torch.where(avg_m2_suffix < m2_threshold)[0]
    num_to_mask = int(below[0].item()) if len(below) > 0 else n - 1
    sorted_mask = torch.ones(n, dtype=torch.bool, device=sorted_m2.device)
    if num_to_mask > 0:
        sorted_mask[:num_to_mask] = False
    if sorted_mask.sum() == 0:
        raise RuntimeError("All tokens are masked out when applying M2PO masking.")
    m2_selected_mask = sorted_mask[restored_indices]
    m2_full_flat = torch.zeros_like(mask_flat, dtype=torch.bool)
    m2_full_flat[mask_flat] = m2_selected_mask
    return m2_full_flat.view_as(loss_mask)


def _internal_grpo_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    eps_clip: float,
    eps_clip_higher: float | None,
    c_clip: float | None,
    behav_imp_weight_cap: float | None,
    m2_threshold: float | None = None,
    importance_sampling_level: str = "token",
    current_version: int | None = None,
    prox_logp_method: str = PROX_LOGP_METHOD_RECOMPUTE,
    use_sapo_loss: bool = False,
    sapo_tau_pos: float = 1.0,
    sapo_tau_neg: float = 1.05,
    use_decoupled_loss: bool = False,
    use_cispo_loss: bool = False,
    is_weight_clip_max: float | None = None,
    # --- VeRL-compatible aggregation and auxiliary loss knobs ---
    loss_agg_mode: str = "token-mean",
    dp_size: int = 1,
    batch_num_tokens: int | None = None,
    global_batch_size: int | None = None,
    rollout_is_weights: torch.Tensor | None = None,
    entropy_coeff: float = 0.0,
    use_kl_loss: bool = False,
    kl_loss_coef: float = 0.001,
    kl_loss_type: str = "low_var_kl",
    prompt_group_ids: torch.Tensor | None = None,
    prompt_token_counts: torch.Tensor | None = None,
    sequence_loss_weights: torch.Tensor | None = None,
    aux_ce_weight: float | None = None,
    echo_global_num_sequences: int | None = None,
    echo_batch_denominator: str = EchoBatchDenominator.ALL_SEQUENCES.value,
    teacher_tau: float = 0.0,
    teacher_clip: float | None = None,
    teacher_clip_negative: float | None = None,
    ratio_masks: RatioMasks | None = None,
) -> Tuple[torch.Tensor, dict]:
    """Internal GRPO loss — same interface as dss/loss_fns/grpo.py."""
    dp_size = _resolve_dp_size(dp_size, batch_num_tokens)
    old_logp = input_data["old_log_probs"]
    advantages = input_data["advantages"]
    loss_mask = canonicalize_loss_mask(
        input_data["loss_mask"],
        logprobs,
        objective="grpo",
        binary=True,
    )
    labels = input_data.get("labels")
    if torch.is_tensor(labels):
        labels = labels.to(loss_mask.device)
        if tuple(labels.shape) != tuple(loss_mask.shape):
            raise ValueError("grpo labels must match loss_mask when labels are present")
        if ((labels == -100) & loss_mask).any().item():
            raise ValueError("grpo loss_mask must be zero where labels use ignore_index=-100")
    # ECHO disjointness is validated against the client's policy mask, not the
    # (possibly M2PO-shrunk) mask used for the policy loss below.
    policy_loss_mask = loss_mask
    nll_mask = input_data.get("nll_mask")
    if nll_mask is not None:
        nll_mask = _validate_nll_mask(nll_mask, loss_mask)
    prox_logp_gt = input_data.get("prox_logp")
    if nll_mask is not None:
        old_logp = torch.where(nll_mask, 0.0, old_logp)
        if prox_logp_gt is not None:
            prox_logp_gt = torch.where(nll_mask, 0.0, prox_logp_gt)
    entropy = entropy.detach()

    # All-padded shard guard. Under Ulysses SP the batch is split into contiguous
    # per-rank sequence shards; a short/padded batch can hand a rank a shard whose
    # tokens are all prompt/padding (loss_mask all False). The PPO ratio + prompt/
    # token normalization then divides by zero and poisons the autograd graph with
    # NaN (surfacing as "rank=k loss contains non-finite values"). Mirror
    # ArcticTraining's SFTTrainer: emit a finite zero that still carries grads
    # (via logprobs) so DeepSpeed's cross-rank gradient all-reduce stays in lockstep
    # and this empty shard contributes nothing. nan_to_num keeps the forward value
    # finite even if the shard produced non-finite logits; the * 0.0 zeroes the grad.
    #
    # When ECHO is configured, skip the early return so sft_mask /
    # echo_observation_mask can still contribute. Empty-policy shards still
    # run the ECHO terms on this path.
    empty_policy_shard = not loss_mask.any()
    validates_zero_denominator = batch_num_tokens == 0 or global_batch_size == 0
    reduces_across_sequence_parallel = _get_sequence_parallel_group() is not None and (
        loss_agg_mode.startswith("seq-mean-")
        or loss_agg_mode == "prompt-mean"
        or importance_sampling_level == "sequence"
    )
    if (
        empty_policy_shard
        and aux_ce_weight is None
        and ratio_masks is None
        and not reduces_across_sequence_parallel
        and not validates_zero_denominator
    ):
        zero_loss = torch.nan_to_num(logprobs).sum() * 0.0
        metrics = {
            "approx_kl": 0.0,
            "importance_weight": 0.0,
            "clip_ratio": 0.0,
            "entropy": 0.0,
        }
        zeros = torch.zeros_like(logprobs)
        stat = dict(importance_weight=zeros, approx_kl=zeros, clip_mask=zeros)
        if use_cispo_loss and is_weight_clip_max is not None:
            stat.update(clipped_is_weight=zeros, lower_tail_mask=zeros)
        metrics.update(_source_statistics(stat, loss_mask))
        metrics["grpo_entropy_sum"] = 0.0
        if nll_mask is not None:
            metrics.update(nll_trainable_token_count=0.0, nll_sum=0.0)
        if teacher_tau > 0.0:
            metrics.update(
                teacher_tau=teacher_tau,
                teacher_term_token_count=0.0,
                teacher_log_ratio_sum=0.0,
                teacher_clipped_log_ratio_sum=0.0,
            )
        return zero_loss, metrics

    # Degenerate-shard logprob sanitization. Under Ulysses SP a short/padded batch
    # can hand a rank a tail shard where some "valid" query positions have no
    # attendable keys, so attention emits non-finite logits -> non-finite logprobs
    # that poison the PPO/CISPO loss (a single NaN makes the whole scalar NaN).
    # Zero the non-finite logprobs before the loss math: nan_to_num keeps the
    # forward finite and yields zero gradient at those positions, so malformed
    # tokens drop out while well-formed tokens are untouched. At real training
    # seqlens the shards are large and this is a no-op.
    if not torch.isfinite(logprobs).all():
        logprobs = torch.nan_to_num(logprobs, nan=0.0, posinf=0.0, neginf=0.0)

    prox_logp = _resolve_proximal_logp(
        prox_logp_gt=prox_logp_gt,
        prox_logp_method=prox_logp_method,
        old_logp=old_logp,
        logprobs=logprobs.detach(),
        versions=input_data.get("versions"),
        current_version=current_version,
    )

    if m2_threshold is not None:
        loss_mask = _apply_m2po_masking(old_logp, prox_logp, loss_mask, m2_threshold)

    teacher_metrics: dict[str, float] = {}
    if teacher_tau > 0.0:
        teacher_delta = input_data["teacher_log_probs"].detach() - logprobs.detach()
        teacher_scored = (
            (loss_mask if nll_mask is None else loss_mask & ~nll_mask)
            & input_data["teacher_policy_finite"]
            & torch.isfinite(teacher_delta)
        )
        teacher_term = torch.where(
            teacher_scored,
            teacher_delta.clamp(
                -(teacher_clip if teacher_clip_negative is None else teacher_clip_negative), teacher_clip
            ),
            torch.zeros_like(teacher_delta),
        )
        advantages = advantages + teacher_tau * teacher_term
        teacher_metrics = {
            "teacher_tau": teacher_tau,
            "teacher_term_token_count": float(teacher_scored.sum()),
            "teacher_log_ratio_sum": float(torch.where(teacher_scored, teacher_delta.double(), 0.0).sum()),
            "teacher_clipped_log_ratio_sum": float(teacher_term.double().sum()),
        }

    if (
        empty_policy_shard
        and ratio_masks is None
        and not reduces_across_sequence_parallel
        and not validates_zero_denominator
    ):
        loss = torch.nan_to_num(logprobs).sum() * 0.0
        metrics = {
            "approx_kl": 0.0,
            "importance_weight": 0.0,
            "clip_ratio": 0.0,
            "entropy": 0.0,
        }
        zeros = torch.zeros_like(logprobs)
        stat = dict(importance_weight=zeros, approx_kl=zeros, clip_mask=zeros)
        if use_cispo_loss and is_weight_clip_max is not None:
            stat.update(clipped_is_weight=zeros, lower_tail_mask=zeros)
        metrics.update(_source_statistics(stat, loss_mask))
        metrics["grpo_entropy_sum"] = 0.0
        if nll_mask is not None:
            metrics.update(nll_trainable_token_count=0.0, nll_sum=0.0)
    else:
        if use_sapo_loss and use_cispo_loss:
            raise ValueError("use_sapo_loss and use_cispo_loss are mutually exclusive.")
        if ratio_masks is not None and not use_cispo_loss:
            raise ValueError("ratio-mask keys act on the CISPO policy term; they need use_cispo_loss=True.")
        ratio_m2_keep = None
        if ratio_masks is not None and ratio_masks.m2_threshold is not None:
            if _get_sequence_parallel_group() is not None:
                raise ValueError("ratio_m2_threshold does not support sequence parallelism")
            ratio_m2_keep = _apply_m2po_masking(old_logp, logprobs.detach(), loss_mask, ratio_masks.m2_threshold)
        if use_cispo_loss and c_clip is not None:
            raise ValueError("c_clip is not supported with use_cispo_loss=True.")

        if use_sapo_loss:
            if use_decoupled_loss:
                raise ValueError("SAPO is not compatible with use_decoupled_loss=True.")
            loss, stat = sapo_loss_fn(
                logprobs=logprobs,
                old_logprobs=old_logp,
                advantages=advantages,
                tau_pos=sapo_tau_pos,
                tau_neg=sapo_tau_neg,
                loss_mask=loss_mask,
                importance_sampling_level=importance_sampling_level,
                cu_seqlens=input_data.get("cu_seqlens"),
                loss_agg_mode=loss_agg_mode,
                dp_size=dp_size,
                batch_num_tokens=batch_num_tokens,
                global_batch_size=global_batch_size,
                prompt_group_ids=prompt_group_ids,
                prompt_token_counts=prompt_token_counts,
                sequence_loss_weights=sequence_loss_weights,
            )
        elif use_cispo_loss:
            loss, stat = cispo_actor_loss_fn(
                logprobs=logprobs,
                proximal_logprobs=prox_logp,
                old_logprobs=old_logp,
                advantages=advantages,
                eps_clip=eps_clip,
                eps_clip_higher=eps_clip_higher,
                is_weight_clip_max=is_weight_clip_max,
                loss_mask=loss_mask,
                behav_imp_weight_cap=behav_imp_weight_cap,
                importance_sampling_level=importance_sampling_level,
                cu_seqlens=input_data.get("cu_seqlens"),
                loss_agg_mode=loss_agg_mode,
                rollout_is_weights=rollout_is_weights,
                dp_size=dp_size,
                batch_num_tokens=batch_num_tokens,
                global_batch_size=global_batch_size,
                prompt_group_ids=prompt_group_ids,
                prompt_token_counts=prompt_token_counts,
                sequence_loss_weights=sequence_loss_weights,
                nll_mask=nll_mask,
                ratio_masks=ratio_masks,
                ratio_m2_keep=ratio_m2_keep,
            )
        else:
            loss, stat = ppo_actor_loss_fn(
                logprobs=logprobs,
                old_logprobs=old_logp,
                advantages=advantages,
                eps_clip=eps_clip,
                eps_clip_higher=eps_clip_higher,
                loss_mask=loss_mask,
                c_clip=c_clip,
                proximal_logprobs=prox_logp,
                behav_imp_weight_cap=behav_imp_weight_cap,
                importance_sampling_level=importance_sampling_level,
                cu_seqlens=input_data.get("cu_seqlens"),
                loss_agg_mode=loss_agg_mode,
                rollout_is_weights=rollout_is_weights,
                dp_size=dp_size,
                batch_num_tokens=batch_num_tokens,
                global_batch_size=global_batch_size,
                prompt_group_ids=prompt_group_ids,
                prompt_token_counts=prompt_token_counts,
                sequence_loss_weights=sequence_loss_weights,
            )

        if entropy_coeff != 0.0:
            entropy_loss = agg_loss(
                -entropy.float(),
                loss_mask,
                loss_agg_mode=loss_agg_mode,
                dp_size=dp_size,
                batch_num_tokens=batch_num_tokens,
                global_batch_size=global_batch_size,
                prompt_group_ids=prompt_group_ids,
                prompt_token_counts=prompt_token_counts,
                sequence_loss_weights=sequence_loss_weights,
                cu_seqlens=input_data.get("cu_seqlens"),
            )
            loss = loss + entropy_coeff * entropy_loss

        if use_kl_loss:
            ref_logprobs = input_data.get("ref_log_probs")
            if ref_logprobs is None:
                raise ValueError("use_kl_loss=True but 'ref_log_probs' not found in context.")
            kl = kl_penalty(logprob=logprobs, ref_logprob=ref_logprobs.to(logprobs.device), method=kl_loss_type)
            kl_loss = agg_loss(
                kl,
                loss_mask,
                loss_agg_mode=loss_agg_mode,
                dp_size=dp_size,
                batch_num_tokens=batch_num_tokens,
                global_batch_size=global_batch_size,
                prompt_group_ids=prompt_group_ids,
                prompt_token_counts=prompt_token_counts,
                sequence_loss_weights=sequence_loss_weights,
                cu_seqlens=input_data.get("cu_seqlens"),
            )
            loss = loss + kl_loss_coef * kl_loss

        stats_mask = loss_mask if nll_mask is None else loss_mask & ~nll_mask
        metrics = {
            "approx_kl": _masked_mean_float(stat["approx_kl"].detach(), stats_mask),
            "importance_weight": _masked_mean_float(stat["importance_weight"].detach(), stats_mask),
            "clip_ratio": _masked_mean_float(stat["clip_mask"].float(), stats_mask),
            "entropy": _masked_mean_float(entropy.float(), stats_mask),
        }
        metrics.update(_source_statistics(stat, stats_mask))
        metrics["grpo_entropy_sum"] = float(torch.where(stats_mask, entropy.double(), 0.0).sum())
        metrics.update(stat.get("ratio_mask_metrics", {}))
        if nll_mask is not None:
            metrics.update(
                nll_trainable_token_count=float(nll_mask.sum()),
                nll_sum=float(torch.where(nll_mask, -logprobs.detach().double(), 0.0).sum()),
            )

    metrics.update(teacher_metrics)

    # ECHO auxiliary Environment-Prediction objective (arXiv 2605.24517):
    # total = rl_loss + aux_ce_weight * mean-of-per-sequence env CE.
    #
    # The PRESENCE of ``aux_ce_weight`` in the config is the switch. A config
    # without the key is bit-for-bit the pre-ECHO objective (no aux term, no
    # ECHO metrics). An explicit ``aux_ce_weight`` — including 0.0 — enables
    # the full ECHO contract: masks, count, and denominator mode are
    # validated and the ECHO metrics are always emitted, so a zero-weight
    # control run (and any client) can verify from the response that the
    # server honored the config instead of silently ignoring it (a server
    # without ECHO support returns no ECHO metrics).
    if aux_ce_weight is not None:
        if (
            isinstance(aux_ce_weight, bool)
            or not isinstance(aux_ce_weight, (int, float))
            or not math.isfinite(aux_ce_weight)
            or aux_ce_weight < 0.0
        ):
            raise ValueError(f"aux_ce_weight must be a finite non-negative number, got {aux_ce_weight!r}")
        aux_ce_weight = float(aux_ce_weight)
        sft_mask = input_data.get("sft_mask")
        observation_mask = input_data.get("echo_observation_mask")
        if sft_mask is None or observation_mask is None:
            raise ValueError(
                "ECHO (aux_ce_weight in config) requires 'sft_mask' and 'echo_observation_mask' "
                "in context — refusing to silently train without the ECHO auxiliary term."
            )
        if echo_global_num_sequences is None:
            raise ValueError(
                "ECHO (aux_ce_weight in config) requires config 'echo_global_num_sequences' — the "
                "client must supply the step-global sequence count; a local fallback would rescale "
                "the auxiliary gradient with microbatch/chunk boundaries."
            )
        # The aux term uses the same DP compensation as the policy term so
        # the echo-to-policy ratio equals the configured aux_ce_weight.
        echo_dp_multiplier = dp_loss_multiplier(loss_agg_mode, sequence_loss_weights, dp_size)
        env_loss, env_stat = echo_env_prediction_loss_fn(
            logprobs=logprobs,
            sft_mask=sft_mask,
            observation_mask=observation_mask,
            loss_mask=policy_loss_mask,
            global_num_echo_sequences=echo_global_num_sequences,
            batch_denominator=echo_batch_denominator,
            cu_seqlens=input_data.get("cu_seqlens"),
            dp_size=echo_dp_multiplier,
            observation_token_counts=input_data.get("echo_observation_token_counts"),
        )
        aux_loss = aux_ce_weight * env_loss
        if input_data.get("echo_observation_token_counts") is not None:
            metrics["echo_full_observation_denominator"] = 1.0
        metrics.update(
            {
                # Additive contributions and counts — named loss_term_* / *_sum /
                # *_count so microbatch and DP-worker reduction SUMS them (see
                # pipeline.metric_is_summed); means and fractions are exactly
                # derivable from the sums. The real/bearing sequence counts let a
                # client reconcile its declared step-global denominator against
                # the step's accumulated response totals before calling /step.
                "loss_term_rl": float(loss.detach()),
                "loss_term_aux": float(aux_loss.detach()),
                # The unweighted environment objective that λ multiplies. With the
                # constant labels below, the paper-unit coefficient is PROVABLE
                # from the response at any reduction level:
                # loss_term_aux == echo_aux_ce_weight * echo_environment_loss_sum.
                "echo_environment_loss_sum": float(env_loss.detach()),
                "echo_environment_prediction_nll_sum": float(env_stat["prediction_nll_sum"]),
                "echo_environment_prediction_token_count": float(env_stat["prediction_token_count"]),
                "echo_environment_observation_token_count": float(env_stat["observation_token_count"]),
                "echo_real_sequence_count": float(env_stat["num_real_sequences"]),
                "echo_observation_bearing_sequence_count": float(env_stat["num_echo_bearing_sequences"]),
                # Constant labels — identical in every microbatch, so the
                # weighted-mean reduction reproduces them exactly.
                "echo_contract_version": 1.0,
                "echo_aux_ce_weight": float(aux_ce_weight),
                "echo_dp_loss_multiplier": float(echo_dp_multiplier),
                "echo_global_num_sequences": float(echo_global_num_sequences),
                "echo_batch_denominator_is_echo_bearing": float(
                    env_stat["batch_denominator"] is EchoBatchDenominator.ECHO_BEARING_SEQUENCES
                ),
            }
        )
        if aux_ce_weight > 0.0:
            loss = loss + aux_loss

    return loss, metrics


# ---------------------------------------------------------------------------
# GRPO config contract
# ---------------------------------------------------------------------------
# One table per contract, ``key -> default``. Both the kwargs handed to
# :func:`_internal_grpo_loss_fn` and the strict ``*_echo_v1`` key schema are
# derived from these tables, so the accepted keys and their defaults cannot
# drift apart. Processor configs arrive as free-form dicts from external
# clients (verl, SkyRL, TRL, the Cortex zone), and a key present with an
# explicit ``null`` keeps ``None`` rather than falling back to the default —
# the math reads ``None`` as "feature off", and changing that would alter
# validated runs.
_GRPO_CONFIG_DEFAULTS: dict[str, Any] = {
    "eps_clip": 0.2,
    "eps_clip_higher": None,
    "c_clip": None,
    "behav_imp_weight_cap": None,
    "m2_threshold": None,
    "importance_sampling_level": "token",
    "current_version": None,
    "prox_logp_method": PROX_LOGP_METHOD_RECOMPUTE,
    "use_sapo_loss": False,
    "sapo_tau_pos": 1.0,
    "sapo_tau_neg": 1.05,
    "use_decoupled_loss": False,
    "use_cispo_loss": False,
    "is_weight_clip_max": None,
    "teacher_tau": 0.0,
    "teacher_clip": None,
    "teacher_clip_negative": None,
    "loss_agg_mode": "token-mean",
    # Unset dp_size means "not supplied"; _resolve_dp_size maps it to 1.
    "dp_size": None,
    "batch_num_tokens": None,
    "global_batch_size": None,
    "entropy_coeff": 0.0,
    "use_kl_loss": False,
    "kl_loss_coef": 0.001,
    "kl_loss_type": "low_var_kl",
}
_ECHO_CONFIG_DEFAULTS: dict[str, Any] = {
    "aux_ce_weight": None,
    "echo_global_num_sequences": None,
    "echo_batch_denominator": EchoBatchDenominator.ALL_SEQUENCES.value,
}
_GRPO_CONFIG_KEYS = frozenset(_GRPO_CONFIG_DEFAULTS)
_ECHO_REQUIRED_CONFIG_KEYS = frozenset({"aux_ce_weight", "echo_global_num_sequences"})
_ECHO_CONFIG_KEYS = frozenset(_ECHO_CONFIG_DEFAULTS)


def _grpo_config_values(config: dict) -> dict:
    """Declared defaults overlaid with the recognized keys this call supplied."""
    values = {**_GRPO_CONFIG_DEFAULTS, **_ECHO_CONFIG_DEFAULTS}
    values.update((key, config[key]) for key in values.keys() & config.keys())
    return values


def _grpo_loss(
    model_outputs: dict,
    context: dict,
    config: dict,
    device: str,
) -> Tuple[torch.Tensor, dict]:
    """Canonical GRPO/PPO loss (shared implementation of ``grpo`` and ``grpo_echo_v1``).

    Supports the full AReaL feature set (M2PO, SAPO, prox logp methods,
    version staleness).  All async/off-policy fields in ``context`` are optional
    -- VERL or simpler clients can omit them.

    Expected ``model_outputs`` keys (after compute_logprobs post-processor):
        ``logprobs`` -- per-token log-probs ``[batch, seq]``

    Expected ``context`` keys:
        Required: ``old_log_probs_shifted`` (behavioral policy log-probs), ``advantages``, ``loss_mask``
        Optional (async): ``prox_logp_shifted``, ``versions``
        Optional (SAPO): ``cu_seqlens``
        Optional (ECHO, required when ``aux_ce_weight`` is set): ``sft_mask``
        (environment-prediction target tokens O') and ``echo_observation_mask``
        (full observation span O, a superset of O'), same shape and shifted
        alignment as ``loss_mask`` and disjoint from it.

    Supported ``config`` keys (all optional):
        ``eps_clip`` (default 0.2), ``eps_clip_higher``, ``c_clip``,
        ``behav_imp_weight_cap``, ``m2_threshold``,
        ``importance_sampling_level`` (default "token"),
        ``current_version``, ``prox_logp_method`` (default "recompute"),
        ``use_sapo_loss``, ``sapo_tau_pos``, ``sapo_tau_neg``,
        ``use_decoupled_loss``,
        ``use_cispo_loss`` (default False; mutually exclusive with ``use_sapo_loss``
        and with a non-None ``c_clip``. Paper recommends asymmetric
        ``eps_clip=0.2`` / ``eps_clip_higher=0.28``.),
        ``is_weight_clip_max`` (optional upper cap for CISPO importance weights),
        ``loss_agg_mode`` (default "token-mean"; also "seq-mean-token-sum",
        "seq-mean-token-sum-norm", "seq-mean-token-mean", "prompt-mean"),
        ``dp_size``, ``batch_num_tokens``, ``global_batch_size`` (distributed normalisation),
        ``rollout_is_weights`` (off-policy correction tensor; send on ``batch``
        so DP shards it with advantages — ``meta`` is replicated),
        ``entropy_coeff`` (default 0.0; subtract entropy bonus from loss),
        ``use_kl_loss`` (default False; add KL penalty vs ``ref_log_probs`` in context),
        ``kl_loss_coef`` (default 0.001), ``kl_loss_type`` (default "low_var_kl"),
        ``aux_ce_weight`` (λ of the ECHO Environment-Prediction auxiliary
        loss, arXiv 2605.24517, IN PAPER UNITS: set it exactly as the paper's
        λ — the implementation compensates for the aggregation mode's
        distributed-reduction convention so the effective echo-to-policy
        ratio equals the configured value at every DP width and microbatch
        split; never pre-scale it. Adds
        ``aux_ce_weight * Σ_seq(Σ_{t∈O'} NLL_t / |O|) / echo_global_num_sequences``
        on top of the policy loss, scaled by the same DP factor as the policy
        term. The KEY'S PRESENCE enables the ECHO contract — omitted means
        bit-for-bit the pre-ECHO objective; an explicit 0.0 is a control run:
        no aux term, but masks/count validated and ECHO metrics emitted as
        proof the server honored the config. Reachable only through
        ``loss_fn="grpo_echo_v1"``, which enforces the strict key schema),
        ``echo_global_num_sequences`` (required with ``aux_ce_weight``: the
        client-computed step-global sequence count the auxiliary term is
        averaged over — the server cannot derive it from one call's slice),
        ``echo_batch_denominator`` (default ``"all_sequences"``, the
        paper-literal batch mean over every rollout in the step; the
        ``"echo_bearing_sequences"`` variant counts only sequences carrying
        observation tokens — see :class:`EchoBatchDenominator`. Declares how
        the client computed ``echo_global_num_sequences``; the count is
        validated against this call's slice and labeled in the metrics)

    Optional ``context`` key ``prompt_group_ids`` (Tensor[B] of ints) is
    required when ``loss_agg_mode="prompt-mean"``; one id per sequence,
    identical for all responses to the same prompt. Under DP, the number of
    global prompts is inferred via allreduce.
    """
    values = _grpo_config_values(config)
    _validate_loss_denominators(values["batch_num_tokens"], values["global_batch_size"])
    values["teacher_tau"] = _resolve_teacher_tau(config, context)
    values["ratio_masks"] = RatioMasks.from_config(config)
    values["dp_size"] = _resolve_dp_size(values["dp_size"], values["batch_num_tokens"])

    logprobs = model_outputs.get("logprobs")
    if logprobs is None:
        logits = model_outputs["logits"]
        input_ids = context["input_ids"].to(logits.device)
        if input_ids.ndim < logits.ndim:
            input_ids = input_ids.view(logits.shape[:-1])
        labels = torch.roll(input_ids, shifts=-1, dims=-1)
        logprobs = torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    cu_seqlens = context.get("cu_seqlens")
    if cu_seqlens is not None:
        # Canonicalize the packed layout ONCE at the loss entry. pack_sequences
        # emits every per-token tensor as singleton [1, T] and the
        # already-packed run_pipeline path forwards them unsqueezed, while the
        # microbatching path delivers 1-D [T] context tensors next to [1, T]
        # model logprobs. Downstream consumers branch on tensor rank (the
        # sequence-level ratio path treats 2-D input as padded rows and would
        # silently collapse a packed call into one sequence), so exactly one
        # packed representation — 1-D [T] — may exist past this point.
        logprobs = _packed_singleton_to_1d(logprobs)

    # Degenerate-shard logprob sanitization (see _internal_grpo_loss_fn). Under
    # Ulysses SP a short/padded batch can hand a rank a tail shard whose "valid"
    # positions have no attendable keys, so attention emits non-finite logits ->
    # non-finite logprobs. A single NaN poisons both the loss and the entropy
    # metric ("result.metrics.entropy is non-finite"). Zero them here, before
    # entropy is derived, so downstream loss + metrics stay finite; nan_to_num
    # yields zero gradient at those positions. No-op at real training seqlens.
    teacher_policy_finite = torch.isfinite(logprobs)
    nll_mask = context.get("nll_mask")
    if nll_mask is not None:
        nll_mask = nll_mask.to(logprobs.device)
        if cu_seqlens is not None:
            nll_mask = _packed_singleton_to_1d(nll_mask)
        active_nll_mask = canonicalize_loss_mask(
            nll_mask,
            logprobs,
            objective="nll_mask",
            binary=True,
        )
        if ((~teacher_policy_finite) & active_nll_mask).any().item():
            raise ValueError("logprobs must be finite at every active nll_mask position")
    if not teacher_policy_finite.all():
        logprobs = torch.nan_to_num(logprobs, nan=0.0, posinf=0.0, neginf=0.0)

    entropy = -logprobs.detach()

    old_log_probs_ctx = context.get("old_log_probs_shifted")
    input_data = {
        "old_log_probs": old_log_probs_ctx.to(logprobs.device) if old_log_probs_ctx is not None else logprobs.detach(),
        "advantages": context["advantages"].to(logprobs.device),
        "loss_mask": context["loss_mask"].to(logprobs.device),
        "prox_logp": context.get("prox_logp_shifted"),
        "versions": context.get("versions"),
        "cu_seqlens": cu_seqlens,
        "ref_log_probs": context.get("ref_log_probs_shifted"),
        "nll_mask": nll_mask,
        "teacher_log_probs": context.get("teacher_log_probs_shifted"),
        "teacher_policy_finite": teacher_policy_finite,
        "sft_mask": context.get("sft_mask"),
        "echo_observation_mask": context.get("echo_observation_mask"),
        "echo_observation_token_counts": context.get("echo_observation_token_counts"),
        "labels": context.get("labels"),
    }
    if input_data["sft_mask"] is not None:
        input_data["sft_mask"] = input_data["sft_mask"].to(logprobs.device)
    if input_data["echo_observation_mask"] is not None:
        input_data["echo_observation_mask"] = input_data["echo_observation_mask"].to(logprobs.device)
    if input_data["echo_observation_token_counts"] is not None:
        input_data["echo_observation_token_counts"] = input_data["echo_observation_token_counts"].to(logprobs.device)
    if input_data["prox_logp"] is not None:
        input_data["prox_logp"] = input_data["prox_logp"].to(logprobs.device)
    if input_data["versions"] is not None:
        input_data["versions"] = input_data["versions"].to(logprobs.device)
    if input_data["ref_log_probs"] is not None:
        input_data["ref_log_probs"] = input_data["ref_log_probs"].to(logprobs.device)
    if input_data["teacher_log_probs"] is not None:
        if not torch.is_tensor(input_data["teacher_log_probs"]):
            raise ValueError("teacher_log_probs_shifted must be a floating-point tensor")
        input_data["teacher_log_probs"] = input_data["teacher_log_probs"].to(logprobs.device)

    rollout_is_weights = context.get("rollout_is_weights")
    if rollout_is_weights is not None:
        rollout_is_weights = rollout_is_weights.to(logprobs.device)

    if cu_seqlens is not None:
        # Per-token tensors only — per-sequence vectors (prompt_group_ids,
        # prompt_token_counts, sequence_loss_weights) are 1-D already.
        for key in (
            "old_log_probs",
            "advantages",
            "loss_mask",
            "prox_logp",
            "versions",
            "ref_log_probs",
            "nll_mask",
            "teacher_log_probs",
            "teacher_policy_finite",
            "sft_mask",
            "echo_observation_mask",
            "labels",
        ):
            input_data[key] = _packed_singleton_to_1d(input_data[key])
        rollout_is_weights = _packed_singleton_to_1d(rollout_is_weights)

    if values["teacher_tau"] > 0.0:
        teacher_log_probs = input_data["teacher_log_probs"]
        if not torch.is_tensor(teacher_log_probs) or not teacher_log_probs.is_floating_point():
            raise ValueError("teacher_log_probs_shifted must be a floating-point tensor")
        if teacher_log_probs.shape != logprobs.shape:
            raise ValueError(
                "teacher_log_probs_shifted must exactly match prediction-aligned logprobs shape, "
                f"got {tuple(teacher_log_probs.shape)} and {tuple(logprobs.shape)}"
            )

    prompt_group_ids = context.get("prompt_group_ids")
    if prompt_group_ids is not None:
        prompt_group_ids = prompt_group_ids.to(logprobs.device)

    prompt_token_counts = context.get("prompt_token_counts")
    if prompt_token_counts is not None:
        prompt_token_counts = prompt_token_counts.to(logprobs.device)

    sequence_loss_weights = context.get("sequence_loss_weights")
    if sequence_loss_weights is not None:
        sequence_loss_weights = sequence_loss_weights.to(logprobs.device)

    # Every key of the config tables is a keyword of _internal_grpo_loss_fn
    # under the same name; the tensors below are the only non-config arguments.
    loss, metrics = _internal_grpo_loss_fn(
        logprobs=logprobs,
        entropy=entropy,
        input_data=input_data,
        rollout_is_weights=rollout_is_weights,
        prompt_group_ids=prompt_group_ids,
        prompt_token_counts=prompt_token_counts,
        sequence_loss_weights=sequence_loss_weights,
        **values,
    )
    return loss, metrics


# Objective-term contributions and token/sequence counts the ECHO path emits.
# They are additive across packed microbatches, gradient accumulation, and DP
# ranks, so the reducers must sum them rather than average them. Declared here
# so the contract is a property of the registration, not of the metric name.
ECHO_SUMMED_METRICS = frozenset(
    {
        "loss_term_rl",
        "loss_term_aux",
        "echo_environment_loss_sum",
        "echo_environment_prediction_nll_sum",
        "echo_environment_prediction_token_count",
        "echo_environment_observation_token_count",
        "echo_real_sequence_count",
        "echo_observation_bearing_sequence_count",
    }
)


def _grpo_model_call_count_callback(model_call_counts: Sequence[int | None], config: dict) -> None:
    if config.get("ratio_m2_threshold") is None:
        return
    counts = tuple(model_call_counts)
    if set(counts) != {1}:
        raise ValueError(f"ratio_m2_threshold requires exactly one synchronized model call per worker, got {counts!r}")


def _grpo_preflight_mask(microbatch: dict) -> torch.Tensor:
    reference = microbatch.get("input_ids")
    if not torch.is_tensor(reference):
        raise ValueError("grpo packed microbatches require tensor input_ids for preflight validation")
    loss_mask = microbatch.get("loss_mask")
    if loss_mask is None:
        raise ValueError("grpo requires context['loss_mask']")
    mask = canonicalize_loss_mask(
        loss_mask,
        reference,
        objective="grpo",
        binary=True,
    )
    labels = microbatch.get("labels")
    if torch.is_tensor(labels):
        if tuple(labels.shape) != tuple(mask.shape):
            raise ValueError("grpo labels must match loss_mask when labels are present")
        if ((labels.to(mask.device) == -100) & mask).any().item():
            raise ValueError("grpo loss_mask must be zero where labels use ignore_index=-100")
    return mask


def _active_sequence_count(microbatch: dict, loss_mask: torch.Tensor) -> float:
    cu_seqlens = microbatch.get("cu_seqlens")
    if torch.is_tensor(cu_seqlens):
        flat_mask = loss_mask.reshape(-1)
        boundaries = cu_seqlens.detach().cpu().tolist()
        return float(sum(bool(flat_mask[start:end].any().item()) for start, end in pairwise(boundaries)))
    if loss_mask.ndim >= 2:
        return float(loss_mask.reshape(loss_mask.shape[0], -1).any(dim=1).sum().item())
    return float(bool(loss_mask.any().item()))


def _local_grpo_packed_loss_reduction(
    microbatches: Sequence[dict],
    config: dict,
    loss_fn_name: str,
    *,
    mixed: bool = False,
    echo: bool = False,
) -> PackedLossReduction:
    masks = [_grpo_preflight_mask(microbatch) for microbatch in microbatches]
    ratio_masks = RatioMasks.from_config(config)
    if ratio_masks is not None and not config.get("use_cispo_loss"):
        raise ValueError("ratio-mask keys act on the CISPO policy term; they need use_cispo_loss=True.")
    for microbatch, mask in zip(microbatches, masks):
        if mixed:
            _validate_mixed_config(microbatch, config)
            _validate_nll_mask(microbatch["nll_mask"], mask)
        elif "nll_mask" in microbatch:
            leaf_name = loss_fn_name.rsplit(".", 1)[-1]
            is_cortex = loss_fn_name.startswith("grpo") or leaf_name.startswith("cortex_grpo")
            mixed_name = "grpo_mixed_v1" if is_cortex else "ap_grpo_mixed_v1"
            raise ValueError(f"nll_mask requires loss_fn='{mixed_name}'")
        _resolve_teacher_tau(config, microbatch)
    mode = _grpo_config_values(config)["loss_agg_mode"]

    if mode == "token-mean":
        weights = [float(mask.sum().item()) for mask in masks]
        reduction = (
            additive_packed_loss_reduction(weights)
            if config.get("batch_num_tokens") is not None
            else local_mean_packed_loss_reduction(weights)
        )
    elif mode in ("seq-mean-token-sum", "seq-mean-token-mean"):
        weights = [_active_sequence_count(microbatch, mask) for microbatch, mask in zip(microbatches, masks)]
        reduction = (
            additive_packed_loss_reduction(weights)
            if config.get("global_batch_size") is not None
            else local_mean_packed_loss_reduction(weights)
        )
    elif mode == "seq-mean-token-sum-norm":
        if len(microbatches) > 1:
            raise ValueError(
                "loss_agg_mode='seq-mean-token-sum-norm' does not declare a "
                "split-invariant packed reduction; use one microbatch"
            )
        weights = [_active_sequence_count(microbatches[0], masks[0])]
        reduction = (
            additive_packed_loss_reduction(weights)
            if config.get("global_batch_size") is not None
            else local_mean_packed_loss_reduction(weights)
        )
    elif mode == "prompt-mean":
        has_sequence_weights = all(
            torch.is_tensor(microbatch.get("sequence_loss_weights")) for microbatch in microbatches
        )
        if has_sequence_weights:
            weights = [float(microbatch["sequence_loss_weights"].abs().sum().item()) for microbatch in microbatches]
            reduction = additive_packed_loss_reduction(weights)
        elif len(microbatches) > 1:
            raise ValueError(
                "loss_agg_mode='prompt-mean' without sequence_loss_weights does "
                "not support multiple packed microbatches because responses for "
                "one prompt may be split across model calls"
            )
        else:
            reduction = local_mean_packed_loss_reduction((1.0,))
    else:
        raise ValueError(
            f"Invalid loss_agg_mode: {mode!r}; packed reduction metadata is "
            "available only for registered GRPO aggregation modes"
        )

    if echo and len(microbatches) > 1 and not reduction.loss_is_additive:
        raise ValueError(
            f"loss_fn {loss_fn_name!r} requires a globally normalized additive "
            "policy objective when split into multiple packed microbatches"
        )
    return reduction


def _raise_synchronized_validation_error(error: Exception | None, reference: torch.Tensor) -> None:
    group = _get_sequence_parallel_group()
    if group is not None and torch.distributed.is_initialized():
        failed = torch.tensor(error is not None, dtype=torch.int32, device=reference.device)
        torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX, group=group)
        if failed.item() and error is None:
            raise ValueError("GRPO validation failed on another sequence-parallel rank")
    if error is not None:
        raise error


def _grpo_packed_loss_reduction(
    microbatches: Sequence[dict],
    config: dict,
    loss_fn_name: str,
    *,
    mixed: bool = False,
    echo: bool = False,
) -> PackedLossReduction:
    error = None
    reduction = None
    try:
        reduction = _local_grpo_packed_loss_reduction(
            microbatches,
            config,
            loss_fn_name,
            mixed=mixed,
            echo=echo,
        )
    except ValueError as caught:
        error = caught
    reference = next(
        (microbatch["input_ids"] for microbatch in microbatches if torch.is_tensor(microbatch.get("input_ids"))),
        None,
    )
    if reference is None:
        if error is not None:
            raise error
        raise ValueError("grpo packed microbatches require tensor input_ids for synchronized validation")
    _raise_synchronized_validation_error(error, reference)
    assert reduction is not None
    return reduction


def _grpo_mixed_packed_loss_reduction(
    microbatches: Sequence[dict],
    config: dict,
    loss_fn_name: str,
) -> PackedLossReduction:
    return _grpo_packed_loss_reduction(
        microbatches,
        config,
        loss_fn_name,
        mixed=True,
    )


def _grpo_echo_packed_loss_reduction(
    microbatches: Sequence[dict],
    config: dict,
    loss_fn_name: str,
) -> PackedLossReduction:
    return _grpo_packed_loss_reduction(
        microbatches,
        config,
        loss_fn_name,
        echo=True,
    )


def _merge_distributed_config(config: dict, batch: dict, meta: dict) -> dict:
    """Fill missing scale keys from meta/batch; config wins when set.

    Stamped ``dp_size`` is routing metadata. It becomes a loss scale only when
    a global denominator is also present (``batch_num_tokens`` for token-mean,
    ``global_batch_size`` for sequence-mean).
    """
    cfg = dict(config)
    for key in ("batch_num_tokens", "global_batch_size"):
        if cfg.get(key) is None:
            if meta.get(key) is not None:
                cfg[key] = meta[key]
            elif batch.get(key) is not None:
                cfg[key] = batch[key]
    uses_weighted_prompt_mean = (
        cfg.get("loss_agg_mode") == "prompt-mean"
        and _grpo_context(batch, meta).get("sequence_loss_weights") is not None
    )
    has_global_scale = (
        cfg.get("batch_num_tokens") is not None
        or cfg.get("global_batch_size") is not None
        or uses_weighted_prompt_mean
    )
    if cfg.get("dp_size") is None and has_global_scale:
        if meta.get("dp_size") is not None:
            cfg["dp_size"] = meta["dp_size"]
        elif batch.get("dp_size") is not None:
            cfg["dp_size"] = batch["dp_size"]
    return cfg


def _grpo_context(batch: dict, meta: dict) -> dict:
    """Merge bags with batch winning (sharded tensors over replicated meta)."""
    return {**meta, **batch}


def _request_grpo_contexts(request: dict) -> list[dict]:
    if "kwargs" in request:
        return [{**(request.get("context") or {}), **(request.get("kwargs") or {})}]
    if "batch" in request:
        meta = request.get("meta") or {}
        batch = request["batch"]
        if isinstance(batch, list):
            return [{**meta, **microbatch} for microbatch in batch]
        return [{**meta, **batch}]
    context = {key: value for key, value in request.items() if key not in {"context", "processing"}}
    context.update(request.get("context") or {})
    return [context]


def _request_grpo_config(request: dict) -> dict:
    processing = request.get("processing")
    if not isinstance(processing, dict):
        raise ValueError("GRPO requests require a processing object")
    config = processing.get("config", {})
    if not isinstance(config, dict):
        raise ValueError("GRPO processing.config must be an object")
    return config


def _validate_grpo_context(context: dict, config: dict) -> torch.Tensor:
    _validate_loss_denominators(config.get("batch_num_tokens"), config.get("global_batch_size"))
    mask = _grpo_preflight_mask(context)
    _resolve_teacher_tau(config, context)
    RatioMasks.from_config(config)
    return mask


def _validate_plain_grpo_context(context: dict, config: dict) -> None:
    _validate_grpo_context(context, config)
    if "nll_mask" in context:
        raise ValueError("nll_mask requires a mixed GRPO loss")
    echo_keys = _ECHO_CONFIG_KEYS & set(config)
    if echo_keys:
        raise ValueError(f"plain GRPO does not accept ECHO config keys {sorted(echo_keys)}")


def _run_synchronized_grpo_validation(context: dict, config: dict, validator) -> None:
    error = None
    try:
        validator(context, config)
    except ValueError as caught:
        error = caught
    reference = context.get("input_ids")
    if not torch.is_tensor(reference):
        if error is not None:
            raise error
        raise ValueError("GRPO validation requires tensor input_ids")
    _raise_synchronized_validation_error(error, reference)


def _grpo_batching_callback(request: dict) -> None:
    config = _request_grpo_config(request)
    for context in _request_grpo_contexts(request):
        _validate_plain_grpo_context(context, config)


def _grpo_validation_callback(context: dict, config: dict) -> None:
    _run_synchronized_grpo_validation(context, config, _validate_plain_grpo_context)


@register_loss_fn(
    "ap_grpo",
    batching_callback=_grpo_batching_callback,
    validation_callback=_grpo_validation_callback,
    packed_loss_reduction=_grpo_packed_loss_reduction,
    model_call_count_callback=_grpo_model_call_count_callback,
    metrics_callback=_grpo_metrics_callback,
)
@declare_loss_capabilities(REQUIRES_ALIGNED_TOKEN_LOGPROBS, PRESERVES_EXPLICIT_LOSS_SCALE_CONFIG)
def grpo_loss(
    model_outputs: dict,
    batch: dict,
    meta: dict,
    config: dict,
    device: str,
) -> Tuple[torch.Tensor, dict]:
    """Plain GRPO/PPO contract — see :func:`_grpo_loss` for the full key set.

    ECHO keys are rejected here on purpose: ``grpo`` keeps its historical
    permissive config parsing (unknown keys ignored), under which a
    misspelled ECHO key or an ECHO-unaware server would silently train the
    baseline objective. ECHO therefore only activates through the strict,
    versioned ``grpo_echo_v1`` contract below.
    """
    config = _merge_distributed_config(config, batch, meta)
    if "nll_mask" in _grpo_context(batch, meta):
        raise ValueError("nll_mask requires loss_fn='ap_grpo_mixed_v1'")
    echo_keys = _ECHO_CONFIG_KEYS & set(config)
    if echo_keys:
        raise ValueError(
            f"loss_fn 'ap_grpo' does not accept ECHO config keys {sorted(echo_keys)} — request "
            "loss_fn 'ap_grpo_echo_v1', whose strict schema fails loudly on typos and on servers "
            "without ECHO support."
        )
    return _grpo_loss(model_outputs, _grpo_context(batch, meta), config, device)


def _validate_mixed_config(context: dict, config: dict) -> None:
    ratio_keys = RATIO_MASK_CONFIG_KEYS & config.keys()
    if ratio_keys:
        raise ValueError(f"grpo_mixed_v1 does not support ratio-mask keys: {sorted(ratio_keys)}")
    unknown = set(config) - _GRPO_CONFIG_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys for grpo_mixed_v1: {sorted(unknown)}")
    if context.get("nll_mask") is None:
        raise ValueError("grpo_mixed_v1 requires nll_mask")
    cap = config.get("is_weight_clip_max")
    if (
        config.get("use_cispo_loss") is not True
        or cap is None
        or isinstance(cap, bool)
        or not isinstance(cap, (int, float))
        or not math.isfinite(cap)
        or cap <= 0
    ):
        raise ValueError("grpo_mixed_v1 requires CISPO and a finite positive is_weight_clip_max")
    if config.get("importance_sampling_level", "token") != "token":
        raise ValueError("grpo_mixed_v1 requires token importance sampling")
    entropy_coeff = config.get("entropy_coeff", 0.0)
    invalid_entropy_coeff = (
        entropy_coeff is None
        or isinstance(entropy_coeff, bool)
        or not isinstance(entropy_coeff, (int, float))
        or not math.isfinite(entropy_coeff)
        or entropy_coeff != 0.0
    )
    unsupported = ("use_sapo_loss", "use_decoupled_loss", "use_kl_loss")
    if (
        any(config.get(key) for key in unsupported)
        or invalid_entropy_coeff
        or config.get("m2_threshold") is not None
        or config.get("c_clip") is not None
        or config.get("behav_imp_weight_cap") is not None
        or config.get("current_version") is not None
        or config.get("prox_logp_method", PROX_LOGP_METHOD_RECOMPUTE) != PROX_LOGP_METHOD_RECOMPUTE
        or context.get("prox_logp_shifted") is not None
        or context.get("rollout_is_weights") is not None
    ):
        raise ValueError("grpo_mixed_v1 does not support SAPO, decoupled/M2PO/RIS, reference KL or entropy")


def _validate_mixed_grpo_context(context: dict, config: dict) -> None:
    loss_mask = _validate_grpo_context(context, config)
    _validate_mixed_config(context, config)
    _validate_nll_mask(context["nll_mask"], loss_mask)


def _grpo_mixed_batching_callback(request: dict) -> None:
    config = _request_grpo_config(request)
    for context in _request_grpo_contexts(request):
        _validate_mixed_grpo_context(context, config)


def _grpo_mixed_validation_callback(context: dict, config: dict) -> None:
    _run_synchronized_grpo_validation(context, config, _validate_mixed_grpo_context)


def _validate_echo_config(config: dict, loss_fn_name: str) -> None:
    unknown_keys = set(config) - _GRPO_CONFIG_KEYS - _ECHO_CONFIG_KEYS - RATIO_MASK_CONFIG_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown config keys for loss_fn {loss_fn_name!r}: {sorted(unknown_keys)}")
    missing_keys = {key for key in _ECHO_REQUIRED_CONFIG_KEYS if config.get(key) is None}
    if missing_keys:
        raise ValueError(f"loss_fn {loss_fn_name!r} requires non-None config keys {sorted(missing_keys)}")
    aux_ce_weight = config["aux_ce_weight"]
    if (
        isinstance(aux_ce_weight, bool)
        or not isinstance(aux_ce_weight, (int, float))
        or not math.isfinite(aux_ce_weight)
        or aux_ce_weight < 0
    ):
        raise ValueError(f"aux_ce_weight must be a finite non-negative number, got {aux_ce_weight!r}")
    global_num_sequences = config["echo_global_num_sequences"]
    if isinstance(global_num_sequences, bool) or not isinstance(global_num_sequences, int) or global_num_sequences < 1:
        raise ValueError(f"echo_global_num_sequences must be a positive integer, got {global_num_sequences!r}")
    try:
        EchoBatchDenominator(config.get("echo_batch_denominator", EchoBatchDenominator.ALL_SEQUENCES.value))
    except ValueError:
        raise ValueError(f"Invalid echo_batch_denominator: {config.get('echo_batch_denominator')!r}") from None


def _validate_echo_grpo_context(context: dict, config: dict) -> None:
    loss_mask = _validate_grpo_context(context, config)
    _validate_echo_config(config, "grpo_echo_v1")
    if "nll_mask" in context:
        raise ValueError("nll_mask requires a mixed GRPO loss")
    sft_mask = context.get("sft_mask")
    observation_mask = context.get("echo_observation_mask")
    if sft_mask is None or observation_mask is None:
        raise ValueError("ECHO requires sft_mask and echo_observation_mask")
    sft_mask = canonicalize_loss_mask(sft_mask, loss_mask, objective="sft_mask", binary=True)
    observation_mask = canonicalize_loss_mask(
        observation_mask,
        loss_mask,
        objective="echo_observation_mask",
        binary=True,
    )
    if (sft_mask & ~observation_mask).any().item():
        raise ValueError("sft_mask must be a subset of echo_observation_mask")
    if (sft_mask & loss_mask).any().item():
        raise ValueError("sft_mask overlaps loss_mask")
    if (observation_mask & loss_mask).any().item():
        raise ValueError("echo_observation_mask overlaps loss_mask")
    observation_token_counts = context.get("echo_observation_token_counts")
    if observation_token_counts is not None:
        cu_seqlens = context.get("cu_seqlens")
        if cu_seqlens is not None:
            _, (visible_observation_counts,) = _packed_per_sequence_sums(
                cu_seqlens,
                observation_mask.reshape(-1).to(torch.float32),
            )
        elif observation_mask.ndim == 1:
            visible_observation_counts = observation_mask.sum(dtype=torch.float32).reshape(1)
        else:
            visible_observation_counts = observation_mask.reshape(observation_mask.shape[0], -1).sum(
                dim=1,
                dtype=torch.float32,
            )
        _full_observation_denominator(observation_token_counts, visible_observation_counts)


def _grpo_echo_batching_callback(request: dict) -> None:
    config = _request_grpo_config(request)
    for context in _request_grpo_contexts(request):
        _validate_echo_grpo_context(context, config)


def _grpo_echo_validation_callback(context: dict, config: dict) -> None:
    _run_synchronized_grpo_validation(context, config, _validate_echo_grpo_context)


@register_loss_fn(
    "ap_grpo_mixed_v1",
    batching_callback=_grpo_mixed_batching_callback,
    validation_callback=_grpo_mixed_validation_callback,
    packed_loss_reduction=_grpo_mixed_packed_loss_reduction,
    model_call_count_callback=_grpo_model_call_count_callback,
    metrics_callback=_grpo_metrics_callback,
)
@declare_loss_capabilities(REQUIRES_ALIGNED_TOKEN_LOGPROBS, PRESERVES_EXPLICIT_LOSS_SCALE_CONFIG)
def grpo_mixed_v1_loss(
    model_outputs: dict,
    batch: dict,
    meta: dict,
    config: dict,
    device: str,
) -> Tuple[torch.Tensor, dict]:
    """CISPO on policy tokens and literal NLL on prediction-aligned ``nll_mask`` tokens."""
    config = _merge_distributed_config(config, batch, meta)
    context = _grpo_context(batch, meta)
    _validate_mixed_config(context, config)
    return _grpo_loss(model_outputs, context, config, device)


@register_loss_fn(
    "ap_grpo_echo_v1",
    batching_callback=_grpo_echo_batching_callback,
    validation_callback=_grpo_echo_validation_callback,
    packed_loss_reduction=_grpo_echo_packed_loss_reduction,
    model_call_count_callback=_grpo_model_call_count_callback,
    metrics_callback=_grpo_metrics_callback,
    summed_metrics=ECHO_SUMMED_METRICS,
)
@declare_loss_capabilities(REQUIRES_ALIGNED_TOKEN_LOGPROBS, PRESERVES_EXPLICIT_LOSS_SCALE_CONFIG)
def grpo_echo_v1_loss(
    model_outputs: dict,
    batch: dict,
    meta: dict,
    config: dict,
    device: str,
) -> Tuple[torch.Tensor, dict]:
    """Versioned GRPO + ECHO contract with a strict config schema.

    Unlike ``grpo`` (permissive by history), every config key must be known
    and the ECHO keys must be present and non-None — a typo like
    ``aux_ce_weigth``, a missing count, or an explicit ``null`` fails before
    any backward pass instead of silently training the baseline objective
    (the shared implementation treats ``aux_ce_weight=None`` as "ECHO not
    configured", so ``None`` must never survive this schema). Requesting
    this loss name on a server
    without ECHO support fails at loss-fn resolution, which is the
    capability handshake: a response carrying ``echo_contract_version`` (and
    the other ECHO metrics) proves the versioned contract executed.
    ``aux_ce_weight: 0.0`` is the control run: baseline loss bit-for-bit,
    full validation, full ECHO metrics.
    """
    config = _merge_distributed_config(config, batch, meta)
    context = _grpo_context(batch, meta)
    if "nll_mask" in context:
        raise ValueError("nll_mask requires loss_fn='ap_grpo_mixed_v1'")
    _validate_echo_config(config, "ap_grpo_echo_v1")
    return _grpo_loss(model_outputs, context, config, device)
