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

from enum import Enum
from typing import Tuple

import torch

from .functional import agg_loss
from .functional import kl_penalty
from .functional import ppo_actor_loss_fn
from .functional import sapo_loss_fn
from .pipeline import register_loss_fn

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
) -> torch.Tensor:
    """Internal GRPO loss — same interface as dss/loss_fns/grpo.py."""
    old_logp = input_data["old_log_probs"]
    advantages = input_data["advantages"]
    loss_mask = input_data["loss_mask"].bool()
    prox_logp_gt = input_data.get("prox_logp")
    entropy = entropy.detach()

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
        )

    # Optional entropy bonus: subtract entropy_coeff * mean_entropy from loss
    if entropy_coeff != 0.0:
        entropy_loss = agg_loss(
            -entropy.float(),
            loss_mask,
            loss_agg_mode=loss_agg_mode,
            dp_size=dp_size,
            batch_num_tokens=batch_num_tokens,
            global_batch_size=global_batch_size,
        )
        loss = loss + entropy_coeff * entropy_loss

    # Optional KL penalty against a reference policy (e.g. SFT model)
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
        )
        loss = loss + kl_loss_coef * kl_loss

    metrics = {
        "approx_kl": float((stat["approx_kl"].detach() * loss_mask).sum() / loss_mask.sum().clamp(min=1)),
        "importance_weight": float(
            (stat["importance_weight"].detach() * loss_mask).sum() / loss_mask.sum().clamp(min=1)
        ),
        "clip_ratio": float((stat["clip_mask"].float() * loss_mask).sum() / loss_mask.sum().clamp(min=1)),
        "entropy": float((entropy.float() * loss_mask).sum() / loss_mask.sum().clamp(min=1)),
        "loss": float(loss.detach().cpu().item()),
    }
    return loss, metrics


@register_loss_fn("grpo")
def grpo_loss(
    model_outputs: dict,
    context: dict,
    config: dict,
    device: str,
) -> Tuple[torch.Tensor, dict]:
    """Canonical GRPO/PPO loss.

    Supports the full AReaL feature set (M2PO, SAPO, prox logp methods,
    version staleness).  All async/off-policy fields in ``context`` are optional
    -- VERL or simpler clients can omit them.

    Expected ``model_outputs`` keys (after compute_logprobs post-processor):
        ``logprobs`` -- per-token log-probs ``[batch, seq]``

    Expected ``context`` keys:
        Required: ``old_log_probs_shifted`` (behavioral policy log-probs), ``advantages``, ``loss_mask``
        Optional (async): ``prox_logp_shifted``, ``versions``
        Optional (SAPO): ``cu_seqlens``

    Supported ``config`` keys (all optional):
        ``eps_clip`` (default 0.2), ``eps_clip_higher``, ``c_clip``,
        ``behav_imp_weight_cap``, ``m2_threshold``,
        ``importance_sampling_level`` (default "token"),
        ``current_version``, ``prox_logp_method`` (default "recompute"),
        ``use_sapo_loss``, ``sapo_tau_pos``, ``sapo_tau_neg``,
        ``use_decoupled_loss``,
        ``loss_agg_mode`` (default "token-mean"; also "seq-mean-token-sum",
        "seq-mean-token-sum-norm", "seq-mean-token-mean"),
        ``dp_size``, ``batch_num_tokens``, ``global_batch_size`` (distributed normalisation),
        ``rollout_is_weights`` (off-policy correction tensor, passed via context),
        ``entropy_coeff`` (default 0.0; subtract entropy bonus from loss),
        ``use_kl_loss`` (default False; add KL penalty vs ``ref_log_probs`` in context),
        ``kl_loss_coef`` (default 0.001), ``kl_loss_type`` (default "low_var_kl")
    """
    logprobs = model_outputs.get("logprobs")
    if logprobs is None:
        logits = model_outputs["logits"]
        input_ids = context["input_ids"].to(logits.device)
        if input_ids.ndim < logits.ndim:
            input_ids = input_ids.view(logits.shape[:-1])
        labels = torch.roll(input_ids, shifts=-1, dims=-1)
        logprobs = torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    entropy = -logprobs.detach()

    old_log_probs_ctx = context.get("old_log_probs_shifted")
    input_data = {
        "old_log_probs": old_log_probs_ctx.to(logprobs.device) if old_log_probs_ctx is not None else logprobs.detach(),
        "advantages": context["advantages"].to(logprobs.device),
        "loss_mask": context["loss_mask"].to(logprobs.device),
        "prox_logp": context.get("prox_logp_shifted"),
        "versions": context.get("versions"),
        "cu_seqlens": context.get("cu_seqlens"),
        "ref_log_probs": context.get("ref_log_probs_shifted"),
    }
    if input_data["prox_logp"] is not None:
        input_data["prox_logp"] = input_data["prox_logp"].to(logprobs.device)
    if input_data["versions"] is not None:
        input_data["versions"] = input_data["versions"].to(logprobs.device)
    if input_data["ref_log_probs"] is not None:
        input_data["ref_log_probs"] = input_data["ref_log_probs"].to(logprobs.device)

    rollout_is_weights = context.get("rollout_is_weights")
    if rollout_is_weights is not None:
        rollout_is_weights = rollout_is_weights.to(logprobs.device)

    loss, metrics = _internal_grpo_loss_fn(
        logprobs=logprobs,
        entropy=entropy,
        input_data=input_data,
        eps_clip=config.get("eps_clip", 0.2),
        eps_clip_higher=config.get("eps_clip_higher"),
        c_clip=config.get("c_clip"),
        behav_imp_weight_cap=config.get("behav_imp_weight_cap"),
        m2_threshold=config.get("m2_threshold"),
        importance_sampling_level=config.get("importance_sampling_level", "token"),
        current_version=config.get("current_version"),
        prox_logp_method=config.get("prox_logp_method", PROX_LOGP_METHOD_RECOMPUTE),
        use_sapo_loss=config.get("use_sapo_loss", False),
        sapo_tau_pos=config.get("sapo_tau_pos", 1.0),
        sapo_tau_neg=config.get("sapo_tau_neg", 1.05),
        use_decoupled_loss=config.get("use_decoupled_loss", False),
        loss_agg_mode=config.get("loss_agg_mode", "token-mean"),
        dp_size=config.get("dp_size", 1),
        batch_num_tokens=config.get("batch_num_tokens"),
        global_batch_size=config.get("global_batch_size"),
        rollout_is_weights=rollout_is_weights,
        entropy_coeff=config.get("entropy_coeff", 0.0),
        use_kl_loss=config.get("use_kl_loss", False),
        kl_loss_coef=config.get("kl_loss_coef", 0.001),
        kl_loss_type=config.get("kl_loss_type", "low_var_kl"),
    )
    return loss, metrics
