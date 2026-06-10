"""Functional math utilities for RL loss computation."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.distributed as dist


@torch.no_grad()
def masked_normalization(
    x: torch.Tensor,
    mask: torch.Tensor | None = None,
    dim=None,
    unbiased=False,
    eps=1e-5,
    high_precision=True,
    all_reduce=True,
    reduce_group=None,
):
    dtype = torch.float64 if high_precision else torch.float32
    x = x.to(dtype)
    if dim is None:
        dim = tuple(range(len(x.shape)))
    if mask is None:
        factor = torch.tensor(
            np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device
        )
    else:
        mask = mask.to(dtype)
        x = x * mask
        factor = mask.sum(dim, keepdim=True)
    x_sum = x.sum(dim=dim, keepdim=True)
    x_sum_sq = x.square().sum(dim=dim, keepdim=True)
    if dist.is_initialized() and all_reduce:
        dist.all_reduce(factor, op=dist.ReduceOp.SUM, group=reduce_group)
        dist.all_reduce(x_sum, op=dist.ReduceOp.SUM, group=reduce_group)
        dist.all_reduce(x_sum_sq, op=dist.ReduceOp.SUM, group=reduce_group)
    mean = x_sum / factor
    meansq = x_sum_sq / factor
    var = meansq - mean**2
    if unbiased:
        var *= factor / (factor - 1)
    return ((x - mean) / (var.sqrt() + eps)).float()


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
) -> torch.Tensor:
    """Aggregate a per-token loss matrix into a scalar.

    Supports four modes:
    - ``"token-mean"`` (default): sum over all valid tokens, divide by token count.
      Equivalent to the existing ``/ loss_mask_count`` behaviour.
    - ``"seq-mean-token-sum"``: sum tokens per sequence, then mean across sequences.
    - ``"seq-mean-token-sum-norm"``: same as above, additionally divided by
      ``loss_scale_factor`` (defaults to ``loss_mask.shape[-1]``).
    - ``"seq-mean-token-mean"``: mean tokens per sequence, then mean across sequences.

    ``dp_size``, ``batch_num_tokens``, and ``global_batch_size`` support
    distributed normalisation when the global batch is split across DP ranks.
    They default to local-only values when omitted (dp_size=1).
    """
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            batch_num_tokens = loss_mask.count_nonzero() or 1
        loss = (torch.where(loss_mask.bool(), loss_mat, 0.0).sum() / batch_num_tokens) * dp_size

    elif loss_agg_mode in ("seq-mean-token-sum", "seq-mean-token-sum-norm"):
        seq_losses = (loss_mat * loss_mask).sum(dim=-1)
        seq_mask = (loss_mask.sum(dim=-1) > 0).float()
        if global_batch_size is None:
            global_batch_size = seq_mask.sum().clamp(min=1)
        loss = ((seq_losses * seq_mask).sum() / global_batch_size) * dp_size
        if loss_agg_mode == "seq-mean-token-sum-norm":
            if loss_scale_factor is None:
                loss_scale_factor = loss_mask.shape[-1]
            loss = loss / loss_scale_factor

    elif loss_agg_mode == "seq-mean-token-mean":
        seq_token_counts = loss_mask.sum(dim=-1).clamp(min=1).float()
        seq_losses = (loss_mat * loss_mask).sum(dim=-1) / seq_token_counts
        seq_mask = (loss_mask.sum(dim=-1) > 0).float()
        if global_batch_size is None:
            global_batch_size = seq_mask.sum().clamp(min=1)
        loss = ((seq_losses * seq_mask).sum() / global_batch_size) * dp_size

    else:
        raise ValueError(
            f"Invalid loss_agg_mode: '{loss_agg_mode}'. "
            "Expected one of: 'token-mean', 'seq-mean-token-sum', "
            "'seq-mean-token-sum-norm', 'seq-mean-token-mean'."
        )
    return loss


def kl_penalty(
    logprob: torch.Tensor,
    ref_logprob: torch.Tensor,
    method: str = "low_var_kl",
) -> torch.Tensor:
    """Per-token KL divergence estimate between current and reference policy.

    Supported methods (see http://joschu.net/blog/kl-approx.html):
    - ``"k1"`` / ``"kl"``: simple ``logprob - ref_logprob`` (biased gradient).
    - ``"abs"``: absolute value of k1.
    - ``"k2"`` / ``"mse"``: ``0.5 * (logprob - ref_logprob)^2``.
    - ``"k3"`` / ``"low_var_kl"``: variance-reduced estimator
      ``exp(ref - logprob) - (ref - logprob) - 1``.
    """
    if method in ("kl", "k1"):
        return logprob - ref_logprob
    if method == "abs":
        return (logprob - ref_logprob).abs()
    if method in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()
    if method in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-20.0, max=20.0)
        return torch.clamp(kl.exp() - kl - 1, min=-10.0, max=10.0)
    raise ValueError(
        f"Invalid kl_penalty method: '{method}'. "
        "Expected one of: 'k1'/'kl', 'abs', 'k2'/'mse', 'k3'/'low_var_kl'."
    )


def _compute_sequence_level_ratio_and_advantages(
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if log_ratio.ndim == 1:
        if cu_seqlens is None:
            raise ValueError("cu_seqlens is required for 1D tensors (packed format).")
        batch_size = cu_seqlens.shape[0] - 1
        seq_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        sequence_idx = torch.arange(batch_size, device=log_ratio.device).repeat_interleave(seq_lengths)
        masked_log_ratio = torch.where(loss_mask, log_ratio, 0.0)
        log_ratio_sum_per_seq = torch.zeros(batch_size, device=log_ratio.device, dtype=log_ratio.dtype).scatter_add_(0, sequence_idx, masked_log_ratio)
        masked_advantages = torch.where(loss_mask, advantages, 0.0)
        advantages_sum_per_seq = torch.zeros(batch_size, device=advantages.device, dtype=advantages.dtype).scatter_add_(0, sequence_idx, masked_advantages)
        valid_count_per_seq = torch.zeros(batch_size, device=loss_mask.device, dtype=torch.int32).scatter_add_(0, sequence_idx, loss_mask.int()).clamp(min=1)
        log_ratio_mean_per_seq = log_ratio_sum_per_seq / valid_count_per_seq.to(log_ratio.dtype)
        adv_mean_per_seq = advantages_sum_per_seq / valid_count_per_seq.to(advantages.dtype)
        ratio = torch.exp(log_ratio_mean_per_seq)[sequence_idx]
        ratio = torch.where(loss_mask, ratio, 0.0)
        advantages = adv_mean_per_seq[sequence_idx]
        advantages = torch.where(loss_mask, advantages, 0.0)
    else:
        seq_log_ratio_mean = torch.where(loss_mask, log_ratio, 0.0).sum(dim=1) / loss_mask.sum(dim=1).clamp(min=1)
        ratio = torch.exp(seq_log_ratio_mean.unsqueeze(1).expand_as(log_ratio))
        ratio = torch.where(loss_mask, ratio, 0.0)
        seq_lengths = loss_mask.sum(dim=-1, keepdim=True).clamp(min=1)
        advantages = (advantages.sum(dim=-1, keepdim=True) / seq_lengths).expand_as(log_ratio)
    return ratio, advantages


def ppo_actor_loss_fn(
    logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    loss_mask: torch.Tensor,
    eps_clip_higher: float | None = None,
    c_clip: float | None = None,
    behav_imp_weight_cap: float | None = None,
    importance_sampling_level: str = "token",
    cu_seqlens: torch.Tensor | None = None,
    loss_agg_mode: str = "token-mean",
    rollout_is_weights: torch.Tensor | None = None,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
) -> tuple[torch.Tensor, dict]:
    if importance_sampling_level == "sequence":
        log_ratio = logprobs - proximal_logprobs
        ratio, advantages = _compute_sequence_level_ratio_and_advantages(log_ratio, advantages, loss_mask, cu_seqlens)
    elif importance_sampling_level == "token":
        ratio = torch.where(loss_mask, torch.exp(logprobs - proximal_logprobs), 0)
    else:
        raise ValueError(f"Invalid importance_sampling_level: {importance_sampling_level}.")
    clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + (eps_clip if eps_clip_higher is None else eps_clip_higher))
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * clipped_ratio
    clip_mask = pg_loss1.detach() < pg_loss2.detach()
    pg_loss = torch.max(pg_loss1, pg_loss2)
    if c_clip is not None:
        assert c_clip > 1.0, c_clip
        pg_loss3 = torch.sign(advantages) * c_clip * advantages
        dual_clip_mask = pg_loss3.detach() < pg_loss.detach()
        pg_loss = torch.min(pg_loss, pg_loss3)
    else:
        dual_clip_mask = torch.zeros_like(clip_mask)
    behav_kl = proximal_logprobs - old_logprobs
    behav_imp_weight = behav_kl.exp()
    behav_mask = (behav_imp_weight <= behav_imp_weight_cap).logical_and(loss_mask) if behav_imp_weight_cap is not None else loss_mask
    behav_kl = torch.where(behav_mask, behav_kl, 0.0)
    behav_imp_weight = torch.where(behav_mask, behav_imp_weight, 0.0)
    pg_loss = pg_loss * behav_imp_weight
    if rollout_is_weights is not None:
        pg_loss = pg_loss * rollout_is_weights
    logging_loss = pg_loss.detach()
    pg_loss = agg_loss(
        pg_loss, loss_mask, loss_agg_mode=loss_agg_mode,
        dp_size=dp_size, batch_num_tokens=batch_num_tokens, global_batch_size=global_batch_size,
    )
    clip_mask.logical_and_(loss_mask)
    dual_clip_mask.logical_and_(loss_mask)
    stat = dict(loss=logging_loss, importance_weight=ratio.detach(), approx_kl=(logprobs - proximal_logprobs).detach(), clip_mask=clip_mask, dual_clip_mask=dual_clip_mask)
    if proximal_logprobs is not None:
        stat["behave_imp_weight"] = behav_imp_weight
        stat["behave_approx_kl"] = behav_kl
        stat["behave_mask"] = behav_mask
    return pg_loss, stat


def sapo_loss_fn(
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    tau_pos: float,
    tau_neg: float,
    loss_mask: torch.Tensor,
    importance_sampling_level: str = "token",
    cu_seqlens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    if tau_pos <= 0 or tau_neg <= 0:
        raise ValueError("SAPO temperatures must be positive.")
    loss_mask_count = loss_mask.count_nonzero() or 1
    advantages = advantages.detach()
    log_ratio = logprobs - old_logprobs
    if importance_sampling_level == "sequence":
        ratio, advantages = _compute_sequence_level_ratio_and_advantages(log_ratio, advantages, loss_mask, cu_seqlens)
    elif importance_sampling_level == "token":
        ratio = torch.exp(log_ratio)
    else:
        raise ValueError(f"Invalid importance_sampling_level: {importance_sampling_level}.")
    gate_pos = torch.sigmoid(tau_pos * (ratio - 1.0))
    gate_neg = torch.sigmoid(tau_neg * (ratio - 1.0))
    scaled_gate_pos = gate_pos * (4.0 / tau_pos)
    scaled_gate_neg = gate_neg * (4.0 / tau_neg)
    soft_gate = torch.where(advantages > 0, scaled_gate_pos, scaled_gate_neg)
    pg_loss = -soft_gate * advantages
    logging_loss = pg_loss.detach()
    pg_loss = torch.where(loss_mask, pg_loss, 0).sum() / loss_mask_count
    stat = dict(loss=logging_loss, importance_weight=ratio.detach(), approx_kl=log_ratio.detach(), clip_mask=torch.zeros_like(loss_mask, dtype=torch.bool), dual_clip_mask=torch.zeros_like(loss_mask, dtype=torch.bool), sapo_soft_gate=soft_gate.detach(), sapo_scaled_gate_pos=scaled_gate_pos.detach(), sapo_scaled_gate_neg=scaled_gate_neg.detach())
    return pg_loss, stat
