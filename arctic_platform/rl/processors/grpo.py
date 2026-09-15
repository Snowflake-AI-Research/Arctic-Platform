"""GRPO loss function and proximal logp utilities."""

from __future__ import annotations

import math
from collections.abc import Sequence
from enum import Enum
from itertools import pairwise
from typing import Tuple

import torch

from .functional import (
    EchoBatchDenominator,
    _resolve_dp_size,
    agg_loss,
    canonicalize_loss_mask,
    cispo_actor_loss_fn,
    dp_loss_multiplier,
    echo_env_prediction_loss_fn,
    kl_penalty,
    ppo_actor_loss_fn,
    sapo_loss_fn,
)
from arctic_platform.common.registry import register_loss_fn

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
    alpha = torch.where((version_diff > 0) & generated_tokens_mask, version_gap / version_diff, torch.zeros_like(v_behave))
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
            raise ValueError(f"prox_logp is None with prox_logp_method='{prox_logp_method}' but versions not available.")
    prox_logp = prox_logp_gt
    if prox_logp_method == PROX_LOGP_METHOD_LOGLINEAR:
        if prox_logp_is_none and versions is not None and current_version is not None:
            approximations = compute_prox_logp_approximations(old_logp=old_logp, logprobs=logprobs, versions=versions, current_version=current_version, method=PROX_APPROX_METHOD_LOGLINEAR)
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
    dp_size: int | None = None,
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
) -> torch.Tensor:
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
            raise ValueError(
                "grpo labels must match loss_mask when labels are present"
            )
        if ((labels == -100) & loss_mask).any().item():
            raise ValueError(
                "grpo loss_mask must be zero where labels use ignore_index=-100"
            )
    # ECHO disjointness is validated against the client's policy mask, not the
    # (possibly M2PO-shrunk) mask used for the policy loss below.
    policy_loss_mask = loss_mask
    prox_logp_gt = input_data.get("prox_logp")
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
    if not loss_mask.any():
        zero_loss = torch.nan_to_num(logprobs).sum() * 0.0
        metrics = {
            "approx_kl": 0.0,
            "importance_weight": 0.0,
            "clip_ratio": 0.0,
            "entropy": 0.0,
        }
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

    if use_sapo_loss and use_cispo_loss:
        raise ValueError("use_sapo_loss and use_cispo_loss are mutually exclusive.")
    if use_cispo_loss and c_clip is not None:
        raise ValueError("c_clip is not supported with use_cispo_loss=True.")

    if use_sapo_loss:
        if use_decoupled_loss:
            raise ValueError("SAPO is not compatible with use_decoupled_loss=True.")
        loss, stat = sapo_loss_fn(logprobs=logprobs, old_logprobs=old_logp, advantages=advantages, tau_pos=sapo_tau_pos, tau_neg=sapo_tau_neg, loss_mask=loss_mask, importance_sampling_level=importance_sampling_level, cu_seqlens=input_data.get("cu_seqlens"), loss_agg_mode=loss_agg_mode, dp_size=dp_size, batch_num_tokens=batch_num_tokens, global_batch_size=global_batch_size, prompt_group_ids=prompt_group_ids, prompt_token_counts=prompt_token_counts, sequence_loss_weights=sequence_loss_weights)
    elif use_cispo_loss:
        loss, stat = cispo_actor_loss_fn(logprobs=logprobs, proximal_logprobs=prox_logp, old_logprobs=old_logp, advantages=advantages, eps_clip=eps_clip, eps_clip_higher=eps_clip_higher, is_weight_clip_max=is_weight_clip_max, loss_mask=loss_mask, behav_imp_weight_cap=behav_imp_weight_cap, importance_sampling_level=importance_sampling_level, cu_seqlens=input_data.get("cu_seqlens"), loss_agg_mode=loss_agg_mode, rollout_is_weights=rollout_is_weights, dp_size=dp_size, batch_num_tokens=batch_num_tokens, global_batch_size=global_batch_size, prompt_group_ids=prompt_group_ids, prompt_token_counts=prompt_token_counts, sequence_loss_weights=sequence_loss_weights)
    else:
        loss, stat = ppo_actor_loss_fn(logprobs=logprobs, old_logprobs=old_logp, advantages=advantages, eps_clip=eps_clip, eps_clip_higher=eps_clip_higher, loss_mask=loss_mask, c_clip=c_clip, proximal_logprobs=prox_logp, behav_imp_weight_cap=behav_imp_weight_cap, importance_sampling_level=importance_sampling_level, cu_seqlens=input_data.get("cu_seqlens"), loss_agg_mode=loss_agg_mode, rollout_is_weights=rollout_is_weights, dp_size=dp_size, batch_num_tokens=batch_num_tokens, global_batch_size=global_batch_size, prompt_group_ids=prompt_group_ids, prompt_token_counts=prompt_token_counts, sequence_loss_weights=sequence_loss_weights)

    # Optional entropy bonus: subtract entropy_coeff * mean_entropy from loss
    if entropy_coeff != 0.0:
        entropy_loss = agg_loss(
            -entropy.float(), loss_mask, loss_agg_mode=loss_agg_mode,
            dp_size=dp_size, batch_num_tokens=batch_num_tokens, global_batch_size=global_batch_size,
            prompt_group_ids=prompt_group_ids, prompt_token_counts=prompt_token_counts,
            sequence_loss_weights=sequence_loss_weights, cu_seqlens=input_data.get("cu_seqlens"),
        )
        loss = loss + entropy_coeff * entropy_loss

    # Optional KL penalty against a reference policy (e.g. SFT model)
    if use_kl_loss:
        ref_logprobs = input_data.get("ref_log_probs")
        if ref_logprobs is None:
            raise ValueError("use_kl_loss=True but 'ref_log_probs' not found in context.")
        kl = kl_penalty(logprob=logprobs, ref_logprob=ref_logprobs.to(logprobs.device), method=kl_loss_type)
        kl_loss = agg_loss(
            kl, loss_mask, loss_agg_mode=loss_agg_mode,
            dp_size=dp_size, batch_num_tokens=batch_num_tokens, global_batch_size=global_batch_size,
            prompt_group_ids=prompt_group_ids, prompt_token_counts=prompt_token_counts,
            sequence_loss_weights=sequence_loss_weights, cu_seqlens=input_data.get("cu_seqlens"),
        )
        loss = loss + kl_loss_coef * kl_loss

    metrics = {
        "approx_kl":         _masked_mean_float(stat["approx_kl"].detach(), loss_mask),
        "importance_weight": _masked_mean_float(stat["importance_weight"].detach(), loss_mask),
        "clip_ratio":        _masked_mean_float(stat["clip_mask"].float(), loss_mask),
        "entropy":           _masked_mean_float(entropy.float(), loss_mask),
        "loss":              float(loss.detach().cpu().item()),
    }

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
            raise ValueError(
                f"aux_ce_weight must be a finite non-negative number, got {aux_ce_weight!r}"
            )
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
        # The aux term inherits the policy term's distributed-reduction
        # convention so the echo-to-policy ratio equals the configured
        # aux_ce_weight (paper λ) at every DP width — passing the raw
        # dp_size would multiply the effective λ by DP width under
        # prompt-mean + sequence_loss_weights.
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
        )
        aux_loss = aux_ce_weight * env_loss
        metrics.update({
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
        })
        if aux_ce_weight > 0.0:
            loss = loss + aux_loss

    return loss, metrics


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
    batch_num_tokens = config.get("batch_num_tokens")
    dp_size = _resolve_dp_size(config.get("dp_size"), batch_num_tokens)

    logprobs = model_outputs.get("logprobs")
    if logprobs is None:
        logits = model_outputs["logits"]
        input_ids = context["input_ids"].to(logits.device)
        if input_ids.ndim < logits.ndim:
            input_ids = input_ids.view(logits.shape[:-1])
        labels = torch.roll(input_ids, shifts=-1, dims=-1)
        logprobs = torch.log_softmax(logits.float(), dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)

    # Degenerate-shard logprob sanitization (see _internal_grpo_loss_fn). Under
    # Ulysses SP a short/padded batch can hand a rank a tail shard whose "valid"
    # positions have no attendable keys, so attention emits non-finite logits ->
    # non-finite logprobs. A single NaN poisons both the loss and the entropy
    # metric ("result.metrics.entropy is non-finite"). Zero them here, before
    # entropy is derived, so downstream loss + metrics stay finite; nan_to_num
    # yields zero gradient at those positions. No-op at real training seqlens.
    if not torch.isfinite(logprobs).all():
        logprobs = torch.nan_to_num(logprobs, nan=0.0, posinf=0.0, neginf=0.0)
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
        "sft_mask": context.get("sft_mask"),
        "echo_observation_mask": context.get("echo_observation_mask"),
        "labels": context.get("labels"),
    }
    if input_data["sft_mask"] is not None:
        input_data["sft_mask"] = input_data["sft_mask"].to(logprobs.device)
    if input_data["echo_observation_mask"] is not None:
        input_data["echo_observation_mask"] = input_data["echo_observation_mask"].to(logprobs.device)
    if input_data["prox_logp"] is not None:
        input_data["prox_logp"] = input_data["prox_logp"].to(logprobs.device)
    if input_data["versions"] is not None:
        input_data["versions"] = input_data["versions"].to(logprobs.device)
    if input_data["ref_log_probs"] is not None:
        input_data["ref_log_probs"] = input_data["ref_log_probs"].to(logprobs.device)

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
            "sft_mask",
            "echo_observation_mask",
            "labels",
        ):
            input_data[key] = _packed_singleton_to_1d(input_data[key])
        rollout_is_weights = _packed_singleton_to_1d(rollout_is_weights)

    prompt_group_ids = context.get("prompt_group_ids")
    if prompt_group_ids is not None:
        prompt_group_ids = prompt_group_ids.to(logprobs.device)

    prompt_token_counts = context.get("prompt_token_counts")
    if prompt_token_counts is not None:
        prompt_token_counts = prompt_token_counts.to(logprobs.device)

    sequence_loss_weights = context.get("sequence_loss_weights")
    if sequence_loss_weights is not None:
        sequence_loss_weights = sequence_loss_weights.to(logprobs.device)

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
        use_cispo_loss=config.get("use_cispo_loss", False),
        is_weight_clip_max=config.get("is_weight_clip_max"),
        loss_agg_mode=config.get("loss_agg_mode", "token-mean"),
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=config.get("global_batch_size"),
        rollout_is_weights=rollout_is_weights,
        entropy_coeff=config.get("entropy_coeff", 0.0),
        use_kl_loss=config.get("use_kl_loss", False),
        kl_loss_coef=config.get("kl_loss_coef", 0.001),
        kl_loss_type=config.get("kl_loss_type", "low_var_kl"),
        prompt_group_ids=prompt_group_ids,
        prompt_token_counts=prompt_token_counts,
        sequence_loss_weights=sequence_loss_weights,
        aux_ce_weight=config.get("aux_ce_weight"),
        echo_global_num_sequences=config.get("echo_global_num_sequences"),
        echo_batch_denominator=config.get(
            "echo_batch_denominator", EchoBatchDenominator.ALL_SEQUENCES.value
        ),
    )
    return loss, metrics


_GRPO_CONFIG_KEYS = frozenset({
    "eps_clip",
    "eps_clip_higher",
    "c_clip",
    "behav_imp_weight_cap",
    "m2_threshold",
    "importance_sampling_level",
    "current_version",
    "prox_logp_method",
    "use_sapo_loss",
    "sapo_tau_pos",
    "sapo_tau_neg",
    "use_decoupled_loss",
    "use_cispo_loss",
    "is_weight_clip_max",
    "loss_agg_mode",
    "dp_size",
    "batch_num_tokens",
    "global_batch_size",
    "entropy_coeff",
    "use_kl_loss",
    "kl_loss_coef",
    "kl_loss_type",
})
_ECHO_REQUIRED_CONFIG_KEYS = frozenset({"aux_ce_weight", "echo_global_num_sequences"})
_ECHO_CONFIG_KEYS = _ECHO_REQUIRED_CONFIG_KEYS | {"echo_batch_denominator"}


def _grpo_preflight_mask(microbatch: dict) -> torch.Tensor:
    reference = microbatch.get("input_ids")
    if not torch.is_tensor(reference):
        raise ValueError(
            "grpo packed microbatches require tensor input_ids for preflight validation"
        )
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
            raise ValueError(
                "grpo loss_mask must be zero where labels use ignore_index=-100"
            )
    return mask


def _active_sequence_count(microbatch: dict, loss_mask: torch.Tensor) -> float:
    cu_seqlens = microbatch.get("cu_seqlens")
    if torch.is_tensor(cu_seqlens):
        flat_mask = loss_mask.reshape(-1)
        boundaries = cu_seqlens.detach().cpu().tolist()
        return float(
            sum(
                bool(flat_mask[start:end].any().item())
                for start, end in pairwise(boundaries)
            )
        )
    if loss_mask.ndim >= 2:
        return float(loss_mask.reshape(loss_mask.shape[0], -1).any(dim=1).sum().item())
    return float(bool(loss_mask.any().item()))


def _grpo_packed_loss_reduction(
    microbatches: Sequence[dict],
    config: dict,
    loss_fn_name: str,
) -> PackedLossReduction:
    masks = [_grpo_preflight_mask(microbatch) for microbatch in microbatches]
    mode = config.get("loss_agg_mode", "token-mean")

    if mode == "token-mean":
        weights = [float(mask.sum().item()) for mask in masks]
        reduction = (
            additive_packed_loss_reduction(weights)
            if config.get("batch_num_tokens") is not None
            else local_mean_packed_loss_reduction(weights)
        )
    elif mode in ("seq-mean-token-sum", "seq-mean-token-mean"):
        weights = [
            _active_sequence_count(microbatch, mask)
            for microbatch, mask in zip(microbatches, masks)
        ]
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
            torch.is_tensor(microbatch.get("sequence_loss_weights"))
            for microbatch in microbatches
        )
        if has_sequence_weights:
            weights = [
                float(microbatch["sequence_loss_weights"].abs().sum().item())
                for microbatch in microbatches
            ]
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

    if (
        loss_fn_name.endswith("grpo_echo_v1")
        and len(microbatches) > 1
        and not reduction.loss_is_additive
    ):
        raise ValueError(
            f"loss_fn {loss_fn_name!r} requires a globally normalized additive "
            "policy objective when split into multiple packed microbatches"
        )
    return reduction


def _merge_distributed_config(config: dict, batch: dict, meta: dict) -> dict:
    """AP puts dp_size / batch_num_tokens / global_batch_size on meta; Cortex reads config."""
    cfg = dict(config)
    for key in ("dp_size", "batch_num_tokens", "global_batch_size"):
        if cfg.get(key) is None:
            if meta.get(key) is not None:
                cfg[key] = meta[key]
            elif batch.get(key) is not None:
                cfg[key] = batch[key]
    return cfg


def _grpo_context(batch: dict, meta: dict) -> dict:
    """Merge bags, but take ``rollout_is_weights`` from ``batch`` only.

    That tensor is batch-dim; ``meta`` is DP-replicated, so a copy there is
    the wrong length after ``_split_batch``.
    """
    context = {**batch, **meta}
    context["rollout_is_weights"] = batch.get("rollout_is_weights")
    return context


@register_loss_fn("ap_grpo", packed_loss_reduction=_grpo_packed_loss_reduction)
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
    echo_keys = _ECHO_CONFIG_KEYS & set(config)
    if echo_keys:
        raise ValueError(
            f"loss_fn 'ap_grpo' does not accept ECHO config keys {sorted(echo_keys)} — request "
            "loss_fn 'ap_grpo_echo_v1', whose strict schema fails loudly on typos and on servers "
            "without ECHO support."
        )
    return _grpo_loss(model_outputs, _grpo_context(batch, meta), config, device)


@register_loss_fn(
    "ap_grpo_echo_v1",
    packed_loss_reduction=_grpo_packed_loss_reduction,
)
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
    unknown_keys = set(config) - _GRPO_CONFIG_KEYS - _ECHO_CONFIG_KEYS
    if unknown_keys:
        raise ValueError(
            f"Unknown config keys for loss_fn 'ap_grpo_echo_v1': {sorted(unknown_keys)} — this "
            "contract fails loudly on unrecognized keys so a typo cannot silently disable an "
            "objective."
        )
    # get() folds present-but-None into "missing": None would pass a
    # key-presence check and silently train baseline GRPO downstream.
    missing_keys = {key for key in _ECHO_REQUIRED_CONFIG_KEYS if config.get(key) is None}
    if missing_keys:
        raise ValueError(
            f"loss_fn 'ap_grpo_echo_v1' requires non-None config keys {sorted(missing_keys)} — "
            "use plain 'ap_grpo' for runs without the ECHO objective."
        )
    return _grpo_loss(model_outputs, _grpo_context(batch, meta), config, device)
