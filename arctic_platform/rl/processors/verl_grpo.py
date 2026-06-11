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

from typing import Any
from typing import Optional

import torch

from .pipeline import register_loss_fn


def mask_and_pad_for_log_probs(tensor: torch.Tensor, batch: dict) -> torch.Tensor:
    """Place per-token logprobs at their response positions in the packed [total_nnz]
    buffer, applying the legacy predict-next shift (padded[i] = values[i-1]) inside
    each response window and zero everywhere else.

    Fully vectorized (no Python per-sequence loop): a window mask is built via a
    prefix-sum over +1/-1 boundary deltas, and the shift is a single roll.
    """
    values = tensor.values() if tensor.is_nested else tensor
    response_lens = batch["response_lens"]
    sequence_offsets = batch["sequence_offsets"]

    N = values.shape[0]
    device = values.device

    # Response window for each sequence is [seq_offset - resp_len, seq_offset).
    ends = torch.as_tensor(sequence_offsets, device=device, dtype=torch.long).reshape(-1)
    starts = ends - torch.as_tensor(response_lens, device=device, dtype=torch.long).reshape(-1)

    # Boolean mask of response positions without a Python loop: +1 at each window
    # start, -1 at each window end, prefix-sum, then > 0. (windows are disjoint.)
    delta = torch.zeros(N + 1, dtype=torch.int32, device=device)
    delta.index_add_(0, starts, torch.ones_like(starts, dtype=torch.int32))
    delta.index_add_(0, ends, torch.full_like(ends, -1, dtype=torch.int32))
    response_mask = delta[:N].cumsum(0) > 0

    # padded[i] = values[i-1] within response windows (predict-next convention),
    # 0 outside. Position 0 is never a response position (prompt precedes response).
    shifted = torch.empty_like(values)
    shifted[1:] = values[:-1]
    shifted[0] = 0

    return torch.where(response_mask, shifted, torch.zeros_like(values))


def shift_log_probs_left(tensor: torch.Tensor, batch: dict) -> torch.Tensor:
    # zorro produces log_probs that are already response-aligned
    # (output index k == log P(response_token[k])).
    # Previously this function did torch.roll(values, shifts=+1, dims=-1)
    # (with zeros at the start of every sequence) to mimic the legacy
    # "predict-next" convention so it would line up with old_log_probs
    # being produced with the matching off-by-one by the verl-side
    # no_padding_2_padding. That made `ratio = exp(new - old)` cancel
    # cleanly (both wrong identically) on the first PPO iteration, but
    # the autograd graph still flowed the policy gradient for
    # advantage A[k] into log P(response_token[k-1]) -- a systematic
    # 1-token misalignment of the policy update. The fix is to keep
    # log_probs response-aligned end-to-end (verl side + server side)
    # so the gradient lands on the token the advantage was actually
    # computed for.
    values = tensor.values() if tensor.is_nested else tensor
    return values


def masked_sum(values: torch.Tensor, mask: torch.Tensor, axis: int | tuple[int, ...] | None = None) -> torch.Tensor:
    """Compute sum of tensor values where mask is True.

    NaN values outside the mask are replaced with zeros to prevent
    contaminating the sum.

    Args:
        values: Input tensor containing values to sum.
        mask: Boolean or numeric mask tensor (same shape as values).
            Non-zero values indicate elements to include.
        axis: Dimension(s) along which to sum. None sums all elements.

    Returns:
        torch.Tensor: Sum of masked values, reduced along specified axis.
    """
    # If NaNs exist out of mask, replace NaNs in values with a value that
    # won't affect the sum (e.g., 0 for masked regions). The torch.where already
    # zeros out-of-mask positions, so a subsequent `* mask` would be redundant.
    valid_values = torch.where(mask.bool(), values, 0.0)
    return valid_values.sum(axis=axis)


def masked_mean(values, mask, axis=None):
    """
    Compute the mean of `values` over elements selected by `mask`.

    Args:
        values (Tensor): Input tensor.
        mask (Tensor): Boolean or numeric mask of the same shape as `values`.
        axis (int or tuple of int, optional): Dimension(s) along which to compute the mean.
            Defaults to None (over all elements).

    Returns:
        Tensor: Masked mean, with shape equal to `values` reduced over `axis`.
    """
    s = masked_sum(values, mask, axis)
    return s / (mask.sum(axis=axis) + 1e-8)


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
    """
    Aggregate the loss across global batch to ensure the loss is invariant to fsdp/megatron parallelism.

    NOTE: The returned loss has different behaviors for different backend:
    - FSDP: the loss is directly used for backward.
    - Megatron: the loss should be scaled by `num_microbatches` and `cp_size` for pp schedule.

    Args:
        loss_mat: micro batch loss matrix, (bs, response_length)
        loss_mask: micro batch loss mask, (bs, response_length)
        loss_agg_mode: method to aggregate the loss matrix into a scalar
        dp_size: data parallel size
        batch_num_tokens: number of valid tokens in global batch
        global_batch_size: global batch size
        loss_scale_factor: scale factor for "seq-mean-token-sum-norm" mode. If None, uses loss_mask.shape[-1].
            Set this to a constant value to ensure consistent normalization throughout training.

    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            if dp_size > 1:
                raise ValueError("(global) batch_num_tokens is required when dp_size > 1")
            batch_num_tokens = loss_mask.sum()
        loss = masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode in ["seq-mean-token-sum", "seq-mean-token-sum-norm"]:
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss = masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
        if loss_agg_mode == "seq-mean-token-sum-norm":
            if loss_scale_factor is None:
                horizon = loss_mask.shape[-1]
                loss_scale_factor = horizon
            loss /= loss_scale_factor
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask = torch.sum(loss_mask, dim=-1)  # per-sequence token count
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean
        seq_mask = (seq_mask > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            if dp_size > 1:
                raise ValueError("global_batch_size is required when dp_size > 1")
            global_batch_size = seq_mask.sum()
        loss = masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss_vanilla(
    old_log_probs: torch.Tensor,
    log_probs: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    global_batch_info: dict[str, Any] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_probs (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_probs (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    # TODO: add config support
    # assert config is not None
    # assert not isinstance(config, AlgoConfig)
    # clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    # clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    # clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    # clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
    #     "clip_ratio_c", 3.0
    # )
    clip_ratio = 0.2
    clip_ratio_low = 0.2
    clip_ratio_high = 0.2
    clip_ratio_c = 3.0

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_probs - old_log_probs
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    # print(f"{old_log_probs.shape=} {log_probs.shape=} {negative_approx_kl.shape=} {response_mask.shape=}")
    # print(f"{old_log_probs.sum()=} {log_probs.sum()=} {negative_approx_kl.sum()=}")
    # print(f"{old_log_probs=}")
    # print(f"{log_probs=}")
    # print(f"{negative_approx_kl=}")
    # print(f"{response_mask=}")

    ppo_kl = masked_mean(-negative_approx_kl, response_mask)
    # print(f"{ppo_kl=}")

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **global_batch_info)

    # Emit raw masked sums + token counts so that the worker / server
    # epilogue can combine them into a global token-mean per metric per
    # mini-batch (see ``utils.batch.combine_metric_microbatches`` and
    # ``utils.batch.combine_metric_shards``). The ``pg_loss`` tensor
    # returned above is unchanged so the backward pass keeps its
    # token-mean × dp_size scaling.
    response_mask_f = response_mask.float()
    pg_clipfrac_mask = torch.gt(pg_losses2, pg_losses1).float() * response_mask_f
    pg_clipfrac_lower_mask = torch.gt(clip_pg_losses1, pg_losses3).float() * (advantages < 0).float() * response_mask_f
    # Stack all masked sums into a single tensor so the GPU->CPU transfer is one
    # sync (instead of 5 separate .item() stalls) on the critical path before backward.
    stacked = torch.stack(
        [
            response_mask_f.sum(),  # mask_token_count
            pg_clipfrac_mask.sum(),  # actor/pg_clipfrac.sum
            ((-negative_approx_kl) * response_mask_f).sum(),  # actor/ppo_kl.sum
            pg_clipfrac_lower_mask.sum(),  # actor/pg_clipfrac_lower.sum
            (pg_losses * response_mask_f).sum(),  # actor/pg_loss.sum
        ]
    ).detach()
    (
        mask_token_count,
        pg_clipfrac_sum,
        ppo_kl_sum,
        pg_clipfrac_lower_sum,
        pg_loss_sum,
    ) = stacked.cpu().tolist()
    pg_metrics = {
        "actor/pg_clipfrac.sum": pg_clipfrac_sum,
        "actor/pg_clipfrac.tokens": mask_token_count,
        "actor/ppo_kl.sum": ppo_kl_sum,
        "actor/ppo_kl.tokens": mask_token_count,
        "actor/pg_clipfrac_lower.sum": pg_clipfrac_lower_sum,
        "actor/pg_clipfrac_lower.tokens": mask_token_count,
        "actor/pg_loss.sum": pg_loss_sum,
        "actor/pg_loss.tokens": mask_token_count,
    }
    return pg_loss, pg_metrics


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expected value of KL, but the expected gradient of k1 and k3
    estimator is not the expected gradient of KL. On the other hand k2 estimator gives right gradient estimator,
    so we use a straight through trick here if the kl_penalty method ends with '+', e.g., k3+.
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


class VerlPolicyConfig:
    def __init__(self, actor_config_dict: dict, policy_loss_config_dict: dict):
        self.loss_agg_mode = actor_config_dict.get("loss_agg_mode", "token-mean")
        self.kl_loss_coef = actor_config_dict.get("kl_loss_coef", 0.001)
        self.kl_loss_type = actor_config_dict.get("kl_loss_type", "low_var_kl")
        self.clip_ratio = actor_config_dict.get("clip_ratio", 0.2)
        self.clip_ratio_low = actor_config_dict.get("clip_ratio_low", 0.2)
        self.clip_ratio_high = actor_config_dict.get("clip_ratio_high", 0.2)
        self.clip_ratio_c = actor_config_dict.get("clip_ratio_c", 3.0)
        self.entropy_coeff = actor_config_dict.get("entropy_coeff", 0.0)
        self.use_kl_loss = actor_config_dict.get("use_kl_loss", False)
        self.loss_mode = policy_loss_config_dict.get("loss_mode", "vanilla")


@register_loss_fn("verl_grpo")
def verl_grpo_loss(model_outputs: dict, batch: dict, meta: dict, config: dict, device: str):
    actor_config = meta.get("actor_config", {})
    policy_loss_config = meta.get("policy_loss_config", {})
    verl_policy_config = VerlPolicyConfig(actor_config, policy_loss_config)

    # print(f"_verl_grpo_loss: {verl_policy_config=}")

    log_probs = model_outputs["logprobs"].squeeze()
    entropy = model_outputs.get("entropy", None)
    if entropy is not None:
        entropy = entropy.squeeze()
        # print(f"_verl_grpo_loss: {entropy.shape=}")
    # print(f"_verl_grpo_loss: {log_probs.shape=}")

    global_batch_info = {k: meta[k] for k in ["dp_size", "batch_num_tokens", "global_batch_size"]}
    global_batch_info["loss_scale_factor"] = None

    metrics = {}

    response_mask = batch["response_mask"].to(bool)
    # compute policy loss
    old_log_probs = batch["old_log_probs"]
    advantages = batch["advantages"]
    rollout_is_weights = meta.get("rollout_is_weights", None)

    loss_agg_mode = verl_policy_config.loss_agg_mode

    if meta.get("zorro_train_enable", False):
        log_probs = shift_log_probs_left(log_probs.squeeze(0), batch)
    else:
        log_probs = mask_and_pad_for_log_probs(log_probs.squeeze(0), batch)

    pg_loss, pg_metrics = compute_policy_loss_vanilla(
        old_log_probs=old_log_probs,
        log_probs=log_probs,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        global_batch_info=global_batch_info,
        rollout_is_weights=rollout_is_weights,
    )

    # ``pg_metrics`` already contains paired ``actor/pg_loss.sum`` /
    # ``actor/pg_loss.tokens`` (plus paired entries for ppo_kl, pg_clipfrac,
    # pg_clipfrac_lower) so that downstream aggregators can collapse the
    # per-(rank × microbatch) shape into a single global token-mean per
    # metric per mini-batch.
    metrics.update(pg_metrics)
    policy_loss = pg_loss
    # Reuse the token count already synced inside compute_policy_loss_vanilla
    # instead of recomputing it with another GPU->CPU .item() stall.
    mask_token_count = metrics["actor/pg_loss.tokens"]
    response_mask_f = response_mask.float()

    # The ``loss`` we expose for logging mirrors the structure of the
    # ``policy_loss`` tensor used for backward, but emitted as a paired
    # ``.sum`` / ``.tokens`` so the server-side combiner can compute a
    # global token-mean. We start from ``pg_loss``'s contribution and add
    # the entropy / KL contributions if they are active.
    loss_sum = metrics["actor/pg_loss.sum"]

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **global_batch_info
        )
        entropy_coeff = verl_policy_config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        entropy_loss_sum = (entropy * response_mask_f).sum().detach().item()
        metrics["actor/entropy_loss.sum"] = entropy_loss_sum
        metrics["actor/entropy_loss.tokens"] = mask_token_count
        loss_sum -= entropy_coeff * entropy_loss_sum

    # add kl loss
    if verl_policy_config.use_kl_loss:
        ref_log_prob = batch["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_probs, ref_logprob=ref_log_prob, kl_penalty=verl_policy_config.kl_loss_type)
        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **global_batch_info)

        policy_loss += kl_loss * verl_policy_config.kl_loss_coef
        kl_loss_sum = (kld * response_mask_f).sum().detach().item()
        metrics["kl_loss.sum"] = kl_loss_sum
        metrics["kl_loss.tokens"] = mask_token_count
        # ``kl_coef`` is a hyperparameter constant; carry through as a plain
        # scalar (the aggregator passes through non-paired numeric values
        # via a simple cross-shard mean — fine for replicated constants).
        metrics["kl_coef"] = verl_policy_config.kl_loss_coef
        loss_sum += verl_policy_config.kl_loss_coef * kl_loss_sum

    metrics["loss.sum"] = loss_sum
    metrics["loss.tokens"] = mask_token_count
    # print(f"_verl_grpo_loss: {policy_loss=}")

    return policy_loss, metrics
