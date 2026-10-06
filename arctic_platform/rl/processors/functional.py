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

"""Functional math utilities for RL loss computation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from numbers import Real
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_autograd
from deepspeed.utils import groups

_GLOBAL_LOSS_SCALE_KEYS = ("dp_size", "batch_num_tokens", "global_batch_size")


def _get_sequence_parallel_group():
    if dist.is_initialized() and groups._get_sequence_parallel_world_size() > 1:
        return groups._get_sequence_parallel_group()
    return None


def _sequence_parallel_sum(*totals: torch.Tensor, group) -> tuple[torch.Tensor, ...]:
    """Add up per-sequence totals over the group's token windows, in one collective.

    The all-reduce is the autograd-aware one: every rank's loss reads the reduced total, so a
    token's gradient owes a term to each rank that read it, and only a reduction in backward can
    collect those terms. Reducing in forward alone leaves each window with its own share of that
    gradient and drops the rest.
    """
    stacked = torch.stack([total.to(totals[0].dtype) for total in totals])
    stacked = dist_autograd.all_reduce(stacked, group=group)
    return tuple(stacked.unbind())


def _masked_values(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask.bool(), values, torch.zeros_like(values))


def _safe_masked_operand(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return torch.where(mask.bool(), values, torch.zeros_like(values))


def canonicalize_loss_mask(
    loss_mask: torch.Tensor,
    reference: torch.Tensor,
    *,
    objective: str,
    binary: bool,
) -> torch.Tensor:
    """Validate a token mask and move it to the objective tensor's device."""
    if not torch.is_tensor(loss_mask):
        raise ValueError(f"{objective} loss_mask must be a tensor, got {type(loss_mask).__name__}")
    if tuple(loss_mask.shape) != tuple(reference.shape):
        raise ValueError(
            f"{objective} loss_mask must match the objective shape, got "
            f"loss_mask={tuple(loss_mask.shape)} objective={tuple(reference.shape)}"
        )
    if torch.is_complex(loss_mask):
        raise ValueError(f"{objective} loss_mask must contain real numeric values")
    if not torch.isfinite(loss_mask).all().item():
        raise ValueError(f"{objective} loss_mask values must be finite")
    if (loss_mask < 0).any().item():
        raise ValueError(f"{objective} loss_mask values must be non-negative")
    if binary:
        is_binary = (loss_mask == 0) | (loss_mask == 1)
        if not is_binary.all().item():
            raise ValueError(
                f"{objective} loss_mask must be binary (0 or 1); use "
                "loss_fn='causal_cross_entropy' for fractional token weights"
            )
        return loss_mask.to(device=reference.device, dtype=torch.bool)
    canonical = loss_mask.to(device=reference.device, dtype=torch.float32)
    if not torch.isfinite(canonical).all().item():
        raise ValueError(f"{objective} loss_mask values must be representable as finite float32 values")
    return canonical


def _packed_per_sequence_sums(
    cu_seqlens: torch.Tensor, *values: torch.Tensor
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Segment-sum packed per-token tensors into per-sequence totals.

    Every packed ([1, T] / [T] + ``cu_seqlens``) loss path needs the same
    reduction: map each token to its sequence row and sum. Centralizing it
    also centralizes the ``cu_seqlens`` validation (1-D integer boundaries,
    starting at 0, non-decreasing, covering the packed token dimension
    exactly) — a malformed ``cu_seqlens`` would otherwise silently
    misattribute tokens across sequence boundaries and corrupt every
    per-sequence quantity downstream.

    Returns the ``[T]`` token→sequence index and one ``[B]`` sum per input
    tensor (``B = len(cu_seqlens) - 1``); each sum keeps its input's dtype.
    """
    if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2 or cu_seqlens.dtype not in (torch.int32, torch.int64):
        raise ValueError(
            "cu_seqlens must be a 1-D integer tensor with at least two boundary entries, "
            f"got shape {tuple(cu_seqlens.shape)} dtype {cu_seqlens.dtype}."
        )
    device = values[0].device
    cu_seqlens = cu_seqlens.to(device)
    if int(cu_seqlens[0].item()) != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {int(cu_seqlens[0].item())}.")
    if ((cu_seqlens[1:] - cu_seqlens[:-1]) < 0).any():
        raise ValueError("cu_seqlens must be non-decreasing.")
    total_tokens = int(cu_seqlens[-1].item())
    for value in values:
        if value.numel() != total_tokens:
            raise ValueError(
                f"packed tensors must cover cu_seqlens exactly: got {value.numel()} "
                f"tokens for cu_seqlens[-1]={total_tokens}"
            )
    num_sequences = cu_seqlens.shape[0] - 1
    sequence_idx = torch.repeat_interleave(
        torch.arange(num_sequences, device=device),
        (cu_seqlens[1:] - cu_seqlens[:-1]).long(),
    )
    sums = [value.new_zeros(num_sequences).scatter_add_(0, sequence_idx, value.reshape(-1)) for value in values]
    return sequence_idx, sums


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
        factor = torch.tensor(np.prod([x.shape[d] for d in dim]), dtype=dtype, device=x.device)
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


def _resolve_dp_size(dp_size: Optional[int], batch_num_tokens: Optional[float]) -> int:
    """Map omitted ``dp_size`` to 1 (single rank). A global token denom still needs an explicit factor."""
    if batch_num_tokens is not None and dp_size is None:
        raise ValueError(
            "batch_num_tokens requires an explicit dp_size: it is a step-global token "
            "denominator, and defaulting dp_size to 1 attenuates gradients by the "
            "data-parallel factor."
        )
    if dp_size is None:
        return 1
    if isinstance(dp_size, bool) or not isinstance(dp_size, Integral) or int(dp_size) < 1:
        raise ValueError(f"dp_size must be an integer >= 1, got {dp_size!r}")
    return int(dp_size)


def _scale_value_present(bag: dict | None, key: str) -> bool:
    return bag is not None and key in bag and bag[key] is not None


def _scale_values_equal(key: str, left, right) -> bool:
    if key == "batch_num_tokens":
        try:
            return math.isclose(float(left), float(right), rel_tol=1e-6, abs_tol=1e-9)
        except (TypeError, ValueError):
            return left == right
    return left == right


def resolve_global_loss_scale(
    context: Optional[dict],
    config: Optional[dict],
) -> dict:
    """Cortex trio: context wins. Both present and unequal raises.

    Missing keys are omitted so callers can tell a global denominator from a
    local fallback. AP's ``_merge_distributed_config`` is the opposite (config wins).
    """
    out: dict = {}
    for key in _GLOBAL_LOSS_SCALE_KEYS:
        have_ctx = _scale_value_present(context, key)
        have_cfg = _scale_value_present(config, key)
        if have_ctx and have_cfg and not _scale_values_equal(key, context[key], config[key]):
            raise ValueError(f"conflicting {key}: context={context[key]!r} config={config[key]!r}")
        if have_ctx:
            out[key] = context[key]
        elif have_cfg:
            out[key] = config[key]
    return out


def _explicit_zero_step_count(name: str, loss_mask: torch.Tensor) -> int:
    has_policy_tokens = loss_mask.any().to(dtype=torch.int8)
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(has_policy_tokens, op=dist.ReduceOp.MAX)
    if has_policy_tokens.item():
        raise ValueError(
            f"{name}=0 declares a step with no policy tokens, but this call has policy tokens on at least one rank."
        )
    return 1


def _validate_loss_denominators(
    batch_num_tokens: int | float | None,
    global_batch_size: int | None,
) -> None:
    if batch_num_tokens is not None and (
        isinstance(batch_num_tokens, bool)
        or not isinstance(batch_num_tokens, Real)
        or not math.isfinite(float(batch_num_tokens))
        or batch_num_tokens < 0
    ):
        raise ValueError(f"batch_num_tokens must be a finite non-negative number, got {batch_num_tokens!r}")
    if global_batch_size is not None and (
        isinstance(global_batch_size, bool) or not isinstance(global_batch_size, Integral) or global_batch_size < 0
    ):
        raise ValueError(f"global_batch_size must be a non-negative integer, got {global_batch_size!r}")


def _per_sequence_loss_totals(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cu_seqlens is not None and (loss_mat.ndim == 1 or loss_mat.shape[0] == 1):
        flat_loss = loss_mat if loss_mat.ndim == 1 else loss_mat[0]
        flat_mask = loss_mask if loss_mask.ndim == 1 else loss_mask[0]
        _, (sequence_sums, sequence_counts) = _packed_per_sequence_sums(
            cu_seqlens,
            _masked_values(flat_loss, flat_mask),
            flat_mask.to(flat_loss.dtype),
        )
        return sequence_sums, sequence_counts
    return _masked_values(loss_mat, loss_mask).sum(dim=-1), loss_mask.sum(dim=-1).to(loss_mat.dtype)


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
    prompt_group_ids: Optional[torch.Tensor] = None,
    prompt_token_counts: Optional[torch.Tensor] = None,
    sequence_loss_weights: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Aggregate a per-token loss matrix into a scalar.

    Supports five modes:
    - ``"token-mean"`` (default): sum over all valid tokens, divide by token count.
      Equivalent to the existing ``/ loss_mask_count`` behaviour.
    - ``"seq-mean-token-sum"``: sum tokens per sequence, then mean across sequences.
    - ``"seq-mean-token-sum-norm"``: same as above, additionally divided by
      ``loss_scale_factor``. When it is omitted: for packed input
      (``cu_seqlens`` with a 1-D or single-row loss) the longest packed segment
      after summing each segment's length over sequence-parallel ranks;
      otherwise ``loss_mask.shape[-1]``,
      which under sequence parallelism is this rank's shard width. GRPO callers
      pass the packing pipeline's ``packed_loss_scale_factor`` when present.
    - ``"seq-mean-token-mean"``: mean tokens per sequence, then mean across sequences.
    - ``"prompt-mean"`` (ScaleRL): for each prompt, token-mean across all its
      responses; then mean across prompts. When ``sequence_loss_weights`` is
      present, applies DP compensation to the native PrimeRL/POC construction:
      ``dp_size * sum(sequence_weight * sequence_loss_sum / sequence_token_count)``.
      Otherwise requires ``prompt_group_ids`` (one id per sequence). Under DP,
      ``local_P`` is all-reduced to get the global prompt count.

    ``dp_size``, ``batch_num_tokens``, and ``global_batch_size`` support
    distributed normalisation when the global batch is split across DP ranks.
    ``dp_size`` is the data-parallel width and defaults to 1; it is never a
    missing/None scale. ``batch_num_tokens`` / ``global_batch_size`` still
    default to local counts when omitted. Every mode compensates for
    DeepSpeed's DP gradient averaging. Under sequence parallelism, packed
    segment boundaries must cover each token window with one segment per row.
    """
    _validate_loss_denominators(batch_num_tokens, global_batch_size)
    if loss_scale_factor is not None and (
        isinstance(loss_scale_factor, bool) or not isinstance(loss_scale_factor, Integral) or loss_scale_factor < 1
    ):
        raise ValueError(f"loss_scale_factor must be a positive integer, got {loss_scale_factor!r}")
    dp_size = _resolve_dp_size(dp_size, batch_num_tokens)
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            batch_num_tokens = loss_mask.count_nonzero() or 1
        elif batch_num_tokens == 0:
            batch_num_tokens = _explicit_zero_step_count("batch_num_tokens", loss_mask)
        loss = (torch.where(loss_mask.bool(), loss_mat, 0.0).sum() / batch_num_tokens) * dp_size

    elif loss_agg_mode in ("seq-mean-token-sum", "seq-mean-token-sum-norm"):
        seq_losses, seq_token_counts = _per_sequence_loss_totals(loss_mat, loss_mask, cu_seqlens)
        sp_group = _get_sequence_parallel_group()
        if sp_group is not None:
            dist.all_reduce(seq_token_counts, op=dist.ReduceOp.SUM, group=sp_group)
        seq_mask = (seq_token_counts > 0).float()
        if global_batch_size is None:
            global_batch_size = seq_mask.sum().clamp(min=1)
        elif global_batch_size == 0:
            global_batch_size = _explicit_zero_step_count("global_batch_size", loss_mask)
        loss = ((seq_losses * seq_mask).sum() / global_batch_size) * dp_size
        if loss_agg_mode == "seq-mean-token-sum-norm":
            if loss_scale_factor is None:
                if cu_seqlens is not None and (loss_mat.ndim == 1 or loss_mat.shape[0] == 1):
                    sequence_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(loss_mat.device)
                    if sp_group is not None:
                        dist.all_reduce(sequence_lengths, op=dist.ReduceOp.SUM, group=sp_group)
                    loss_scale_factor = int(sequence_lengths.max().clamp(min=1).item())
                else:
                    loss_scale_factor = loss_mask.shape[-1]
            loss = loss / loss_scale_factor

    elif loss_agg_mode == "seq-mean-token-mean":
        seq_loss_sums, seq_token_counts = _per_sequence_loss_totals(loss_mat, loss_mask, cu_seqlens)
        sp_group = _get_sequence_parallel_group()
        if sp_group is not None:
            dist.all_reduce(seq_token_counts, op=dist.ReduceOp.SUM, group=sp_group)
        seq_mask = (seq_token_counts > 0).float()
        seq_losses = seq_loss_sums / seq_token_counts.clamp(min=1)
        if global_batch_size is None:
            global_batch_size = seq_mask.sum().clamp(min=1)
        elif global_batch_size == 0:
            global_batch_size = _explicit_zero_step_count("global_batch_size", loss_mask)
        loss = ((seq_losses * seq_mask).sum() / global_batch_size) * dp_size

    elif loss_agg_mode == "prompt-mean":
        if global_batch_size == 0:
            global_batch_size = _explicit_zero_step_count("global_batch_size", loss_mask)
        sp_group = _get_sequence_parallel_group()
        # When sequences are packed ([1, T] with cu_seqlens present), recover
        # per-rollout sums using cu_seqlens segment boundaries, then group.
        # In the non-packed [B, S] case, sum(dim=-1) gives one value per rollout.
        # Packed input arrives as canonical 1-D [T] (the loss entry squeezes
        # pack_sequences' singleton [1, T] form) or as [1, T] from direct
        # callers — treat both as packed; only genuinely padded B > 1 rows
        # take the per-row reduction below.
        seq_sum, seq_cnt = _per_sequence_loss_totals(loss_mat, loss_mask, cu_seqlens)

        if sp_group is not None:
            dist.all_reduce(seq_cnt, op=dist.ReduceOp.SUM, group=sp_group)

        if sequence_loss_weights is not None:
            weights = sequence_loss_weights.to(loss_mat.device).to(seq_sum.dtype).reshape(-1)
            if weights.shape[0] != seq_sum.shape[0]:
                raise ValueError(
                    "sequence_loss_weights must have one value per sequence when loss_agg_mode='prompt-mean'."
                )
            loss = (weights * seq_sum / seq_cnt.clamp(min=1.0)).sum() * dp_size
            return loss

        if prompt_group_ids is None:
            raise ValueError("prompt-mean requires prompt_group_ids (one int per sequence) or sequence_loss_weights.")
        _, ids = torch.unique(
            prompt_group_ids.to(loss_mat.device).long(),
            return_inverse=True,
        )
        local_P = int(ids.max().item()) + 1 if ids.numel() > 0 else 0

        if global_batch_size is not None:
            global_num_prompts = global_batch_size
        elif dist.is_initialized() and dp_size > 1:
            t = torch.tensor(local_P, device=loss_mat.device, dtype=torch.long)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            global_num_prompts = int(t.item())
            if sp_group is not None:
                global_num_prompts //= dist.get_world_size(sp_group)
        else:
            global_num_prompts = local_P

        p_sum = torch.zeros(local_P, device=loss_mat.device, dtype=seq_sum.dtype)
        p_cnt = torch.zeros(local_P, device=loss_mat.device, dtype=seq_cnt.dtype)
        p_sum.scatter_add_(0, ids, seq_sum)
        if prompt_token_counts is None:
            p_cnt.scatter_add_(0, ids, seq_cnt)
        else:
            prompt_token_counts = prompt_token_counts.to(loss_mat.device).to(seq_cnt.dtype)
            if prompt_token_counts.shape[0] != ids.shape[0]:
                raise ValueError(
                    "prompt_token_counts must have one value per sequence when loss_agg_mode='prompt-mean'."
                )
            for local_prompt_idx in range(local_P):
                p_cnt[local_prompt_idx] = prompt_token_counts[ids == local_prompt_idx][0]
        p_mean = p_sum / p_cnt.clamp(min=1.0)

        loss = p_mean.sum() / max(global_num_prompts, 1) * dp_size

    else:
        raise ValueError(
            f"Invalid loss_agg_mode: '{loss_agg_mode}'. "
            "Expected one of: 'token-mean', 'seq-mean-token-sum', "
            "'seq-mean-token-sum-norm', 'seq-mean-token-mean', 'prompt-mean'."
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
        f"Invalid kl_penalty method: '{method}'. Expected one of: 'k1'/'kl', 'abs', 'k2'/'mse', 'k3'/'low_var_kl'."
    )


class EchoBatchDenominator(str, Enum):
    """Which step-global sequence count ``global_num_echo_sequences`` declares.

    - ``ALL_SEQUENCES`` (default): count of ALL rollout sequences in the
      optimizer step — the paper-literal reading of eq. 3 (a mean of
      per-sequence terms over the batch; sequences without observations
      contribute zero to the numerator but stay in the denominator).
    - ``ECHO_BEARING_SEQUENCES``: count of sequences carrying at least one
      observation token. Identical to ``ALL_SEQUENCES`` on domains where
      every rollout has observations; on mixed batches it keeps the
      auxiliary term's per-sequence scale independent of the fraction of
      observation-free rollouts.
    """

    ALL_SEQUENCES = "all_sequences"
    ECHO_BEARING_SEQUENCES = "echo_bearing_sequences"


def _full_observation_denominator(supplied: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(supplied):
        raise ValueError("observation_token_counts must be a tensor.")
    if supplied.ndim != 1 or supplied.shape != local.shape:
        raise ValueError(
            "observation_token_counts must have one full observation count per sequence in "
            f"this call, got {tuple(supplied.shape)} for {tuple(local.shape)} sequences."
        )
    if supplied.dtype == torch.bool or torch.is_complex(supplied):
        raise ValueError("observation_token_counts values must be real integer counts.")
    counts = supplied.to(device=local.device, dtype=local.dtype)
    if not torch.isfinite(counts).all().item():
        raise ValueError("observation_token_counts values must be finite.")
    if (counts != counts.round()).any().item():
        raise ValueError("observation_token_counts values must be integer counts.")
    if (counts < 0).any().item():
        raise ValueError("observation_token_counts values must be non-negative.")
    if (counts < local).any().item():
        raise ValueError("observation_token_counts holds a count below this call's own observation-token count.")
    return counts


def echo_env_prediction_loss_fn(
    logprobs: torch.Tensor,
    sft_mask: torch.Tensor,
    observation_mask: torch.Tensor,
    loss_mask: torch.Tensor,
    global_num_echo_sequences: int,
    batch_denominator: str = EchoBatchDenominator.ALL_SEQUENCES.value,
    cu_seqlens: torch.Tensor | None = None,
    dp_size: int = 1,
    observation_token_counts: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """ECHO Environment-Prediction auxiliary loss (https://arxiv.org/abs/2605.24517).

    Length-normalized cross-entropy on the environment-observation target
    tokens ``O'`` (``sft_mask``), normalized per sequence by the FULL
    observation length ``|O|`` (``observation_mask``) — paper eq. 3 /
    Algorithm 1 line 4. The denominator is deliberately ``|O|`` and not
    ``|O'|`` so runs with different target subsets stay comparable on a
    per-observation scale (the paper's stated design intent).

    Batch scope: the per-sequence terms are summed locally and divided by
    ``global_num_echo_sequences`` — the CLIENT-supplied global sequence count
    for the whole optimizer step (all DP ranks, all fwd-bwd chunks, all
    microbatches), analogous to ``global_batch_size`` in :func:`agg_loss`.
    There is intentionally no local-count fallback: one call only ever sees
    its local slice, and a local denominator would rescale the total gradient
    with microbatch/chunk boundaries (gradient accumulation sums per-call
    losses, so only a constant global denominator keeps the auxiliary
    gradient mass invariant to how the batch was split).

    ``dp_size`` must match the raw data-parallel width used by the policy
    aggregation. :func:`agg_loss` multiplies by that width for every
    aggregation mode, so callers scale the auxiliary term by ``dp_size`` too
    and both terms compensate for data-parallel gradient averaging identically.

    ``batch_denominator`` declares which count the client computed (see
    :class:`EchoBatchDenominator`). The division always uses the supplied
    count; the mode is used to validate it — a step-global count can never
    be smaller than the matching count of the slice in this call, so a
    violation means the declared mode and the supplied count disagree.

    Masks follow the same shifted alignment as ``loss_mask``: position ``t``
    weights ``logprobs[t]`` = log p(input_ids[t+1] | prefix). Every token
    belongs to at most ONE objective, so both masks must be disjoint from the
    policy ``loss_mask`` and ``O'`` must be contained in ``O`` — violations
    raise instead of silently double-training tokens.
    """
    try:
        batch_denominator = EchoBatchDenominator(batch_denominator)
    except ValueError:
        raise ValueError(
            f"Invalid echo_batch_denominator: '{batch_denominator}'. Expected one of: "
            + ", ".join(f"'{mode.value}'" for mode in EchoBatchDenominator)
        ) from None
    if isinstance(global_num_echo_sequences, bool) or not isinstance(global_num_echo_sequences, int):
        raise ValueError(
            f"global_num_echo_sequences must be an integer sequence count, got {global_num_echo_sequences!r}"
        )
    if global_num_echo_sequences < 1:
        raise ValueError(f"global_num_echo_sequences must be a positive count, got {global_num_echo_sequences}")
    # Exact-shape requirement: broadcasting (e.g. a [1, S] mask against [B, S]
    # logprobs) would silently train the wrong sequences instead of failing.
    if not (sft_mask.shape == observation_mask.shape == loss_mask.shape):
        raise ValueError(
            "sft_mask, echo_observation_mask, and loss_mask must have identical shapes, got "
            f"{tuple(sft_mask.shape)}, {tuple(observation_mask.shape)}, {tuple(loss_mask.shape)}."
        )
    # Two exact, mutually exclusive layouts, nothing else: padded logprobs
    # share the masks' [B, S] shape (no cu_seqlens); packed logprobs are
    # [1, T] against 1-D [T] masks — or against singleton [1, T] masks, the
    # representation ``pack_sequences`` emits and the already-packed
    # ``run_pipeline`` path forwards without squeezing (cu_seqlens required,
    # checked below). Any other combination — a transposed [S, B] with equal
    # numel, or cu_seqlens alongside padded B > 1 rows — would reshape or
    # resegment into silently wrong per-sequence attribution.
    logprobs_match_masks = logprobs.shape == sft_mask.shape or (
        logprobs.ndim == 2 and logprobs.shape[0] == 1 and sft_mask.ndim == 1 and logprobs.shape[1] == sft_mask.shape[0]
    )
    if not logprobs_match_masks:
        raise ValueError(
            f"logprobs shape {tuple(logprobs.shape)} does not match the ECHO mask shape "
            f"{tuple(sft_mask.shape)} (padded masks must equal logprobs exactly; packed "
            "[1, T] logprobs take 1-D [T] masks)."
        )
    mask_is_packed_shape = sft_mask.ndim == 1 or (sft_mask.ndim == 2 and sft_mask.shape[0] == 1)
    if cu_seqlens is not None and not mask_is_packed_shape:
        raise ValueError(
            "cu_seqlens was supplied with padded ECHO tensors — packed calls take 1-D [T] "
            "or singleton [1, T] masks; flattening already-padded rows against cu_seqlens "
            "would silently resegment them and change the per-sequence normalization."
        )
    sft_mask = sft_mask.bool()
    observation_mask = observation_mask.bool()
    loss_mask = loss_mask.bool()
    if (sft_mask & ~observation_mask).any():
        raise ValueError("sft_mask must be a subset of echo_observation_mask (O' must be contained in O).")
    if (sft_mask & loss_mask).any():
        raise ValueError("sft_mask overlaps loss_mask — every token must belong to exactly one objective.")
    if (observation_mask & loss_mask).any():
        raise ValueError("echo_observation_mask overlaps loss_mask — observations cannot be policy targets.")

    reduction_dtype = torch.float64 if logprobs.dtype == torch.float64 else torch.float32
    nll = _masked_values(-logprobs.reshape(sft_mask.shape), sft_mask).to(reduction_dtype)
    if cu_seqlens is not None:
        flat_nll = nll.reshape(-1)
        _, (seq_nll_sum, seq_obs_count, seq_policy_count) = _packed_per_sequence_sums(
            cu_seqlens,
            flat_nll,
            observation_mask.reshape(-1).to(reduction_dtype),
            loss_mask.reshape(-1).to(reduction_dtype),
        )
    elif nll.ndim == 2:
        seq_nll_sum = nll.sum(dim=-1)
        seq_obs_count = observation_mask.sum(dim=-1).to(reduction_dtype)
        seq_policy_count = loss_mask.sum(dim=-1).to(reduction_dtype)
    else:
        raise ValueError("cu_seqlens is required for packed 1D ECHO tensors.")

    sp_group = _get_sequence_parallel_group()
    if sp_group is not None:
        group_counts = torch.stack((seq_obs_count, seq_policy_count))
        dist.all_reduce(group_counts, op=dist.ReduceOp.SUM, group=sp_group)
        seq_obs_count, seq_policy_count = group_counts[0], group_counts[1]

    num_sequences = int(seq_obs_count.shape[0])
    num_echo_bearing_sequences = int((seq_obs_count > 0).sum().item())
    # A row carrying neither policy nor observation tokens contributes zero
    # gradient to every objective — that is a padding/dummy row, not a real
    # rollout. Real-rollout counts are what the client-declared step-global
    # denominator refers to, so validation uses them (a raw row count would
    # false-positive on row-padded batches).
    num_real_sequences = int(((seq_obs_count > 0) | (seq_policy_count > 0)).sum().item())
    local_matched_count = (
        num_real_sequences if batch_denominator is EchoBatchDenominator.ALL_SEQUENCES else num_echo_bearing_sequences
    )
    if global_num_echo_sequences < local_matched_count:
        raise ValueError(
            f"global_num_echo_sequences={global_num_echo_sequences} is smaller than this call's "
            f"local '{batch_denominator.value}' count ({local_matched_count}) — a step-global "
            "count can never be below one slice's count, so the declared echo_batch_denominator "
            "and the supplied count disagree."
        )

    if sp_group is not None and dist.get_rank(group=sp_group) != 0:
        num_echo_bearing_sequences = 0
        num_real_sequences = 0

    if observation_token_counts is not None:
        seq_obs_count = _full_observation_denominator(observation_token_counts, seq_obs_count)

    per_sequence_env_loss = seq_nll_sum / seq_obs_count.clamp(min=1.0)
    loss = per_sequence_env_loss.sum() / global_num_echo_sequences * dp_size

    stat = dict(
        per_sequence_env_loss=per_sequence_env_loss.detach(),
        prediction_nll_sum=seq_nll_sum.sum().detach(),
        prediction_token_count=sft_mask.sum().detach(),
        observation_token_count=observation_mask.sum().detach(),
        num_sequences=num_sequences,
        num_real_sequences=num_real_sequences,
        num_echo_bearing_sequences=num_echo_bearing_sequences,
        local_matched_count=local_matched_count,
        batch_denominator=batch_denominator,
    )
    return loss, stat


def _compute_sequence_level_ratio_and_advantages(
    log_ratio: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Give every token of a sequence that sequence's mean importance ratio and mean
    advantage.

    Under sequence parallelism a rank holds one token window of the frame, so each total
    below covers only the part of a sequence inside that window. The mean has to be the
    whole sequence's: the ratio is the exponential of it, and exp of a partial mean is not
    a factor of the whole one, so numerators and denominators are both reduced over the
    group before the division. Window boundaries carry one segment per row -- empty for a
    row the window never reaches -- so the per-sequence vectors align elementwise across
    ranks and reduce as they are.
    """
    sp_group = _get_sequence_parallel_group()
    if log_ratio.ndim == 1:
        if cu_seqlens is None:
            raise ValueError("cu_seqlens is required for 1D tensors (packed format).")
        sequence_idx, (log_ratio_sum_per_seq, advantages_sum_per_seq, valid_count_per_seq) = _packed_per_sequence_sums(
            cu_seqlens,
            torch.where(loss_mask, log_ratio, 0.0),
            torch.where(loss_mask, advantages, 0.0),
            loss_mask.int(),
        )
        if sp_group is not None:
            log_ratio_sum_per_seq, advantages_sum_per_seq, valid_count_per_seq = _sequence_parallel_sum(
                log_ratio_sum_per_seq, advantages_sum_per_seq, valid_count_per_seq, group=sp_group
            )
        # Clamping before the reduction would turn each empty window into a token,
        # inflating the divisor.
        valid_count_per_seq = valid_count_per_seq.clamp(min=1)
        log_ratio_mean_per_seq = log_ratio_sum_per_seq / valid_count_per_seq.to(log_ratio.dtype)
        adv_mean_per_seq = advantages_sum_per_seq / valid_count_per_seq.to(advantages.dtype)
        ratio = torch.exp(log_ratio_mean_per_seq)[sequence_idx]
        ratio = torch.where(loss_mask, ratio, 0.0)
        advantages = adv_mean_per_seq[sequence_idx]
        advantages = torch.where(loss_mask, advantages, 0.0)
    else:
        log_ratio_sum_per_seq = torch.where(loss_mask, log_ratio, 0.0).sum(dim=1)
        advantages_sum_per_seq = torch.where(loss_mask, advantages, 0.0).sum(dim=-1)
        valid_count_per_seq = loss_mask.sum(dim=1)
        if sp_group is not None:
            log_ratio_sum_per_seq, advantages_sum_per_seq, valid_count_per_seq = _sequence_parallel_sum(
                log_ratio_sum_per_seq, advantages_sum_per_seq, valid_count_per_seq, group=sp_group
            )
        valid_count_per_seq = valid_count_per_seq.clamp(min=1)
        seq_log_ratio_mean = log_ratio_sum_per_seq / valid_count_per_seq
        ratio = torch.exp(seq_log_ratio_mean.unsqueeze(1).expand_as(log_ratio))
        ratio = torch.where(loss_mask, ratio, 0.0)
        advantages = (advantages_sum_per_seq / valid_count_per_seq).unsqueeze(1).expand_as(log_ratio)
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
    loss_scale_factor: Optional[int] = None,
    prompt_group_ids: Optional[torch.Tensor] = None,
    prompt_token_counts: Optional[torch.Tensor] = None,
    sequence_loss_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    if importance_sampling_level == "sequence":
        log_ratio = logprobs - proximal_logprobs
        ratio, advantages = _compute_sequence_level_ratio_and_advantages(log_ratio, advantages, loss_mask, cu_seqlens)
    elif importance_sampling_level == "token":
        ratio = torch.where(loss_mask, torch.exp(logprobs - proximal_logprobs), 0)
    else:
        raise ValueError(f"Invalid importance_sampling_level: {importance_sampling_level}.")
    clipped_ratio = torch.clamp(
        ratio, 1.0 - eps_clip, 1.0 + (eps_clip if eps_clip_higher is None else eps_clip_higher)
    )
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
    behav_mask = (
        (behav_imp_weight <= behav_imp_weight_cap).logical_and(loss_mask)
        if behav_imp_weight_cap is not None
        else loss_mask
    )
    behav_kl = torch.where(behav_mask, behav_kl, 0.0)
    behav_imp_weight = torch.where(behav_mask, behav_imp_weight, 0.0)
    pg_loss = pg_loss * behav_imp_weight
    if rollout_is_weights is not None:
        pg_loss = pg_loss * rollout_is_weights
    logging_loss = pg_loss.detach()
    pg_loss = agg_loss(
        pg_loss,
        loss_mask,
        loss_agg_mode=loss_agg_mode,
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
        loss_scale_factor=loss_scale_factor,
        prompt_group_ids=prompt_group_ids,
        prompt_token_counts=prompt_token_counts,
        sequence_loss_weights=sequence_loss_weights,
        cu_seqlens=cu_seqlens,
    )
    clip_mask.logical_and_(loss_mask)
    dual_clip_mask.logical_and_(loss_mask)
    stat = dict(
        loss=logging_loss,
        importance_weight=ratio.detach(),
        approx_kl=(logprobs - proximal_logprobs).detach(),
        clip_mask=clip_mask,
        dual_clip_mask=dual_clip_mask,
    )
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
    loss_agg_mode: str = "token-mean",
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
    prompt_group_ids: Optional[torch.Tensor] = None,
    prompt_token_counts: Optional[torch.Tensor] = None,
    sequence_loss_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict]:
    if tau_pos <= 0 or tau_neg <= 0:
        raise ValueError("SAPO temperatures must be positive.")
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
    pg_loss = agg_loss(
        pg_loss,
        loss_mask,
        loss_agg_mode=loss_agg_mode,
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
        loss_scale_factor=loss_scale_factor,
        prompt_group_ids=prompt_group_ids,
        prompt_token_counts=prompt_token_counts,
        sequence_loss_weights=sequence_loss_weights,
        cu_seqlens=cu_seqlens,
    )
    stat = dict(
        loss=logging_loss,
        importance_weight=ratio.detach(),
        approx_kl=log_ratio.detach(),
        clip_mask=torch.zeros_like(loss_mask, dtype=torch.bool),
        dual_clip_mask=torch.zeros_like(loss_mask, dtype=torch.bool),
        sapo_soft_gate=soft_gate.detach(),
        sapo_scaled_gate_pos=scaled_gate_pos.detach(),
        sapo_scaled_gate_neg=scaled_gate_neg.detach(),
    )
    return pg_loss, stat


SAMPLER_LOGPROB_BIN_EDGES = (-10.0, -6.0, -4.0, -3.0, -2.0, -1.5, -1.0, -0.5, -0.2, -0.05)
LOG_RATIO_BIN_EDGES = (
    math.log(0.1),
    math.log(0.2),
    math.log(0.5),
    math.log(0.8),
    -0.1,
    -0.03,
    -0.01,
    0.01,
    0.03,
    0.1,
    math.log(1.25),
    math.log(2.0),
    math.log(5.0),
    math.log(10.0),
)
SEQ_STAT_BIN_EDGES = (-0.2, -0.05, -0.01, 0.01, 0.05, 0.2)
_JOINT_SHAPE = (2, len(SAMPLER_LOGPROB_BIN_EDGES) + 1, len(LOG_RATIO_BIN_EDGES) + 1)
_VECTOR_COUNT_NAMES = {
    "ratio_joint": tuple(
        f"ratio_joint_{row}_{column}_{sign}_token_count"
        for sign in ("pos", "neg")
        for row in range(_JOINT_SHAPE[1])
        for column in range(_JOINT_SHAPE[2])
    ),
    "seq_stat_bin": tuple(f"seq_stat_bin_{index}_sequence_count" for index in range(len(SEQ_STAT_BIN_EDGES) + 1)),
}
RATIO_MASK_CONFIG_KEYS = frozenset(
    {
        "ratio_mask_bounds_pos",
        "ratio_mask_bounds_neg",
        "prob_diff_mask_max_pos",
        "prob_diff_mask_max_neg",
        "seq_mask_stat",
        "seq_mask_bounds_pos",
        "seq_mask_bounds_neg",
        "log_ratio_sq_coef",
        "ratio_m2_threshold",
        "ratio_stats",
    }
)


def _config_bounds(config: dict, key: str, *, nonnegative: bool) -> tuple[float, float] | None:
    if key not in config:
        return None
    value = config[key]
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(
            bound is not None
            and (isinstance(bound, bool) or not isinstance(bound, (int, float)) or not math.isfinite(bound))
            for bound in value
        )
    ):
        raise ValueError(f"{key} must be a [low, high] pair, got {value!r}")
    low = (0.0 if nonnegative else -math.inf) if value[0] is None else float(value[0])
    high = math.inf if value[1] is None else float(value[1])
    if not (low < high) or (nonnegative and low < 0.0):
        raise ValueError(f"{key} must have low < high{' and low >= 0' if nonnegative else ''}, got {value!r}")
    return low, high


def _config_nonnegative(config: dict, key: str) -> float | None:
    if key not in config:
        return None
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{key} must be a finite non-negative number, got {value!r}")
    return float(value)


def _config_positive(config: dict, key: str) -> float | None:
    if key not in config:
        return None
    value = config[key]
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{key} must be a finite positive number, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class RatioMasks:
    ratio_bounds_pos: tuple[float, float] | None = None
    ratio_bounds_neg: tuple[float, float] | None = None
    prob_diff_max_pos: float | None = None
    prob_diff_max_neg: float | None = None
    seq_stat: str = "mean_log_ratio"
    seq_bounds_pos: tuple[float, float] | None = None
    seq_bounds_neg: tuple[float, float] | None = None
    log_ratio_sq_coef: float = 0.0
    m2_threshold: float | None = None

    @classmethod
    def from_config(cls, config: dict) -> RatioMasks | None:
        if "ratio_stats" in config and not isinstance(config["ratio_stats"], bool):
            raise ValueError(f"ratio_stats must be a bool, got {config['ratio_stats']!r}")
        if not config.get("ratio_stats", False) and not (RATIO_MASK_CONFIG_KEYS - {"ratio_stats"}) & config.keys():
            return None
        seq_stat = config.get("seq_mask_stat", "mean_log_ratio")
        if seq_stat not in ("mean_log_ratio", "mean_k3"):
            raise ValueError(f"seq_mask_stat must be one of mean_log_ratio, mean_k3, got {seq_stat!r}")
        seq_bounds_pos = _config_bounds(config, "seq_mask_bounds_pos", nonnegative=False)
        seq_bounds_neg = _config_bounds(config, "seq_mask_bounds_neg", nonnegative=False)
        return cls(
            ratio_bounds_pos=_config_bounds(config, "ratio_mask_bounds_pos", nonnegative=True),
            ratio_bounds_neg=_config_bounds(config, "ratio_mask_bounds_neg", nonnegative=True),
            prob_diff_max_pos=_config_nonnegative(config, "prob_diff_mask_max_pos"),
            prob_diff_max_neg=_config_nonnegative(config, "prob_diff_mask_max_neg"),
            seq_stat=seq_stat,
            seq_bounds_pos=seq_bounds_pos,
            seq_bounds_neg=seq_bounds_neg,
            log_ratio_sq_coef=_config_nonnegative(config, "log_ratio_sq_coef") or 0.0,
            m2_threshold=_config_positive(config, "ratio_m2_threshold"),
        )

    def echo(self) -> dict[str, float]:
        result = {"ratio_masks_contract_version": 1.0}
        for name, bounds in (
            ("ratio_mask_pos", self.ratio_bounds_pos),
            ("ratio_mask_neg", self.ratio_bounds_neg),
            ("seq_mask_pos", self.seq_bounds_pos),
            ("seq_mask_neg", self.seq_bounds_neg),
        ):
            if bounds is not None:
                for side, bound in zip(("low", "high"), bounds):
                    metric_name = f"{name}_{side}" if math.isfinite(bound) else f"{name}_{side}_unbounded"
                    result[metric_name] = bound if math.isfinite(bound) else 1.0
        for name, value in (
            ("prob_diff_mask_max_pos", self.prob_diff_max_pos),
            ("prob_diff_mask_max_neg", self.prob_diff_max_neg),
            ("ratio_m2_threshold", self.m2_threshold),
        ):
            if value is not None:
                result[name] = value
        result[f"seq_mask_stat_{self.seq_stat}"] = 1.0
        result["log_ratio_sq_coef"] = self.log_ratio_sq_coef
        return result


@torch.no_grad()
def _ratio_mask_keep(
    masks: RatioMasks,
    logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    cu_seqlens: torch.Tensor | None,
    m2_keep: torch.Tensor | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    log_ratio = torch.where(loss_mask, logprobs.detach() - old_logprobs, torch.zeros_like(old_logprobs))
    # ``advantages`` is token-level here; a zero advantage counts as positive for every sign-split gate below.
    sides = {"pos": loss_mask & (advantages >= 0), "neg": loss_mask & (advantages < 0)}
    drops: dict[str, torch.Tensor] = {}
    for sign, bounds in (("pos", masks.ratio_bounds_pos), ("neg", masks.ratio_bounds_neg)):
        if bounds is not None:
            log_low = math.log(bounds[0]) if bounds[0] > 0.0 else -math.inf
            drops[f"ratio_mask_{sign}_low_drop"] = sides[sign] & (log_ratio < log_low)
            drops[f"ratio_mask_{sign}_high_drop"] = sides[sign] & (log_ratio > math.log(bounds[1]))
    if masks.prob_diff_max_pos is not None or masks.prob_diff_max_neg is not None:
        prob_gap = logprobs.detach().exp() - old_logprobs.exp()
        for sign, limit, gap in (
            ("pos", masks.prob_diff_max_pos, prob_gap),
            ("neg", masks.prob_diff_max_neg, -prob_gap),
        ):
            if limit is not None:
                drops[f"prob_diff_mask_{sign}_drop"] = sides[sign] & (gap > limit)
    values = [log_ratio, sides["pos"].float(), sides["neg"].float()]
    if masks.seq_stat == "mean_k3":
        values.append(log_ratio.expm1() - log_ratio)
    packed = cu_seqlens is not None
    if packed:
        sequence_idx, totals = _packed_per_sequence_sums(cu_seqlens, *values)
    else:
        totals = [value.sum(dim=-1) for value in values]
    sp_group = _get_sequence_parallel_group()
    if sp_group is not None:
        totals = _sequence_parallel_sum(*totals, group=sp_group)
    log_sum, pos_count, neg_count = totals[:3]
    token_count = pos_count + neg_count
    mean_log_ratio = log_sum / token_count.clamp(min=1)
    seq_stat = totals[-1] / token_count.clamp(min=1) if masks.seq_stat == "mean_k3" else mean_log_ratio
    sequence_dropped = torch.zeros_like(token_count, dtype=torch.bool)
    counts: dict[str, torch.Tensor] = {}
    for sign, bounds, sign_count in (
        ("pos", masks.seq_bounds_pos, pos_count),
        ("neg", masks.seq_bounds_neg, neg_count),
    ):
        if bounds is not None:
            hit = (sign_count > 0) & ((seq_stat < bounds[0]) | (seq_stat > bounds[1]))
            sequence_dropped |= hit
            counts[f"seq_mask_{sign}_sequence_count"] = hit.sum()
            token_hit = hit[sequence_idx].reshape_as(loss_mask) if packed else hit.unsqueeze(-1)
            drops[f"seq_mask_{sign}_drop"] = sides[sign] & token_hit
    # Always bins the mean log ratio, also when ``seq_mask_stat="mean_k3"`` gates on another statistic.
    counts["seq_stat_bin"] = torch.zeros(
        len(SEQ_STAT_BIN_EDGES) + 1, dtype=torch.int64, device=log_ratio.device
    ).scatter_add_(
        0,
        torch.bucketize(mean_log_ratio, log_ratio.new_tensor(SEQ_STAT_BIN_EDGES)).reshape(-1),
        (token_count > 0).long().reshape(-1),
    )
    counts["seq_mask_dropped_sequence_count"] = sequence_dropped.sum()
    if sp_group is not None and dist.get_rank(group=sp_group) != 0:
        counts = {name: torch.zeros_like(value) for name, value in counts.items()}
    if m2_keep is not None:
        drops["ratio_m2_drop"] = loss_mask & ~m2_keep
    dropped = torch.zeros_like(loss_mask)
    for hit in drops.values():
        dropped |= hit
    mu_bins = torch.bucketize(
        torch.where(loss_mask, old_logprobs, torch.zeros_like(old_logprobs)),
        log_ratio.new_tensor(SAMPLER_LOGPROB_BIN_EDGES),
    )
    cells = ((advantages < 0).long() * _JOINT_SHAPE[1] + mu_bins) * _JOINT_SHAPE[2] + torch.bucketize(
        log_ratio, log_ratio.new_tensor(LOG_RATIO_BIN_EDGES)
    )
    counts["ratio_joint"] = torch.zeros(math.prod(_JOINT_SHAPE), dtype=torch.int64, device=cells.device).scatter_add_(
        0, cells.reshape(-1), loss_mask.long().reshape(-1)
    )
    counts["ratio_trainable_token_count"] = loss_mask.sum()
    counts.update({f"{name}_count": hit.sum() for name, hit in drops.items()})
    counts["ratio_mask_dropped_token_count"] = dropped.sum()
    return loss_mask & ~dropped, counts


def _reduce_ratio_mask_counts(counts: dict[str, torch.Tensor]) -> dict[str, float]:
    names = [name for key in sorted(counts) for name in _VECTOR_COUNT_NAMES.get(key, (key,))]
    stacked = torch.cat([counts[key].detach().reshape(-1).to(torch.float64) for key in sorted(counts)])
    sp_group = _get_sequence_parallel_group()
    if sp_group is not None:
        dist.all_reduce(stacked, group=sp_group)
        if dist.get_rank(group=sp_group) != 0:
            stacked = torch.zeros_like(stacked)
    return dict(zip(names, stacked.tolist()))


def cispo_actor_loss_fn(
    logprobs: torch.Tensor,
    proximal_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    loss_mask: torch.Tensor,
    eps_clip_higher: float | None = None,
    is_weight_clip_max: float | None = None,
    behav_imp_weight_cap: float | None = None,
    importance_sampling_level: str = "token",
    cu_seqlens: torch.Tensor | None = None,
    loss_agg_mode: str = "token-mean",
    rollout_is_weights: torch.Tensor | None = None,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
    prompt_group_ids: Optional[torch.Tensor] = None,
    prompt_token_counts: Optional[torch.Tensor] = None,
    sequence_loss_weights: Optional[torch.Tensor] = None,
    nll_mask: torch.Tensor | None = None,
    ratio_masks: RatioMasks | None = None,
    ratio_m2_keep: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict]:
    """CISPO policy loss (https://arxiv.org/abs/2506.13585).

    Replaces PPO's ``min(r*A, clip(r)*A)`` with ``-sg(clip(r)) * A * log π_θ``:
    the IS ratio is clipped AND stop-gradient'd; gradient flows only through
    the current-policy log-prob, so every token — including clipped ones —
    contributes a non-zero gradient (unlike PPO whose ``min()`` zeroes it for
    clipped tokens).

    When ``is_weight_clip_max`` is set, upper-cap the stop-gradient
    importance weight at that max with no lower clip. Otherwise fall back to
    the older PPO-style epsilon band.
    """
    objective_mask = loss_mask
    nll_loss = None
    if nll_mask is not None:
        nll_loss = -_safe_masked_operand(logprobs, nll_mask)
        loss_mask = loss_mask & ~nll_mask
        proximal_logprobs = _safe_masked_operand(proximal_logprobs, loss_mask)
        old_logprobs = _safe_masked_operand(old_logprobs, loss_mask)
    # Gates split on the token-level advantage, before sequence-level IS replaces it below.
    mask_advantages = advantages
    if importance_sampling_level == "sequence":
        log_ratio = torch.where(loss_mask, logprobs - proximal_logprobs, torch.zeros_like(logprobs))
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
        ratio, advantages = _compute_sequence_level_ratio_and_advantages(log_ratio, advantages, loss_mask, cu_seqlens)
    elif importance_sampling_level == "token":
        log_ratio = torch.where(loss_mask, logprobs - proximal_logprobs, torch.zeros_like(logprobs))
        log_ratio = torch.clamp(log_ratio, min=-20.0, max=20.0)
        ratio = torch.where(loss_mask, torch.exp(log_ratio), torch.zeros_like(log_ratio))
    else:
        raise ValueError(f"Invalid importance_sampling_level: {importance_sampling_level}.")
    policy_mask = loss_mask
    if ratio_masks is not None:
        if ratio_masks.log_ratio_sq_coef > 0.0 and loss_agg_mode == "token-mean" and sequence_loss_weights is not None:
            raise ValueError("log_ratio_sq_coef with token-mean cannot carry sequence_loss_weights to the penalty")
        policy_mask, ratio_mask_counts = _ratio_mask_keep(
            ratio_masks, logprobs, old_logprobs, mask_advantages, loss_mask, cu_seqlens, ratio_m2_keep
        )
        penalty_logprobs = logprobs if ratio_masks.log_ratio_sq_coef > 0.0 else logprobs.detach()
        log_ratio_sq = torch.where(loss_mask, penalty_logprobs - old_logprobs, 0.0).square()
    advantages = _safe_masked_operand(advantages, loss_mask)
    logprobs = _safe_masked_operand(logprobs, loss_mask)

    if is_weight_clip_max is not None:
        if is_weight_clip_max <= 0.0:
            raise ValueError(f"is_weight_clip_max must be positive, got {is_weight_clip_max}")
        clipped_ratio = torch.clamp(ratio, max=is_weight_clip_max)
    else:
        eps_high = eps_clip if eps_clip_higher is None else eps_clip_higher
        clipped_ratio = torch.clamp(ratio, 1.0 - eps_clip, 1.0 + eps_high)
    clip_mask = (ratio != clipped_ratio).logical_and(loss_mask)

    # CISPO: -sg(clip(r)) * A * log π_θ — gradient flows ONLY through `logprobs`
    pg_loss = _masked_values(-clipped_ratio.detach() * advantages * logprobs, policy_mask)

    # Behavioral IS correction (decoupled PPO). Identical plumbing to PPO.
    behav_kl = torch.where(loss_mask, proximal_logprobs - old_logprobs, torch.zeros_like(proximal_logprobs))
    behav_imp_weight = behav_kl.exp()
    behav_mask = (
        (behav_imp_weight <= behav_imp_weight_cap).logical_and(loss_mask)
        if behav_imp_weight_cap is not None
        else loss_mask
    )
    behav_kl = torch.where(behav_mask, behav_kl, 0.0)
    behav_imp_weight = torch.where(behav_mask, behav_imp_weight, 0.0)
    pg_loss = pg_loss * behav_imp_weight
    if rollout_is_weights is not None:
        pg_loss = pg_loss * _safe_masked_operand(rollout_is_weights, loss_mask)
    pg_loss = _masked_values(pg_loss, loss_mask)

    if nll_loss is not None:
        pg_loss = torch.where(nll_mask, nll_loss, pg_loss)

    logging_loss = pg_loss.detach()
    if ratio_masks is not None and ratio_masks.log_ratio_sq_coef > 0.0:
        pg_loss = pg_loss + ratio_masks.log_ratio_sq_coef * log_ratio_sq
    pg_loss = agg_loss(
        pg_loss,
        objective_mask,
        loss_agg_mode=loss_agg_mode,
        dp_size=dp_size,
        batch_num_tokens=batch_num_tokens,
        global_batch_size=global_batch_size,
        loss_scale_factor=loss_scale_factor,
        prompt_group_ids=prompt_group_ids,
        prompt_token_counts=prompt_token_counts,
        sequence_loss_weights=sequence_loss_weights,
        cu_seqlens=cu_seqlens,
    )

    stat = dict(
        loss=logging_loss,
        importance_weight=ratio.detach(),
        approx_kl=log_ratio.detach(),
        clip_mask=clip_mask,
        dual_clip_mask=torch.zeros_like(loss_mask, dtype=torch.bool),
        behave_imp_weight=behav_imp_weight,
        behave_approx_kl=behav_kl,
        behave_mask=behav_mask,
    )
    if is_weight_clip_max is not None:
        stat["clipped_is_weight"] = clipped_ratio.detach()
        stat["lower_tail_mask"] = (
            (ratio < 1.0 / is_weight_clip_max) & loss_mask if is_weight_clip_max > 1.0 else torch.zeros_like(loss_mask)
        )
    if ratio_masks is not None:
        ratio_mask_counts["ratio_mask_dropped_weight_sum"] = torch.where(
            loss_mask & ~policy_mask, clipped_ratio.detach(), 0.0
        ).sum()
        ratio_mask_counts["log_ratio_sq_sum"] = log_ratio_sq.detach().sum()
        stat["ratio_mask_metrics"] = {**ratio_masks.echo(), **_reduce_ratio_mask_counts(ratio_mask_counts)}
    return pg_loss, stat
