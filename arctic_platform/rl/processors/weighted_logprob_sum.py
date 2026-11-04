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

"""Direct per-token log-probability gradient objective.

The caller supplies prediction-aligned ``logprob_weights = dL/dlogprobs``.
This loss forms their exact float32 dot product with the current model's
log-probabilities, avoiding policy-ratio/clipping semantics in surrogate RL
losses. It is useful when a client evaluates and differentiates a sequence-level
objective such as DPO while the service owns model backpropagation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import torch

from arctic_platform.common.registry import declare_loss_capabilities
from arctic_platform.common.registry import register_loss_fn

from .base_loss import REQUIRES_TOKEN_LOGPROBS
from .functional import _resolve_dp_size
from .functional import canonicalize_loss_mask
from .functional import resolve_global_loss_scale
from .packed_reduction import PackedLossReduction
from .packed_reduction import additive_packed_loss_reduction

_ALLOWED_CONFIG_KEYS = frozenset({"batch_num_tokens", "dp_size"})
_IGNORED_CONFIG_KEYS = frozenset({"global_batch_size"})


def _packed_singleton_to_1d(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is not None and tensor.ndim == 2 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def _connected_zero(tensor: torch.Tensor) -> torch.Tensor:
    """Return finite zero while retaining a minimal autograd connection."""
    first = tensor.float().reshape(-1)[:1]
    return torch.nan_to_num(first, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0


def _resolve_config(config: dict, batch: dict, meta: dict) -> int:
    """Validate scale metadata and return the data-parallel multiplier."""
    unknown_keys = set(config) - _ALLOWED_CONFIG_KEYS - _IGNORED_CONFIG_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown config keys for loss_fn 'weighted_logprob_sum': {sorted(unknown_keys)}")
    scale = resolve_global_loss_scale({**meta, **batch}, config)
    batch_num_tokens = scale.get("batch_num_tokens")
    if batch_num_tokens is not None and (
        isinstance(batch_num_tokens, bool)
        or not isinstance(batch_num_tokens, Real)
        or not math.isfinite(float(batch_num_tokens))
        or float(batch_num_tokens) < 0
    ):
        raise ValueError("weighted_logprob_sum batch_num_tokens must be finite and non-negative")
    # The backend stamps batch_num_tokens for every masked objective. It is
    # validation metadata here, not a denominator: logprob_weights already are
    # the caller's complete dL/dlogprobs values.
    return _resolve_dp_size(scale.get("dp_size"), batch_num_tokens)


def _validate_context(
    context: dict,
    logprobs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    loss_mask = context.get("loss_mask")
    if loss_mask is None:
        raise ValueError("weighted_logprob_sum requires context['loss_mask']")
    mask = canonicalize_loss_mask(
        loss_mask,
        logprobs,
        objective="weighted_logprob_sum",
        binary=True,
    )

    weights = context.get("logprob_weights")
    if not torch.is_tensor(weights):
        raise ValueError("weighted_logprob_sum requires tensor context['logprob_weights']")
    if weights.dtype != torch.float32:
        raise ValueError(f"weighted_logprob_sum logprob_weights must use float32, got {weights.dtype}")
    if tuple(weights.shape) != tuple(logprobs.shape):
        raise ValueError(
            "weighted_logprob_sum logprob_weights must match logprobs, got "
            f"weights={tuple(weights.shape)} logprobs={tuple(logprobs.shape)}"
        )
    # Client-supplied derivatives are constants for the server-side VJP.
    weights = weights.detach().to(device=logprobs.device)
    if not torch.isfinite(weights).all().item():
        raise ValueError("weighted_logprob_sum logprob_weights must be finite")
    if (weights.masked_select(~mask) != 0).any().item():
        raise ValueError("weighted_logprob_sum logprob_weights must be zero outside loss_mask")

    temperature = context.get("temperature")
    if temperature is not None:
        if torch.is_tensor(temperature):
            neutral_temperature = (
                not torch.is_complex(temperature)
                and torch.isfinite(temperature).all().item()
                and (temperature == 1).all().item()
            )
        else:
            neutral_temperature = (
                isinstance(temperature, Real)
                and not isinstance(temperature, bool)
                and math.isfinite(float(temperature))
                and float(temperature) == 1.0
            )
        if not neutral_temperature:
            raise ValueError(
                "weighted_logprob_sum requires untempered model log probabilities; "
                "omit temperature or set every value to 1"
            )
    if context.get("action_masks") is not None:
        raise ValueError("weighted_logprob_sum does not support vocabulary action_masks")

    labels = context.get("labels")
    if torch.is_tensor(labels):
        if tuple(labels.shape) != tuple(mask.shape):
            raise ValueError("weighted_logprob_sum labels must match loss_mask when present")
        if ((labels.to(mask.device) == -100) & mask).any().item():
            raise ValueError("weighted_logprob_sum loss_mask must be zero where labels use ignore_index=-100")
    return weights, mask


def _weighted_logprob_sum_packed_reduction(
    microbatches: Sequence[dict],
    config: dict,
    loss_fn_name: str,
) -> PackedLossReduction:
    del loss_fn_name
    reporting_weights: list[float] = []
    for index, microbatch in enumerate(microbatches):
        reference = microbatch.get("input_ids")
        if not torch.is_tensor(reference):
            raise ValueError(
                f"weighted_logprob_sum packed microbatch {index} requires tensor input_ids for preflight validation"
            )
        _, loss_mask = _validate_context(microbatch, reference)
        # Use active-token count only for reporting. Every microbatch loss is an
        # already-normalized additive contribution and therefore has scale 1.
        reporting_weights.append(float(loss_mask.sum().item()))

    _resolve_config(config, microbatches[0] if microbatches else {}, {})
    return additive_packed_loss_reduction(reporting_weights)


def _weighted_logprob_sum_metrics_callback(worker_metrics: Sequence[dict], metrics: dict) -> None:
    """Pool the additive surrogate metrics across data-parallel workers."""
    total = sum(float(worker["loss.sum"]) for worker in worker_metrics)
    tokens = sum(float(worker["loss.tokens"]) for worker in worker_metrics)
    metrics.update({"loss.sum": total, "loss.tokens": tokens, "loss": total / tokens if tokens else 0.0})


@register_loss_fn(
    "weighted_logprob_sum",
    summed_metrics=("loss.sum", "loss.tokens"),
    metrics_callback=_weighted_logprob_sum_metrics_callback,
    packed_loss_reduction=_weighted_logprob_sum_packed_reduction,
)
@declare_loss_capabilities(REQUIRES_TOKEN_LOGPROBS)
def weighted_logprob_sum_loss(
    model_outputs: dict,
    batch: dict,
    meta: dict,
    config: dict,
    device: str,
) -> tuple[torch.Tensor, dict]:
    """Return ``dp_size * sum(logprob_weights * current_logprobs)``.

    DeepSpeed averages gradients across data-parallel replicas. Multiplication
    by ``dp_size`` makes the resulting averaged gradient equal the sum of the
    caller-supplied global-objective token gradients across disjoint shards.
    """
    del device
    dp_size = _resolve_config(config, batch, meta)
    context = {**meta, **batch}

    logprobs = model_outputs.get("logprobs")
    if logprobs is None:
        raise ValueError(
            "weighted_logprob_sum requires model_outputs['logprobs']; configure processing.post=['compute_logprobs']"
        )
    if context.get("cu_seqlens") is not None:
        logprobs = _packed_singleton_to_1d(logprobs)
        context = {
            **context,
            "loss_mask": _packed_singleton_to_1d(context.get("loss_mask")),
            "logprob_weights": _packed_singleton_to_1d(context.get("logprob_weights")),
            "labels": _packed_singleton_to_1d(context.get("labels")),
        }

    weights, mask = _validate_context(context, logprobs)
    invalid = ~torch.isfinite(logprobs)
    if invalid.any().item():
        first_index = tuple(int(i) for i in invalid.nonzero(as_tuple=False)[0].tolist())
        raise ValueError(f"weighted_logprob_sum found a non-finite logprob at index={first_index}")

    active_tokens = float(mask.sum().item())
    if active_tokens == 0:
        return _connected_zero(logprobs), {"loss.sum": 0.0, "loss.tokens": 0.0}

    safe_logprobs = logprobs.float().masked_fill(~mask, 0.0)
    local_dot_product = (safe_logprobs * weights).sum()
    loss = local_dot_product * dp_size
    if not torch.isfinite(loss).item():
        raise ValueError("weighted_logprob_sum produced a non-finite weighted sum")
    metrics = {
        "loss.sum": float(local_dot_product.detach().item()),
        "loss.tokens": active_tokens,
    }
    return loss, metrics
