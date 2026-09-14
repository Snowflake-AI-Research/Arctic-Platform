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

"""Weighted causal cross-entropy for supervised updates through the RL path."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import torch

from arctic_platform.common.registry import register_loss_fn

from .functional import _resolve_dp_size
from .functional import canonicalize_loss_mask
from .packed_reduction import PackedLossReduction
from .packed_reduction import additive_packed_loss_reduction
from .packed_reduction import local_mean_packed_loss_reduction

_ALLOWED_CONFIG_KEYS = frozenset({"batch_num_tokens", "dp_size"})
_IGNORED_CONFIG_KEYS = frozenset({"global_batch_size"})


def _packed_singleton_to_1d(tensor: torch.Tensor | None) -> torch.Tensor | None:
    if tensor is not None and tensor.ndim == 2 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def _connected_zero(tensor: torch.Tensor) -> torch.Tensor:
    """Return finite zero while keeping a minimal autograd connection."""
    first = tensor.float().reshape(-1)[:1]
    return torch.nan_to_num(first, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0


def _merge_distributed_config(config: dict, batch: dict, meta: dict) -> dict:
    """Fill missing scale keys from meta/batch; config wins when set."""
    cfg = dict(config)
    for key in ("dp_size", "batch_num_tokens"):
        if cfg.get(key) is None:
            if meta.get(key) is not None:
                cfg[key] = meta[key]
            elif batch.get(key) is not None:
                cfg[key] = batch[key]
    return cfg


def _validate_neutral_temperature(context: dict) -> None:
    temperature = context.get("temperature")
    if temperature is None:
        return
    if torch.is_tensor(temperature):
        if torch.is_complex(temperature) or not torch.isfinite(temperature).all().item():
            raise ValueError("causal_cross_entropy temperature must be finite and real")
        neutral = (temperature == 1).all().item()
    else:
        neutral = (
            isinstance(temperature, Real)
            and not isinstance(temperature, bool)
            and math.isfinite(float(temperature))
            and float(temperature) == 1.0
        )
    if not neutral:
        raise ValueError(
            "causal_cross_entropy requires untempered model log probabilities; "
            "omit temperature or set every value to 1"
        )


def _validate_global_normalization(config: dict, local_weight_sum: float) -> tuple[float | None, int]:
    unknown_keys = set(config) - _ALLOWED_CONFIG_KEYS - _IGNORED_CONFIG_KEYS
    if unknown_keys:
        raise ValueError(
            f"Unknown config keys for loss_fn 'causal_cross_entropy': {sorted(unknown_keys)}"
        )

    global_weight_sum = config.get("batch_num_tokens")
    dp_size = config.get("dp_size")
    if global_weight_sum is None:
        if dp_size is not None:
            raise ValueError("causal_cross_entropy config 'dp_size' requires 'batch_num_tokens'")
        return None, 1

    if (
        isinstance(global_weight_sum, bool)
        or not isinstance(global_weight_sum, Real)
        or not math.isfinite(float(global_weight_sum))
        or float(global_weight_sum) < 0
    ):
        raise ValueError(
            "causal_cross_entropy batch_num_tokens must be a finite non-negative "
            f"global loss-weight sum, got {global_weight_sum!r}"
        )
    dp_size = _resolve_dp_size(dp_size, global_weight_sum)
    if (
        isinstance(dp_size, bool)
        or not isinstance(dp_size, Real)
        or not math.isfinite(float(dp_size))
        or float(dp_size) <= 0
        or not float(dp_size).is_integer()
    ):
        raise ValueError(
            "causal_cross_entropy config 'dp_size' must be a positive integer "
            "when batch_num_tokens is supplied"
        )
    dp_size = int(dp_size)

    global_weight_sum = float(global_weight_sum)
    if local_weight_sum > global_weight_sum and not math.isclose(
        local_weight_sum,
        global_weight_sum,
        rel_tol=1e-6,
        abs_tol=1e-6,
    ):
        raise ValueError(
            "causal_cross_entropy local loss-weight sum exceeds the configured "
            f"global sum: local={local_weight_sum} global={global_weight_sum}"
        )
    return global_weight_sum, dp_size


def _validate_context_weights(context: dict, reference: torch.Tensor) -> torch.Tensor:
    loss_mask = context.get("loss_mask")
    if loss_mask is None:
        raise ValueError("causal_cross_entropy requires context['loss_mask']")
    weights = canonicalize_loss_mask(
        loss_mask,
        reference,
        objective="causal_cross_entropy",
        binary=False,
    )
    _validate_neutral_temperature(context)
    if context.get("action_masks") is not None:
        raise ValueError("causal_cross_entropy does not support vocabulary action_masks")

    labels = context.get("labels")
    if torch.is_tensor(labels):
        if tuple(labels.shape) != tuple(weights.shape):
            raise ValueError("causal_cross_entropy labels must match loss_mask when labels are present")
        if ((labels.to(weights.device) == -100) & (weights > 0)).any().item():
            raise ValueError(
                "causal_cross_entropy loss_mask must be zero where labels use ignore_index=-100"
            )
    return weights


def _causal_cross_entropy_packed_reduction(
    microbatches: Sequence[dict],
    config: dict,
    loss_fn_name: str,
) -> PackedLossReduction:
    del loss_fn_name
    weights = []
    for index, microbatch in enumerate(microbatches):
        reference = microbatch.get("input_ids")
        if not torch.is_tensor(reference):
            raise ValueError(
                "causal_cross_entropy packed microbatch "
                f"{index} requires tensor input_ids for preflight validation"
            )
        loss_weights = _validate_context_weights(microbatch, reference)
        weights.append(float(loss_weights.sum(dtype=torch.float32).item()))

    cfg = dict(config)
    first = microbatches[0] if microbatches else {}
    for key in ("dp_size", "batch_num_tokens"):
        if cfg.get(key) is None and first.get(key) is not None:
            cfg[key] = first[key]
    global_weight_sum, _ = _validate_global_normalization(cfg, sum(weights))
    if global_weight_sum is None:
        return local_mean_packed_loss_reduction(weights)
    return additive_packed_loss_reduction(weights)


@register_loss_fn(
    "causal_cross_entropy",
    packed_loss_reduction=_causal_cross_entropy_packed_reduction,
)
def causal_cross_entropy_loss(
    model_outputs: dict,
    batch: dict,
    meta: dict,
    config: dict,
    device: str,
) -> tuple[torch.Tensor, dict]:
    """Compute prediction-aligned weighted next-token negative log-likelihood."""
    del device
    config = _merge_distributed_config(config, batch, meta)
    context = {**meta, **batch}

    logprobs = model_outputs.get("logprobs")
    if logprobs is None:
        raise ValueError(
            "causal_cross_entropy requires model_outputs['logprobs']; configure "
            "processing.post=['compute_logprobs']"
        )
    if context.get("cu_seqlens") is not None:
        logprobs = _packed_singleton_to_1d(logprobs)
        context = {
            **context,
            "loss_mask": _packed_singleton_to_1d(context.get("loss_mask")),
            "labels": _packed_singleton_to_1d(context.get("labels")),
        }

    weights = _validate_context_weights(context, logprobs)

    active = weights > 0
    local_weight_sum_tensor = weights.sum(dtype=torch.float32)
    local_weight_sum = float(local_weight_sum_tensor.detach().item())
    global_weight_sum, dp_size = _validate_global_normalization(config, local_weight_sum)

    if not active.any().item():
        return _connected_zero(logprobs), {}

    invalid = active & ~torch.isfinite(logprobs)
    if invalid.any().item():
        invalid_indices = invalid.nonzero(as_tuple=False)
        first_index = tuple(int(i) for i in invalid_indices[0].tolist())
        invalid_count = invalid_indices.shape[0]
        active_count = int(active.sum().item())
        first_logprob = logprobs[first_index].detach().item()
        first_weight = weights[first_index].detach().item()
        raise ValueError(
            "causal_cross_entropy found "
            f"{invalid_count} non-finite logprob value(s) among {active_count} "
            f"positive-weight positions (shape={tuple(logprobs.shape)}); first "
            f"invalid index={first_index}, logprob={first_logprob!r}, "
            f"loss_weight={first_weight!r}. Non-finite logprobs are ignored only "
            "where loss_mask is 0. Check model logits and attention masks, and "
            "ensure padding or sequence-parallel tail positions have loss_mask=0."
        )

    safe_logprobs = logprobs.float().masked_fill(~active, 0.0)
    nll = -safe_logprobs
    numerator = (nll * weights).sum()
    if global_weight_sum is None:
        loss = numerator / local_weight_sum_tensor
    elif global_weight_sum == 0:
        loss = _connected_zero(logprobs)
    else:
        loss = numerator * dp_size / global_weight_sum
    return loss, {}
