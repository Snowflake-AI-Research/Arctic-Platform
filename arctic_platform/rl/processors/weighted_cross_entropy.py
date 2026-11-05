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

"""Signed per-token weighted log-likelihood, unnormalized.

``weighted_cross_entropy`` trains ``L_surrogate = -sum(w * logprobs)`` over the whole request, where ``w`` is
``context["logprob_weights_shifted"]``, so ``dL_surrogate/dlogprobs = -w``. A caller with a custom loss ``L``
computes ``g = dL/dlogprobs`` and sends ``w = -g``; the model gradient is then that of ``L``. With ``w = loss_mask``
it is the summed causal cross-entropy.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from arctic_platform.common.registry import declare_loss_capabilities
from arctic_platform.common.registry import register_loss_fn
from arctic_platform.rl.processors.base_loss import REQUIRES_TOKEN_LOGPROBS

from .causal_cross_entropy import _packed_singleton_to_1d
from .causal_cross_entropy import _validate_neutral_temperature
from .functional import _GLOBAL_LOSS_SCALE_KEYS
from .functional import _resolve_dp_size
from .functional import canonicalize_loss_mask
from .functional import resolve_global_loss_scale
from .grpo import _request_grpo_contexts
from .packed_reduction import PackedLossReduction
from .packed_reduction import additive_packed_loss_reduction

_NAME = "weighted_cross_entropy"
WEIGHTS_KEY = "logprob_weights_shifted"
OBJECTIVE_SUM_METRIC = "weighted_cross_entropy_objective_sum"
WEIGHTED_TOKEN_COUNT_METRIC = "weighted_cross_entropy_weighted_token_count"


def _validate_config(config: dict) -> None:
    # The global loss-scale keys are stamped by the server; the objective itself has no options.
    unknown_keys = set(config) - set(_GLOBAL_LOSS_SCALE_KEYS)
    if unknown_keys:
        raise ValueError(
            f"Unknown config keys for loss_fn {_NAME!r}: {sorted(unknown_keys)}; it takes no config, "
            f"{WEIGHTS_KEY!r} carries the whole objective"
        )


def _resolve_scale(context: dict, config: dict) -> int:
    scale = resolve_global_loss_scale(context, config)
    return _resolve_dp_size(scale.get("dp_size"), scale.get("batch_num_tokens"))


def _validate_context(context: dict, reference: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(active, weights)``: where ``loss_mask`` is positive, and the weights on ``reference``'s device."""
    loss_mask = context.get("loss_mask")
    if loss_mask is None:
        raise ValueError(f"{_NAME} requires context['loss_mask']")
    active = canonicalize_loss_mask(loss_mask, reference, objective=_NAME, binary=False) > 0

    weights = context.get(WEIGHTS_KEY)
    if weights is None:
        raise ValueError(f"{_NAME} requires context[{WEIGHTS_KEY!r}], one float32 weight per position of loss_mask")
    if not torch.is_tensor(weights) or weights.dtype != torch.float32:
        received = f"dtype {weights.dtype}" if torch.is_tensor(weights) else type(weights).__name__
        raise ValueError(f"{_NAME} {WEIGHTS_KEY} must be a float32 tensor, got {received}")
    if tuple(weights.shape) != tuple(active.shape):
        raise ValueError(
            f"{_NAME} {WEIGHTS_KEY} must match loss_mask, got {WEIGHTS_KEY}={tuple(weights.shape)} "
            f"loss_mask={tuple(active.shape)}"
        )
    weights = weights.to(active.device)
    if not torch.isfinite(weights).all().item():
        raise ValueError(f"{_NAME} {WEIGHTS_KEY} values must be finite")
    outside = (weights != 0) & ~active
    if outside.any().item():
        first_index = tuple(int(i) for i in outside.nonzero(as_tuple=False)[0].tolist())
        raise ValueError(
            f"{_NAME} {WEIGHTS_KEY} must be zero where loss_mask is 0; found {int(outside.sum().item())} "
            f"nonzero weight(s) outside loss_mask, first at index {first_index}"
        )

    _validate_neutral_temperature(context, _NAME)
    labels = context.get("labels")
    if torch.is_tensor(labels):
        if tuple(labels.shape) != tuple(weights.shape):
            raise ValueError(f"{_NAME} labels must match loss_mask when labels are present")
        if ((labels.to(weights.device) == -100) & (weights != 0)).any().item():
            raise ValueError(f"{_NAME} {WEIGHTS_KEY} must be zero where labels use ignore_index=-100")
    return active, weights


def _weighted_cross_entropy_batching_callback(request: dict) -> None:
    """Reject the whole request before it is sharded, so no rank starts a step its peers refuse."""
    processing = request.get("processing")
    config = processing.get("config") if isinstance(processing, dict) else None
    if config is None:
        config = {}
    if not isinstance(config, dict):
        raise ValueError(f"{_NAME} processing.config must be an object")
    _validate_config(config)
    for context in _request_grpo_contexts(request):
        reference = context.get("input_ids")
        if not torch.is_tensor(reference):
            reference = context.get("loss_mask")
        _validate_context(context, reference)


def _weighted_cross_entropy_packed_reduction(
    microbatches: Sequence[dict],
    config: dict,
    loss_fn_name: str,
) -> PackedLossReduction:
    del loss_fn_name
    _validate_config(config)
    _resolve_scale(microbatches[0] if microbatches else {}, config)
    weighted_token_counts = []
    for index, microbatch in enumerate(microbatches):
        reference = microbatch.get("input_ids")
        if not torch.is_tensor(reference):
            raise ValueError(f"{_NAME} packed microbatch {index} requires tensor input_ids for preflight validation")
        _, weights = _validate_context(microbatch, reference)
        weighted_token_counts.append(float((weights != 0).sum().item()))
    return additive_packed_loss_reduction(weighted_token_counts)


@register_loss_fn(
    _NAME,
    batching_callback=_weighted_cross_entropy_batching_callback,
    packed_loss_reduction=_weighted_cross_entropy_packed_reduction,
    summed_metrics=(OBJECTIVE_SUM_METRIC, WEIGHTED_TOKEN_COUNT_METRIC),
)
@declare_loss_capabilities(REQUIRES_TOKEN_LOGPROBS)
def weighted_cross_entropy_loss(
    model_outputs: dict,
    batch: dict,
    meta: dict,
    config: dict,
    device: str,
) -> tuple[torch.Tensor, dict]:
    """Return ``dp_size * -sum(w * logprobs)`` over this rank's tokens.

    DeepSpeed sums gradients over sequence-parallel ranks and averages them over data-parallel ranks, so the
    ``dp_size`` factor makes the model gradient that of the request-wide ``L_surrogate``, and contributions from
    packed microbatches add. ``OBJECTIVE_SUM_METRIC`` is this rank's share of ``L_surrogate``.
    """
    del device
    _validate_config(config)
    context = {**meta, **batch}
    dp_size = _resolve_scale(context, config)

    logprobs = model_outputs.get("logprobs")
    if logprobs is None:
        raise ValueError(f"{_NAME} requires model_outputs['logprobs']; configure processing.post=['compute_logprobs']")
    if context.get("cu_seqlens") is not None:
        logprobs = _packed_singleton_to_1d(logprobs)
        context = {
            **context,
            **{key: _packed_singleton_to_1d(context.get(key)) for key in ("loss_mask", "labels", WEIGHTS_KEY)},
        }

    active, weights = _validate_context(context, logprobs)
    invalid = active & ~torch.isfinite(logprobs)
    if invalid.any().item():
        first_index = tuple(int(i) for i in invalid.nonzero(as_tuple=False)[0].tolist())
        raise ValueError(
            f"{_NAME} found {int(invalid.sum().item())} non-finite logprob value(s) where loss_mask is positive "
            f"(shape={tuple(logprobs.shape)}); first invalid index={first_index}, "
            f"logprob={logprobs[first_index].detach().item()!r}. Check model logits and attention masks, and "
            "ensure padding or sequence-parallel tail positions have loss_mask=0."
        )

    safe_logprobs = logprobs.float().masked_fill(~active, 0.0)
    objective = -(weights * safe_logprobs).sum()
    metrics = {
        OBJECTIVE_SUM_METRIC: float(objective.detach().item()),
        WEIGHTED_TOKEN_COUNT_METRIC: float((weights != 0).sum().item()),
    }
    return objective * dp_size, metrics
