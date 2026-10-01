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

"""Cortex GRPO entrypoints: context-wins trio, registered as ``grpo`` / ``grpo_echo_v1``.

Inner PPO/ECHO math is shared with ``ap_grpo``. Only the global-loss-scale trio
(``dp_size`` / ``batch_num_tokens`` / ``global_batch_size``) resolves
differently: here ``resolve_global_loss_scale`` lets the context win and raises
on a conflict, matching what the Cortex zone already does, while ``ap_grpo``
fills missing keys from meta/batch and lets the config win.

The precedence is a property of the requested loss-fn NAME, not of the backend:
``grpo`` is the Cortex contract and ``ap_grpo`` is the AP one on every backend.
A client that keeps sending ``grpo`` therefore gets identical scaling whether it
runs against the Cortex zone or this training kernel; only a client that also
renames its loss fn changes contracts. See
``tests/rl/test_phase_a_gates.py::TestTrioPrecedenceIsPerLossName``.
"""

from __future__ import annotations

from typing import Tuple

import torch

from arctic_platform.common.registry import declare_loss_capabilities
from arctic_platform.common.registry import register_loss_fn
from arctic_platform.rl.processors.base_loss import PRESERVES_EXPLICIT_LOSS_SCALE_CONFIG
from arctic_platform.rl.processors.base_loss import REQUIRES_ALIGNED_TOKEN_LOGPROBS
from arctic_platform.rl.processors.functional import resolve_global_loss_scale
from arctic_platform.rl.processors.grpo import _ECHO_CONFIG_KEYS
from arctic_platform.rl.processors.grpo import ECHO_SUMMED_METRICS
from arctic_platform.rl.processors.grpo import _grpo_batching_callback
from arctic_platform.rl.processors.grpo import _grpo_context
from arctic_platform.rl.processors.grpo import _grpo_echo_batching_callback
from arctic_platform.rl.processors.grpo import _grpo_echo_packed_loss_reduction
from arctic_platform.rl.processors.grpo import _grpo_echo_validation_callback
from arctic_platform.rl.processors.grpo import _grpo_loss
from arctic_platform.rl.processors.grpo import _grpo_metrics_callback
from arctic_platform.rl.processors.grpo import _grpo_mixed_batching_callback
from arctic_platform.rl.processors.grpo import _grpo_mixed_packed_loss_reduction
from arctic_platform.rl.processors.grpo import _grpo_mixed_validation_callback
from arctic_platform.rl.processors.grpo import _grpo_model_call_count_callback
from arctic_platform.rl.processors.grpo import _grpo_packed_loss_reduction
from arctic_platform.rl.processors.grpo import _grpo_validation_callback
from arctic_platform.rl.processors.grpo import _validate_echo_config
from arctic_platform.rl.processors.grpo import _validate_mixed_config


def _cortex_distributed_config(config: dict, batch: dict, meta: dict) -> tuple[dict, dict]:
    context = _grpo_context(batch, meta)
    scale = resolve_global_loss_scale(context, config)
    cfg = dict(config)
    cfg.update(scale)
    uses_weighted_prompt_mean = (
        cfg.get("loss_agg_mode") == "prompt-mean" and context.get("sequence_loss_weights") is not None
    )
    has_global_scale = (
        cfg.get("batch_num_tokens") is not None
        or cfg.get("global_batch_size") is not None
        or uses_weighted_prompt_mean
    )
    if not has_global_scale and config.get("dp_size") is None:
        cfg.pop("dp_size", None)
    return cfg, context


@register_loss_fn(
    "grpo",
    batching_callback=_grpo_batching_callback,
    validation_callback=_grpo_validation_callback,
    packed_loss_reduction=_grpo_packed_loss_reduction,
    model_call_count_callback=_grpo_model_call_count_callback,
    metrics_callback=_grpo_metrics_callback,
)
@declare_loss_capabilities(REQUIRES_ALIGNED_TOKEN_LOGPROBS, PRESERVES_EXPLICIT_LOSS_SCALE_CONFIG)
def cortex_grpo_loss(
    model_outputs: dict,
    batch: dict,
    meta: dict,
    config: dict,
    device: str,
) -> Tuple[torch.Tensor, dict]:
    """Cortex ``grpo`` contract on the AP 5-arg ABI."""
    config, context = _cortex_distributed_config(config, batch, meta)
    if "nll_mask" in context:
        raise ValueError("nll_mask requires loss_fn='grpo_mixed_v1'")
    echo_keys = _ECHO_CONFIG_KEYS & set(config)
    if echo_keys:
        raise ValueError(
            f"loss_fn 'grpo' does not accept ECHO config keys {sorted(echo_keys)} — request 'grpo_echo_v1'"
        )
    return _grpo_loss(model_outputs, context, config, device)


@register_loss_fn(
    "grpo_mixed_v1",
    batching_callback=_grpo_mixed_batching_callback,
    validation_callback=_grpo_mixed_validation_callback,
    packed_loss_reduction=_grpo_mixed_packed_loss_reduction,
    model_call_count_callback=_grpo_model_call_count_callback,
    metrics_callback=_grpo_metrics_callback,
)
@declare_loss_capabilities(REQUIRES_ALIGNED_TOKEN_LOGPROBS, PRESERVES_EXPLICIT_LOSS_SCALE_CONFIG)
def cortex_grpo_mixed_v1_loss(
    model_outputs: dict,
    batch: dict,
    meta: dict,
    config: dict,
    device: str,
) -> Tuple[torch.Tensor, dict]:
    """Cortex ``grpo_mixed_v1`` contract on the AP 5-arg ABI."""
    config, context = _cortex_distributed_config(config, batch, meta)
    _validate_mixed_config(context, config)
    return _grpo_loss(model_outputs, context, config, device)


@register_loss_fn(
    "grpo_echo_v1",
    batching_callback=_grpo_echo_batching_callback,
    validation_callback=_grpo_echo_validation_callback,
    packed_loss_reduction=_grpo_echo_packed_loss_reduction,
    model_call_count_callback=_grpo_model_call_count_callback,
    metrics_callback=_grpo_metrics_callback,
    summed_metrics=ECHO_SUMMED_METRICS,
)
@declare_loss_capabilities(REQUIRES_ALIGNED_TOKEN_LOGPROBS, PRESERVES_EXPLICIT_LOSS_SCALE_CONFIG)
def cortex_grpo_echo_v1_loss(
    model_outputs: dict,
    batch: dict,
    meta: dict,
    config: dict,
    device: str,
) -> Tuple[torch.Tensor, dict]:
    """Cortex ``grpo_echo_v1`` contract on the AP 5-arg ABI."""
    config, context = _cortex_distributed_config(config, batch, meta)
    if "nll_mask" in context:
        raise ValueError("nll_mask requires loss_fn='grpo_mixed_v1'")
    _validate_echo_config(config, "grpo_echo_v1")
    return _grpo_loss(model_outputs, context, config, device)
