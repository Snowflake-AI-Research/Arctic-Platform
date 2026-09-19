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

Inner PPO/ECHO math is shared with ``ap_grpo``. The contracts differ: AP fills
missing trio keys from meta/batch with config winning; Cortex
``resolve_global_loss_scale`` lets context win and raises on a conflict.
"""

from __future__ import annotations

from typing import Tuple

import torch

from arctic_platform.common.registry import register_loss_fn
from arctic_platform.rl.processors.functional import resolve_global_loss_scale
from arctic_platform.rl.processors.grpo import _ECHO_CONFIG_KEYS
from arctic_platform.rl.processors.grpo import _ECHO_REQUIRED_CONFIG_KEYS
from arctic_platform.rl.processors.grpo import _GRPO_CONFIG_KEYS
from arctic_platform.rl.processors.grpo import _grpo_context
from arctic_platform.rl.processors.grpo import _grpo_loss
from arctic_platform.rl.processors.grpo import _grpo_packed_loss_reduction


def _cortex_distributed_config(config: dict, batch: dict, meta: dict) -> tuple[dict, dict]:
    context = _grpo_context(batch, meta)
    scale = resolve_global_loss_scale(context, config)
    cfg = dict(config)
    cfg.update(scale)
    has_global_denom = cfg.get("batch_num_tokens") is not None or cfg.get("global_batch_size") is not None
    if not has_global_denom and config.get("dp_size") is None:
        cfg.pop("dp_size", None)
    return cfg, context


@register_loss_fn("grpo", packed_loss_reduction=_grpo_packed_loss_reduction)
def cortex_grpo_loss(
    model_outputs: dict,
    batch: dict,
    meta: dict,
    config: dict,
    device: str,
) -> Tuple[torch.Tensor, dict]:
    """Cortex ``grpo`` contract on the AP 5-arg ABI."""
    config, context = _cortex_distributed_config(config, batch, meta)
    echo_keys = _ECHO_CONFIG_KEYS & set(config)
    if echo_keys:
        raise ValueError(
            f"loss_fn 'grpo' does not accept ECHO config keys {sorted(echo_keys)} — request 'grpo_echo_v1'"
        )
    return _grpo_loss(model_outputs, context, config, device)


@register_loss_fn("grpo_echo_v1", packed_loss_reduction=_grpo_packed_loss_reduction)
def cortex_grpo_echo_v1_loss(
    model_outputs: dict,
    batch: dict,
    meta: dict,
    config: dict,
    device: str,
) -> Tuple[torch.Tensor, dict]:
    """Cortex ``grpo_echo_v1`` contract on the AP 5-arg ABI."""
    config, context = _cortex_distributed_config(config, batch, meta)
    unknown_keys = set(config) - _GRPO_CONFIG_KEYS - _ECHO_CONFIG_KEYS
    if unknown_keys:
        raise ValueError(f"Unknown config keys for loss_fn 'grpo_echo_v1': {sorted(unknown_keys)}")
    missing_keys = {key for key in _ECHO_REQUIRED_CONFIG_KEYS if config.get(key) is None}
    if missing_keys:
        raise ValueError(f"loss_fn 'grpo_echo_v1' requires non-None config keys {sorted(missing_keys)}")
    return _grpo_loss(model_outputs, context, config, device)
