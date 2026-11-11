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
"""Gradient-checkpointing patch applied before the DeepSpeed wrap."""

from __future__ import annotations

import torch.nn as nn

from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.patch import register_patch
from arctic_platform.model.patches.utils import transformer_layers


def _text_config(model: nn.Module):
    config = getattr(model, "config", None)
    get_text_config = getattr(config, "get_text_config", None)
    return get_text_config() if callable(get_text_config) else config


def _disable_kv_cache(config) -> None:
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False


@register_patch("gradient_checkpointing")
def apply_gradient_checkpointing(model: nn.Module, ctx: LoaderContext) -> None:
    value = ctx.spec.patches.gradient_checkpointing
    root_config = getattr(model, "config", None)
    config = _text_config(model)
    is_qwen3 = getattr(config, "model_type", "") == "qwen3"
    _disable_kv_cache(root_config)
    if config is not root_config:
        _disable_kv_cache(config)

    if value is True or value == 1:
        if is_qwen3:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:
            model.gradient_checkpointing_enable()
    else:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

        layers = transformer_layers(model, patch_name="periodic gradient checkpointing")
        num_layers = len(layers._modules)
        if isinstance(value, list):
            missing = sorted(index for index in value if index >= num_layers)
            if missing:
                raise ValueError(f"gradient_checkpointing layers {missing} are past the model's {num_layers} layers")
            checkpointed = set(value)
        else:
            checkpointed = set(range(0, num_layers, int(value)))
        for layer_index, layer_name in enumerate(list(layers._modules)):
            if layer_index in checkpointed:
                layers._modules[layer_name] = checkpoint_wrapper(
                    layers._modules[layer_name],
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                    preserve_rng_state=False,
                )

    if is_qwen3 and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
