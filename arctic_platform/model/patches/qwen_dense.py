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

"""Qwen dense setup applied before PEFT and DeepSpeed wrapping."""

from __future__ import annotations

import torch.nn as nn

from arctic_platform.model.config import ActivationOffloadConfig
from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.patch import register_patch


def _text_config(model: nn.Module):
    config = getattr(model, "config", None)
    get_text_config = getattr(config, "get_text_config", None)
    return get_text_config() if callable(get_text_config) else config


def _transformer_layers(model: nn.Module):
    target = model
    for part in ("model", "layers"):
        target = getattr(target, part, None)
        if target is None:
            raise ValueError("Qwen dense setup requires model.model.layers")
    return target


def _apply_activation_checkpointing(model: nn.Module, value: bool | int) -> None:
    if value is False:
        return

    config = _text_config(model)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = False

    frequency = 1 if value is True else int(value)
    if frequency <= 1:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

        layers = _transformer_layers(model)
        for layer_index, layer_name in enumerate(list(layers._modules)):
            if layer_index % frequency == 0:
                layers._modules[layer_name] = checkpoint_wrapper(
                    layers._modules[layer_name],
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                    preserve_rng_state=False,
                )

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()


def _apply_activation_offload(model: nn.Module, config: ActivationOffloadConfig | None) -> None:
    if config is None:
        return

    from arctic_platform.model.implementations.gpu.activation_offload import install_activation_offload

    manager = install_activation_offload(model, config=config)
    backbone = getattr(model, "base_model", None)
    if backbone is not None and backbone is not model:
        install_activation_offload(
            backbone,
            config=config,
            manager=manager,
        )


@register_patch("qwen_dense")
def apply_qwen_dense(model: nn.Module, ctx: LoaderContext) -> None:
    settings = ctx.spec.patches.qwen_dense
    assert settings is not None
    config = _text_config(model)
    model_type = getattr(config, "model_type", "")
    if model_type != "qwen3":
        raise ValueError(f"qwen_dense patch requires model_type='qwen3', got {model_type!r}")

    if ctx.spec.parallelism.sequence_parallel > 1 and hasattr(config, "use_cache"):
        config.use_cache = False
    _apply_activation_checkpointing(model, settings.activation_checkpointing)
    _apply_activation_offload(model, settings.activation_offload)
    if settings.compile is not None:
        for layer in _transformer_layers(model):
            layer.compile(fullgraph=settings.compile.fullgraph)
    if settings.tiled_mlp_token_chunk_size is not None:
        from arctic_platform.model.patches._tiled_mlp import apply_dense_tiled_mlp

        apply_dense_tiled_mlp(
            model,
            token_chunk_size=settings.tiled_mlp_token_chunk_size,
        )
    if settings.fp32_lm_head:
        from arctic_platform.model.implementations.gpu.lm_head import enable_fp32_lm_head

        enable_fp32_lm_head(model)
    if settings.fused_lm_head_token_chunk_size is not None:
        from arctic_platform.model.implementations.gpu.lm_head import enable_chunked_lm_head_logprobs

        enable_chunked_lm_head_logprobs(
            model,
            token_chunk_size=settings.fused_lm_head_token_chunk_size,
            vocab_chunk_size=settings.fused_lm_head_vocab_chunk_size,
            fp32_lm_head=settings.fp32_lm_head,
        )
