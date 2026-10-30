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

"""DeepSpeed entry point for DeepSeek-V4-Flash.

The decoder is a causal LM on the shared MoE lifecycle. Sequence parallel is
the model's own context parallel: each rank keeps a contiguous query shard and
the sparse attention all-gathers keys. The generic MoE sequence-parallel wrapper
rewrites flash-attention modules, so it stays off and the DSS SP group is
installed on this model directly.
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh

from arctic_platform.model.implementations.qwen35 import deepspeed_integration as qwen_ds
from arctic_platform.model.implementations.qwen35.config import ModelConfig
from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions


def _build_model_config(*args, dequantize_checkpoint: bool, **kwargs) -> ModelConfig:
    return replace(qwen_ds._build_model_config(*args, **kwargs), dequantize_checkpoint=dequantize_checkpoint)


def _install_context_parallel(model: nn.Module, sp_group, sp_size: int) -> None:
    from arctic_platform.model.implementations.deepseek_v4.attention import DeepseekV4Attention
    from arctic_platform.model.implementations.deepseek_v4.cp import CPContext
    from arctic_platform.model.implementations.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Model
    from arctic_platform.model.implementations.deepseek_v4.modeling_deepseek_v4 import _seq_lens_from_position_ids

    if sp_group is None:
        raise ValueError("DeepSeek V4 sequence parallel requires the DSS SP group")
    world = dist.get_world_size(sp_group)
    if world != sp_size:
        raise ValueError(f"SP group world {world} does not match sp_size {sp_size}")
    context = CPContext(cp_group=sp_group, cp_rank=dist.get_rank(sp_group), cp_world_size=world)

    backbone = None
    attentions = []
    for module in model.modules():
        if isinstance(module, DeepseekV4Model):
            backbone = module
        elif isinstance(module, DeepseekV4Attention):
            attentions.append(module)
    if backbone is None or not attentions:
        raise RuntimeError("DeepSeek V4 sequence parallel found no backbone or attention modules")
    backbone.cp_context = context
    for attention in attentions:
        attention.cp_context = context

    original = backbone.forward

    def forward(
        input_ids=None,
        position_ids=None,
        inputs_embeds=None,
        routed_experts=None,
        *,
        seq_lens=None,
        seq_lens_are_pre_shard=False,
        **kwargs,
    ):
        # DSS has already sliced the row. Document lengths have to describe the
        # whole row, which the local position shard cannot.
        if seq_lens is None:
            if position_ids is None:
                raise ValueError("DeepSeek V4 sequence parallel needs position_ids to recover document lengths")
            local = position_ids.contiguous()
            gathered = [torch.empty_like(local) for _ in range(world)]
            dist.all_gather(gathered, local, group=sp_group)
            seq_lens = _seq_lens_from_position_ids(torch.cat(gathered, dim=1))
            seq_lens_are_pre_shard = True
        return original(
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            routed_experts=routed_experts,
            seq_lens=seq_lens,
            seq_lens_are_pre_shard=seq_lens_are_pre_shard,
            **kwargs,
        )

    setattr(backbone, "forward", forward)


def load_deepseek_v4_model(
    *,
    model_name: str,
    optimization_dtype: str,
    attn_implementation: str,
    ep_size: int,
    sp_size: int = 1,
    sp_group=None,
    ep_group=None,
    options: Qwen3_5MoeOptions,
    dequantize_checkpoint: bool = False,
) -> nn.Module:
    """Load DeepSeek-V4-Flash; ``dequantize_checkpoint`` trains the official FP8/MXFP4 weights in bf16."""
    adapter = replace(
        qwen_ds._generic_adapter(),
        build_model_config=partial(_build_model_config, dequantize_checkpoint=dequantize_checkpoint),
    )
    model = qwen_ds._load_moe_model(
        adapter,
        qwen_ds.load_generic_moe_model_for_deepspeed,
        model_name=model_name,
        optimization_dtype=optimization_dtype,
        attn_implementation=attn_implementation,
        ep_size=ep_size,
        sp_size=1,
        sp_group=None,
        ep_group=ep_group,
        options=options,
        patch_moe_detection=qwen_ds.patch_deepspeed_moe_detection,
        device_mesh_type=DeviceMesh,
    )
    qwen_ds.maybe_apply_row_invariant_projections(model, options.model_dump())
    if sp_size > 1:
        _install_context_parallel(model, sp_group, sp_size)
    return model
