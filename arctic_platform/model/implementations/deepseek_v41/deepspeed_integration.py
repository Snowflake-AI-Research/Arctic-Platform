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

"""DeepSpeed entry point for DeepSeek-V4.1 text.

Sequence parallel is the model's context parallel. The generic MoE wrapper rewrites
flash-attention modules, so it stays off and the DSS SP group is installed here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn

from arctic_platform.model.implementations.qwen35.deepspeed_integration import load_generic_moe_model
from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions

logger = logging.getLogger(__name__)


def _install_context_parallel(model: nn.Module, sp_group, sp_size: int) -> None:
    from arctic_platform.model.implementations.deepseek_v4.cp import CPContext
    from arctic_platform.model.implementations.deepseek_v41.attention import DeepseekV41Attention
    from arctic_platform.model.implementations.deepseek_v41.modeling_deepseek_v41 import DeepseekV41Model
    from arctic_platform.model.implementations.deepseek_v41.modeling_deepseek_v41 import _seq_lens_from_position_ids

    if sp_group is None:
        raise ValueError("DeepSeek V4.1 sequence parallel requires the DSS SP group")
    world = dist.get_world_size(sp_group)
    if world != sp_size:
        raise ValueError(f"SP group world {world} does not match sp_size {sp_size}")
    context = CPContext(cp_group=sp_group, cp_rank=dist.get_rank(sp_group), cp_world_size=world)

    backbone = None
    attentions = []
    for module in model.modules():
        if isinstance(module, DeepseekV41Model):
            backbone = module
        elif isinstance(module, DeepseekV41Attention):
            attentions.append(module)
    if backbone is None or not attentions:
        raise RuntimeError("DeepSeek V4.1 sequence parallel found no backbone or attention modules")
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
        compressed_token_map=None,
        compressed_token_ids=None,
        compressed_pad_id=None,
        **kwargs,
    ):
        if seq_lens is None:
            if position_ids is None:
                raise ValueError("DeepSeek V4.1 sequence parallel needs position_ids to recover document lengths")
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
            compressed_token_map=compressed_token_map,
            compressed_token_ids=compressed_token_ids,
            compressed_pad_id=compressed_pad_id,
            **kwargs,
        )

    setattr(backbone, "forward", forward)


def install_tokenizer_compressed_map(model: nn.Module, model_path: str) -> bool:
    """Build the Engram map from a local tokenizer and keep it on the hasher. No hub download."""
    target = model.model if hasattr(model, "model") else model
    if getattr(target, "engram_hasher", None) is None:
        return False
    path = Path(model_path)
    if not (path / "tokenizer.json").is_file():
        logger.warning(
            "DeepSeek V4.1 Engram has no local tokenizer.json under %s; "
            "pass compressed_token_map or compressed_token_ids until one is installed",
            path,
        )
        return False
    from transformers import AutoTokenizer

    from arctic_platform.model.implementations.deepseek_v41.engram import build_compressed_token_map

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    lookup, size = build_compressed_token_map(tokenizer)
    expected = int(target.config.engram_compressed_vocab_size)
    if size != expected:
        raise ValueError(
            f"tokenizer compressed vocab is {size} but engram_compressed_vocab_size is {expected}. "
            "Hash multipliers are derived from that size, so the tables would be rehashed."
        )
    target.install_compressed_token_map(torch.tensor(lookup, dtype=torch.long))
    return True


def load_deepseek_v41_model(
    *,
    model_name: str,
    optimization_dtype: str,
    attn_implementation: str,
    ep_size: int,
    sp_size: int = 1,
    sp_group=None,
    ep_group=None,
    options: Qwen3_5MoeOptions,
) -> nn.Module:
    model = load_generic_moe_model(
        model_name=model_name,
        optimization_dtype=optimization_dtype,
        attn_implementation=attn_implementation,
        ep_size=ep_size,
        sp_size=1,
        sp_group=None,
        ep_group=ep_group,
        options=options,
    )
    if sp_size > 1:
        _install_context_parallel(model, sp_group, sp_size)
    install_tokenizer_compressed_map(model, model_name)
    return model
