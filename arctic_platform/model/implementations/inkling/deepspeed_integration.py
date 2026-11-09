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

"""Use the shared AP MoE lifecycle with native Inkling text outputs."""

from __future__ import annotations

from dataclasses import replace

import torch
from transformers import PretrainedConfig
from transformers.models.inkling.configuration_inkling import InklingTextConfig

from arctic_platform.model.implementations.inkling.modeling_inkling import InklingForCausalLMPrimeRL
from arctic_platform.model.implementations.inkling.sequence_parallel import apply_sequence_parallelism
from arctic_platform.model.implementations.inkling.weights import build_iter_vllm_weights
from arctic_platform.model.implementations.inkling.weights import load_weights
from arctic_platform.model.implementations.moe.deepspeed_integration import load_moe_model
from arctic_platform.model.implementations.moe.deepspeed_integration import load_moe_model_for_deepspeed
from arctic_platform.model.implementations.qwen35.deepspeed_integration import _adapter
from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions


def get_model(config, device=torch.device("cpu"), dtype=torch.bfloat16):
    raw, _ = PretrainedConfig.get_config_dict(config.name)
    text = dict(raw.get("text_config", raw))
    if "dense_intermediate_size" in text:
        text["moe_intermediate_size"] = text["intermediate_size"]
        text["intermediate_size"] = text.pop("dense_intermediate_size")
    text["use_cache"] = False
    text["use_grouped_mm"] = config.moe_use_grouped_mm and dtype == torch.bfloat16
    text["tie_word_embeddings"] = False
    text["_attn_implementation"] = "eager"
    text["inkling_mlp_token_chunk_size"] = getattr(config, "tiled_mlp_token_chunk_size", None)
    text["_name_or_path"] = config.name
    hf_config = InklingTextConfig(**text)
    if hf_config.n_routed_experts % config.ep:
        raise ValueError("Inkling expert count must be divisible by expert_parallel")
    with device:
        model = InklingForCausalLMPrimeRL.from_config(hf_config, dtype=dtype, attn_implementation="eager")
    if getattr(config, "attn", "eager") == "flex_attention":
        from arctic_platform.model.implementations.inkling.attention import enable_flex_attention

        enable_flex_attention(model, query_chunk_size=getattr(config, "attention_query_chunk_size", 2048))
        model.config._attn_implementation = "flex_attention"
    return model


def _configure_lm_head(model, *, chunk_size, fused_cross_entropy, fp32_lm_head):
    model._inkling_lm_head_token_chunk_size = chunk_size
    model._inkling_fp32_lm_head = fp32_lm_head
    model._dss_chunked_lm_head_logprobs = chunk_size is not None
    if fp32_lm_head and chunk_size is None:
        from arctic_platform.model.implementations.gpu.lm_head import enable_fp32_lm_head

        enable_fp32_lm_head(model)


def _adapter_for_inkling(options=None):
    def configured_model(config, **kwargs):
        if options is not None:
            config.attention_query_chunk_size = options.attention_query_chunk_size
            config.tiled_mlp_token_chunk_size = options.tiled_mlp_token_chunk_size
        return get_model(config, **kwargs)

    return replace(
        _adapter(),
        get_model=configured_model,
        inject_lm_head=_configure_lm_head,
        apply_sequence_parallelism=apply_sequence_parallelism,
        load_dcp_from_hf=load_weights,
        vllm_weight_export=build_iter_vllm_weights,
    )


def load_inkling_model(**kwargs):
    options = kwargs["options"]
    values = options.model_dump(exclude={"attention_query_chunk_size", "tiled_mlp_token_chunk_size"})
    kwargs["options"] = Qwen3_5MoeOptions.model_validate({**values, "fused_cross_entropy": False})
    adapter = _adapter_for_inkling(options)

    def family_loader(*args, **options):
        return load_moe_model_for_deepspeed(adapter, *args, **options)

    return load_moe_model(adapter, family_loader, **kwargs)
