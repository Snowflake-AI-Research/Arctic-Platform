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
    text["_name_or_path"] = config.name
    hf_config = InklingTextConfig(**text)
    if hf_config.n_routed_experts % config.ep:
        raise ValueError("Inkling expert count must be divisible by expert_parallel")
    with device:
        return InklingForCausalLMPrimeRL.from_config(hf_config, dtype=dtype, attn_implementation="eager")


def _keep_native_lm_head(model, **kwargs):
    pass


def _adapter_for_inkling():
    return replace(
        _adapter(),
        get_model=get_model,
        inject_lm_head=_keep_native_lm_head,
        load_dcp_from_hf=load_weights,
        vllm_weight_export=build_iter_vllm_weights,
    )


def load_inkling_model(**kwargs):
    kwargs["options"] = Qwen3_5MoeOptions.model_validate(
        {**kwargs["options"].model_dump(), "fused_cross_entropy": False, "fused_lm_head_token_chunk_size": "disabled"}
    )
    adapter = _adapter_for_inkling()

    def family_loader(*args, **options):
        return load_moe_model_for_deepspeed(adapter, *args, **options)

    return load_moe_model(adapter, family_loader, **kwargs)
