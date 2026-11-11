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

"""Use the shared AP MoE lifecycle with the Inkling2 text model."""

from __future__ import annotations

from dataclasses import replace

import torch
from transformers import PretrainedConfig

from arctic_platform.model.implementations.inkling2.configuration_inkling2 import Inkling2TextConfig
from arctic_platform.model.implementations.inkling2.modeling_inkling2 import Inkling2ForCausalLMPrimeRL
from arctic_platform.model.implementations.inkling2.weights import load_weights
from arctic_platform.model.implementations.moe.deepspeed_integration import load_moe_model
from arctic_platform.model.implementations.moe.deepspeed_integration import load_moe_model_for_deepspeed
from arctic_platform.model.implementations.qwen35.deepspeed_integration import _adapter
from arctic_platform.model.loaders.qwen3_5_moe import Qwen3_5MoeOptions


def get_model(config, device=torch.device("cpu"), dtype=torch.bfloat16):
    raw, _ = PretrainedConfig.get_config_dict(config.name)
    text = dict(raw.get("text_config", raw))
    text.pop("model_type", None)
    text["use_grouped_mm"] = config.moe_use_grouped_mm and dtype == torch.bfloat16
    text["inkling2_attention_query_chunk_size"] = getattr(config, "attention_query_chunk_size", 2048)
    text["_name_or_path"] = config.name
    hf_config = Inkling2TextConfig(**text)
    if hf_config.n_routed_experts % config.ep:
        raise ValueError("Inkling2 expert count must be divisible by expert_parallel")
    with device:
        return Inkling2ForCausalLMPrimeRL.from_config(hf_config, dtype=dtype)


def _configure_lm_head(model, *, chunk_size, fused_cross_entropy, fp32_lm_head):
    if fused_cross_entropy:
        raise ValueError("Inkling2 training uses its native LM head")
    model._inkling2_lm_head_token_chunk_size = chunk_size
    model._inkling2_fp32_lm_head = fp32_lm_head
    model._dss_chunked_lm_head_logprobs = chunk_size is not None


def _apply_sequence_parallelism(model, sp_size, sp_group):
    if sp_size > 1:
        raise ValueError("Inkling2 training does not support sequence parallelism yet")


def _adapter_for_inkling2(options=None):
    def configured_model(config, **kwargs):
        if options is not None:
            config.attention_query_chunk_size = options.attention_query_chunk_size
        return get_model(config, **kwargs)

    return replace(
        _adapter(),
        get_model=configured_model,
        inject_lm_head=_configure_lm_head,
        apply_sequence_parallelism=_apply_sequence_parallelism,
        load_dcp_from_hf=load_weights,
        vllm_weight_export=None,
    )


def load_inkling2_model(**kwargs):
    options = kwargs["options"]
    values = options.model_dump(exclude={"attention_query_chunk_size"})
    kwargs["options"] = Qwen3_5MoeOptions.model_validate({**values, "fused_cross_entropy": False})
    adapter = _adapter_for_inkling2(options)

    def family_loader(*args, **options):
        return load_moe_model_for_deepspeed(adapter, *args, **options)

    return load_moe_model(adapter, family_loader, **kwargs)
