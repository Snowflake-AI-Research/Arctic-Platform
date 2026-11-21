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

"""OLMoE as a model plugin: the dense OLMo-2 example plus mixture of experts.

On top of the dense path it shows routing through ``route`` (router replay), ``experts_core`` and a
``StackedExpertLayer`` (expert parallelism), and converters between stacked trainer experts and per-expert
checkpoint tensors. The sampler is still vLLM's built-in model, whose fused MoE layer records routed experts for
router replay.
"""

from pathlib import Path

from transformers import OlmoeConfig

from arctic_platform.model_plugin import ModelPlugin
from arctic_platform.model_plugin import ParallelismInfo
from arctic_platform.model_plugin import StackedExpertLayer
from arctic_platform.model_plugin import load_manifest

from .modeling import FULL_ATTENTION
from .modeling import OlmoeForCausalLM
from .modeling import OlmoeSparseMoeBlock
from .reference import reference_forward
from .weights import convert_layer_to_hf
from .weights import convert_layer_to_prime


def parallelism(config: OlmoeConfig) -> ParallelismInfo:
    """Every decoder layer is attention with an expert block."""
    return ParallelismInfo(
        layer_types=(FULL_ATTENTION,) * config.num_hidden_layers,
        heads={FULL_ATTENTION: (config.num_attention_heads, config.num_key_value_heads)},
        moe_layers=tuple(range(config.num_hidden_layers)),
        num_experts=config.num_experts,
        top_k=config.num_experts_per_tok,
    )


def validate(request, config: OlmoeConfig, options: None) -> list[str]:
    """Config features this trainer model does not implement, refused at admission instead of trained wrong."""
    reasons = []
    if config.clip_qkv is not None:
        reasons.append("clip_qkv is not supported")
    if config.tie_word_embeddings:
        reasons.append("tied input and output embeddings are not supported")
    rope_type = config.rope_parameters.get("rope_type", "default")
    if rope_type != "default":
        reasons.append(f"rope_type {rope_type!r} is not supported, only default rotary embedding")
    if config.attention_bias:
        reasons.append("attention biases are not supported")
    return reasons


PLUGIN = ModelPlugin(
    manifest=load_manifest(Path(__file__).with_name("manifest.toml")),
    config_class=OlmoeConfig,
    build_model=OlmoeForCausalLM,
    parallelism=parallelism,
    expert_layer=StackedExpertLayer(OlmoeSparseMoeBlock, ("experts.w1", "experts.w2", "experts.w3")),
    convert_layer_to_hf=convert_layer_to_hf,
    convert_layer_to_prime=convert_layer_to_prime,
    reference_forward=reference_forward,
    validate=validate,
)
