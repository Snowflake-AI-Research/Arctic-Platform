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

"""gpt-oss as a model plugin: mixture of experts with its own attention core, on vLLM's built-in model.

Next to the OLMoE example it shows a plugin layer type for an attention variant the platform's core does not cover
(learned sinks), two layer types sharing one core (sliding and full attention), and stacked experts with biases kept
in the checkpoint's own layout. Released gpt-oss checkpoints store experts in MXFP4; this plugin loads bf16
checkpoints, so dequantize the released weights first.
"""

from pathlib import Path

from transformers import GptOssConfig

from arctic_platform.model_plugin import HeadParallelMixer
from arctic_platform.model_plugin import ModelPlugin
from arctic_platform.model_plugin import ParallelismInfo
from arctic_platform.model_plugin import StackedExpertLayer
from arctic_platform.model_plugin import load_manifest

from .modeling import FULL_ATTENTION
from .modeling import SLIDING_ATTENTION
from .modeling import SUPPORTED_ROPE_TYPES
from .modeling import GptOssAttention
from .modeling import GptOssForCausalLM
from .modeling import GptOssMoE
from .reference import reference_forward

EXPERT_TENSORS = ("experts.gate_up_proj", "experts.gate_up_proj_bias", "experts.down_proj", "experts.down_proj_bias")


def parallelism(config: GptOssConfig) -> ParallelismInfo:
    """Sliding and full attention layers from ``layer_types``, every one with an expert block."""
    heads = (config.num_attention_heads, config.num_key_value_heads)
    return ParallelismInfo(
        layer_types=tuple(config.layer_types),
        heads={SLIDING_ATTENTION: heads, FULL_ATTENTION: heads},
        moe_layers=tuple(range(config.num_hidden_layers)),
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
    )


def validate(request, config: GptOssConfig, options: None) -> list[str]:
    """Config features this trainer model does not implement, refused at admission instead of trained wrong."""
    reasons = []
    rope_type = config.rope_parameters.get("rope_type", "default")
    if rope_type not in SUPPORTED_ROPE_TYPES:
        reasons.append(f"rope_type {rope_type!r} is not supported; supported: {list(SUPPORTED_ROPE_TYPES)}")
    if config.tie_word_embeddings:
        reasons.append("tied input and output embeddings are not supported")
    return reasons


PLUGIN = ModelPlugin(
    manifest=load_manifest(Path(__file__).with_name("manifest.toml")),
    config_class=GptOssConfig,
    build_model=GptOssForCausalLM,
    parallelism=parallelism,
    mixers={
        SLIDING_ATTENTION: HeadParallelMixer(SLIDING_ATTENTION, GptOssAttention),
        FULL_ATTENTION: HeadParallelMixer(FULL_ATTENTION, GptOssAttention),
    },
    expert_layer=StackedExpertLayer(GptOssMoE, EXPERT_TENSORS),
    reference_forward=reference_forward,
    validate=validate,
)
