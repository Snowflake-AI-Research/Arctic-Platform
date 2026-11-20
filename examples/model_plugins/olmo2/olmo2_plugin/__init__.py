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

"""OLMo-2 as a model plugin: the minimal dense plugin, and the template to start from.

It shows the shortest path: one platform attention layer type, a model named like its Hugging Face checkpoint (no
converters), Hugging Face's own implementation as the conformance reference, and vLLM's built-in model as the
sampler (no vLLM code). Everything else in the contract is optional; ``examples/model_plugins/toy_hybrid_moe``
shows it.
"""

from pathlib import Path

from transformers import Olmo2Config

from arctic_platform.model_plugin import ModelPlugin
from arctic_platform.model_plugin import ParallelismInfo
from arctic_platform.model_plugin import load_manifest

from .modeling import FULL_ATTENTION
from .modeling import Olmo2ForCausalLM
from .reference import reference_forward


def parallelism(config: Olmo2Config) -> ParallelismInfo:
    """Every decoder layer is attention; under sequence parallelism both head counts must divide by ``sp_size``."""
    return ParallelismInfo(
        layer_types=(FULL_ATTENTION,) * config.num_hidden_layers,
        heads={FULL_ATTENTION: (config.num_attention_heads, config.num_key_value_heads)},
    )


def validate(request, config: Olmo2Config, options: None) -> list[str]:
    """Config features this trainer model does not implement, refused at admission instead of trained wrong."""
    reasons = []
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
    config_class=Olmo2Config,  # transformers already registers it: no `register` needed
    build_model=Olmo2ForCausalLM,
    parallelism=parallelism,
    reference_forward=reference_forward,
    validate=validate,
)
