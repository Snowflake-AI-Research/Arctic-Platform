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

"""LFM2 as a model plugin: a new layer type, on vLLM's built-in model.

It adds one thing to the dense OLMo-2 example: a sequence-mixing layer the platform has never seen, the gated short
convolution, with its core ``mix`` and the platform's ``HeadParallelMixer`` as its sequence-parallel policy. The
sampler is still vLLM's built-in ``Lfm2ForCausalLM``, so the plugin has no vLLM code.
"""

from pathlib import Path

from transformers import Lfm2Config

from arctic_platform.model_plugin import HeadParallelMixer
from arctic_platform.model_plugin import ModelPlugin
from arctic_platform.model_plugin import ParallelismInfo
from arctic_platform.model_plugin import load_manifest

from .modeling import CONV
from .modeling import FULL_ATTENTION
from .modeling import Lfm2ForCausalLM
from .modeling import Lfm2ShortConv
from .reference import reference_forward


def parallelism(config: Lfm2Config) -> ParallelismInfo:
    """Layer kinds from ``layer_types``. Under sequence parallelism, attention head counts and the convolution's
    channels (its "heads") must divide by ``sp_size``."""
    return ParallelismInfo(
        layer_types=tuple(config.layer_types),
        heads={
            FULL_ATTENTION: (config.num_attention_heads, config.num_key_value_heads),
            CONV: (config.hidden_size,),
        },
    )


def validate(request, config: Lfm2Config, options: None) -> list[str]:
    """Config features this trainer model does not implement, refused at admission instead of trained wrong."""
    reasons = []
    if config.conv_bias:
        reasons.append("conv_bias is not supported")
    for name in ("conv_dim", "block_dim"):  # released checkpoints carry them; this model uses hidden_size for both
        value = getattr(config, name, config.hidden_size)
        if value != config.hidden_size:
            reasons.append(f"{name} {value} differs from hidden_size {config.hidden_size}")
    rope_type = config.rope_parameters.get("rope_type", "default")
    if rope_type != "default":
        reasons.append(f"rope_type {rope_type!r} is not supported, only default rotary embedding")
    return reasons


PLUGIN = ModelPlugin(
    manifest=load_manifest(Path(__file__).with_name("manifest.toml")),
    config_class=Lfm2Config,
    build_model=Lfm2ForCausalLM,
    parallelism=parallelism,
    mixers={CONV: HeadParallelMixer(CONV, Lfm2ShortConv)},
    reference_forward=reference_forward,
    validate=validate,
)
