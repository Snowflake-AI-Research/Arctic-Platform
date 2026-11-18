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

"""ToyHybridMoE: an example model plugin that exercises every part of the model plugin contract."""

from pathlib import Path

from arctic_platform.model_plugin import ModelPlugin
from arctic_platform.model_plugin import load_manifest

from .config import GATED_RECURRENCE
from .config import ToyHybridMoEConfig
from .modeling import ToyHybridMoEForCausalLM
from .parallel import GatedRecurrenceMixer
from .parallel import ToyExpertLayer
from .parallel import parallelism
from .reference import reference_forward
from .weights import convert_layer_to_hf
from .weights import convert_layer_to_prime


def _register() -> None:
    from transformers import AutoConfig

    AutoConfig.register(ToyHybridMoEConfig.model_type, ToyHybridMoEConfig, exist_ok=True)


PLUGIN = ModelPlugin(
    manifest=load_manifest(Path(__file__).with_name("manifest.toml")),
    config_class=ToyHybridMoEConfig,
    build_model=ToyHybridMoEForCausalLM,
    parallelism=parallelism,
    mixers={GATED_RECURRENCE: GatedRecurrenceMixer()},
    expert_layer=ToyExpertLayer(),
    convert_layer_to_hf=convert_layer_to_hf,
    convert_layer_to_prime=convert_layer_to_prime,
    reference_forward=reference_forward,
    register=_register,
)
