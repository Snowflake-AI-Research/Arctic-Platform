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
"""Build the ModelSpec used to load an expert-parallel MoE."""

from __future__ import annotations

from typing import Any

from arctic_platform.model.config import ModelSpec
from arctic_platform.model.config import ParallelismConfig
from arctic_platform.model.config import Patches
from arctic_platform.model.implementations.moe.config_validation import effective_fused_cross_entropy
from arctic_platform.model.implementations.moe.config_validation import validate_lm_head_fused_ce_config


def _omit_none(value: Any) -> Any:
    """Drop ``None`` entries so loader-option fields retain their defaults."""
    if isinstance(value, dict):
        return {key: _omit_none(item) for key, item in value.items() if item is not None}
    return value


def moe_model_spec(
    *,
    model_name: str,
    ep_size: int,
    sp_size: int = 1,
    loader_options: dict | None = None,
    tiled_mlp_token_chunk_size: int | None = None,
    peft_config: dict | None = None,
) -> ModelSpec:
    """Translate MoE training options into one ``ModelSpec``.

    ``optimization_dtype`` and ``attn`` are spec fields, not loader options.
    """
    import arctic_platform.model.loaders  # noqa: F401  # registers loader schemas

    config = dict(loader_options or {})
    optimization_dtype = config.pop("optimization_dtype", "bfloat16")
    attn_implementation = config.pop("attn", None)
    if config.get("ac_config") is None:
        config.pop("ac_config", None)
    else:
        config["ac_config"] = _omit_none(config["ac_config"])
    if tiled_mlp_token_chunk_size is not None:
        config["tiled_mlp_token_chunk_size"] = tiled_mlp_token_chunk_size
    config["fused_cross_entropy"] = effective_fused_cross_entropy(config)
    validate_lm_head_fused_ce_config(config)
    return ModelSpec(
        model_path_or_name=model_name,
        dtype=optimization_dtype,
        attn_implementation=attn_implementation,
        parallelism=ParallelismConfig(expert_parallel=ep_size, sequence_parallel=sp_size),
        patches=Patches(peft=peft_config),
        loader_options=config,
    )
