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


def hf_text_model_type(model_path: str) -> str:
    """Read the text model type without requiring its config class to be registered."""
    from transformers import PretrainedConfig

    config, _ = PretrainedConfig.get_config_dict(model_path)
    text_config = config.get("text_config")
    if isinstance(text_config, dict) and text_config.get("model_type"):
        return str(text_config["model_type"])
    return str(config.get("model_type") or "")


def is_glm_moe_model(model_path: str) -> bool:
    return hf_text_model_type(model_path) in {"glm_moe_dsa", "glm5_next_text"}


def build_moe_model(
    *,
    model_name: str,
    ep_size: int,
    sp_size: int = 1,
    sp_group: Any = None,
    loader_options: dict | None = None,
    tiled_mlp_token_chunk_size: int | None = None,
    peft_config: dict | None = None,
):
    """Build an expert-parallel MoE from training options and the registered expert group."""
    from arctic_platform.model import build_model
    from arctic_platform.model.parallel.groups import expert_parallel_group

    spec = moe_model_spec(
        model_name=model_name,
        ep_size=ep_size,
        sp_size=sp_size,
        loader_options=loader_options,
        tiled_mlp_token_chunk_size=tiled_mlp_token_chunk_size,
        peft_config=peft_config,
    )
    return build_model(
        spec,
        parallel_groups={"ep_group": expert_parallel_group(ep_size), "sp_group": sp_group},
    ).model


def prepare_moe_weights(
    *,
    model_name: str,
    ep_size: int,
    sp_size: int = 1,
    loader_options: dict | None = None,
) -> None:
    """Write the node-local weight cache ``build_moe_model`` reads (the HF->Prime conversion), with no process group.

    A runtime runs it once per node before it starts the ranks, so a cold node's conversion runs outside their
    start-up deadline and first collective; ``build_moe_model`` then reuses the cache.
    """
    from arctic_platform.model.loader import prepare_weights

    prepare_weights(moe_model_spec(model_name=model_name, ep_size=ep_size, sp_size=sp_size, loader_options=loader_options))


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
