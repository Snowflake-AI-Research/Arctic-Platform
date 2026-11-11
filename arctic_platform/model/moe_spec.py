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
"""Build the ModelSpec used to load an expert-parallel MoE.

``moe_model_spec`` is the one translation from MoE loader options to a ``ModelSpec``. Both servers call it through
``settings_moe_model_spec`` with the job's validated ``ModelSettings``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from arctic_platform.common import model_option_aliases as aliases
from arctic_platform.common.model_settings import ModelSettings
from arctic_platform.model.config import ModelSpec
from arctic_platform.model.config import ParallelismConfig
from arctic_platform.model.config import Patches
from arctic_platform.model.implementations.moe.config_validation import effective_fused_cross_entropy
from arctic_platform.model.implementations.moe.config_validation import validate_lm_head_fused_ce_config
from arctic_platform.model.loader import model_type_family


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
    return model_type_family(hf_text_model_type(model_path)) in {"glm_moe_dsa", "glm5_next"}


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
    model_type = hf_text_model_type(model_name)
    if model_type in {"inkling_mm_model", "inkling_text"}:
        config.setdefault("fused_cross_entropy", False)
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


def settings_moe_loader_options(
    settings: ModelSettings,
    *,
    seq_len: int | None = None,
    trust_remote_code: bool | None = None,
    weight_conversion_cache_dir: str | None = None,
) -> dict[str, Any]:
    """The expert-parallel loaders' options for a job's typed ``ModelSettings``, as ``moe_model_spec`` takes them.

    On this route an ``ac_config`` block alone turns activation checkpointing on; ``activation_checkpointing`` and
    the dense-route fields are not read. The ``prime_rl`` block overrides ``ac_config`` and ``seq_len``, and the
    job-level keys override the block.
    """
    options: dict[str, Any] = {"ac_config": deepcopy(settings.ac_config)}
    if seq_len is not None:
        options["seq_len"] = seq_len
    options.update(deepcopy(settings.prime_rl or {}))
    options["attn"] = settings.resolved_attn_implementation()
    if trust_remote_code is not None:
        options["trust_remote_code"] = trust_remote_code
    if weight_conversion_cache_dir is not None:
        options["weight_conversion_cache_dir"] = weight_conversion_cache_dir
    return options


def settings_moe_model_spec(
    settings: ModelSettings,
    *,
    model_path: str,
    sequence_parallel: int,
    seq_len: int | None = None,
    trust_remote_code: bool | None = None,
    weight_conversion_cache_dir: str | None = None,
    debug: dict | None = None,
) -> ModelSpec:
    """``moe_model_spec`` for a job's typed ``ModelSettings`` with ``ep_size > 1``, which both servers validate the
    same way."""
    ep_size = settings.ep_size or 1
    if not aliases.moe_route(ep_size):
        raise ValueError("settings_moe_model_spec requires ep_size > 1")
    if settings.model_provider != "arctic":
        raise ValueError("ep_size>1 (expert parallelism) is only supported for the arctic model provider")
    spec = moe_model_spec(
        model_name=model_path,
        ep_size=ep_size,
        sp_size=sequence_parallel,
        loader_options=settings_moe_loader_options(
            settings,
            seq_len=seq_len,
            trust_remote_code=trust_remote_code,
            weight_conversion_cache_dir=weight_conversion_cache_dir,
        ),
        tiled_mlp_token_chunk_size=settings.tiled_mlp_token_chunk_size,
        peft_config=settings.peft_config,
    )
    if debug:
        spec.debug = {**spec.debug, **debug}
    return spec
