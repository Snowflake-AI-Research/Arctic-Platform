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

from copy import deepcopy
from typing import Any

from arctic_platform.common.config import resolve_sp_size
from arctic_platform.model.config import ActivationCheckpointConfig
from arctic_platform.model.config import ModelSpec
from arctic_platform.model.config import ParallelismConfig
from arctic_platform.model.config import Patches
from arctic_platform.model.implementations.debug.gpu_memory import gpu_memory_debug_config
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
    if model_type in {"inkling_mm_model", "inkling_text", "inkling2"}:
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


def ds_worker_moe_model_spec(model_name: str, ds_worker_config: dict) -> ModelSpec:
    """``moe_model_spec`` for the on-prem worker's flat ``ds_worker_config`` when ``ep_size > 1``.

    The loader owns the forward and checkpointing options, so worker-level ZoRRo Train, Liger, and
    gradient checkpointing are refused. ``model_options`` and the flat checkpointing keys fold into
    the ``prime_rl`` loader options, and a key given twice must agree.
    """
    cfg = ds_worker_config or {}
    if cfg.get("attn_implementation") is None:
        raise ValueError("ds_worker_config requires attn_implementation")
    parallelism = ParallelismConfig.model_validate(
        dict(
            **({"expert_parallel": cfg["ep_size"]} if "ep_size" in cfg else {}),
            sequence_parallel=resolve_sp_size({"ds_worker_config": cfg}),
        )
    )
    if parallelism.expert_parallel <= 1:
        raise ValueError("ds_worker_moe_model_spec requires ep_size > 1")
    if cfg.get("zorro_train_enable") or cfg.get("use_liger") or cfg.get("enable_gradient_checkpointing"):
        raise ValueError("expert-parallel workers require loader-owned forward/checkpointing options")
    options = deepcopy(cfg.get("prime_rl") or {})
    for key in ("ac_config", "tiled_mlp_token_chunk_size", "weight_conversion_cache_dir"):
        if key in cfg:
            if key in options and options[key] != cfg[key]:
                raise ValueError(f"conflicting worker and prime_rl {key}")
            options[key] = deepcopy(cfg[key])
    checkpointing = cfg.get("activation_checkpointing")
    if checkpointing is False and options.get("ac_config") is not None:
        raise ValueError("activation_checkpointing=False conflicts with ac_config")
    if checkpointing is True and options.get("ac_config") is None:
        options["ac_config"] = ActivationCheckpointConfig().model_dump()
    model_options = cfg.get("model_options") or {}
    if set(model_options) - {"lm_head", "moe"}:
        raise ValueError("unsupported expert-parallel worker model_options")
    lm_head = model_options.get("lm_head") or {}
    head_keys = {
        "cross_entropy": "fused_cross_entropy",
        "token_chunk_size": "fused_lm_head_token_chunk_size",
        "fp32": "fp32_lm_head",
    }
    if set(lm_head) - head_keys.keys():
        raise ValueError("unsupported expert-parallel worker lm_head options")
    for key, value in lm_head.items():
        target = head_keys[key]
        if target in options and options[target] != value:
            raise ValueError(f"conflicting model_options.lm_head.{key} and prime_rl.{target}")
        options[target] = value
    moe = model_options.get("moe") or {}
    moe_keys = {
        "comm_backend": "ep_comm_backend",
        "comm_sms": "deepep_num_sms",
        "comm_token_chunk": "deepep_token_chunk_size",
        "grouped_mm": "moe_use_grouped_mm",
    }
    if set(moe) - moe_keys.keys():
        raise ValueError("unsupported expert-parallel worker moe options")
    for key, value in moe.items():
        target = moe_keys[key]
        if target in options and options[target] != value:
            raise ValueError(f"conflicting model_options.moe.{key} and prime_rl.{target}")
        options[target] = value
    if "attn" in options and options["attn"] != cfg["attn_implementation"]:
        raise ValueError("conflicting prime_rl.attn and attn_implementation")
    options["attn"] = cfg["attn_implementation"]
    spec = moe_model_spec(
        model_name=model_name,
        ep_size=parallelism.expert_parallel,
        sp_size=parallelism.sequence_parallel,
        loader_options=options,
        peft_config=cfg.get("peft_config"),
    )
    spec.debug = {**spec.debug, **gpu_memory_debug_config(cfg)}
    return spec
