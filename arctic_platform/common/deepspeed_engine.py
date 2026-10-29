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
"""DeepSpeed engine bring-up for a training job.

Builds the DeepSpeed config, constructs the optimizer, and calls
``deepspeed.initialize``. This is separate from model construction. The
on-prem RL worker in ``arctic_platform.common.deepspeed_worker`` still builds
its own config in ``ds_training_config`` and calls ``deepspeed.initialize``
itself.

TODO: combine this module with ``deepspeed_worker`` once both callers share
one engine-construction path.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from arctic_platform.common.zero_grad_accum import patch_zero2_grad_accum_dtype
from arctic_platform.model.implementations.fp8 import preserve_fp8_params_through_ds_cast
from arctic_platform.model.moe_spec import is_glm_moe_model

logger = logging.getLogger(__name__)

_NO_DECAY = ("bias", "layer_norm.weight", "layernorm.weight", "norm.weight", "ln_f.weight")
_OPTIMIZER_ALIASES = {
    "fusedadam": "fused_adam",
    "fused_adam": "fused_adam",
    "torch_adamw": "torch_adamw",
    "deepspeed_cpu_adam": "deepspeed_cpu_adam",
    "deepspeedcpuadam": "deepspeed_cpu_adam",
    "cpu_adam": "deepspeed_cpu_adam",
    "cpuadam": "deepspeed_cpu_adam",
    "deep_speed_cpu_adam": "deepspeed_cpu_adam",
}


class OptimizerSettings(BaseModel):
    """Optimizer block handed to DeepSpeed after the caller has resolved aliases."""

    model_config = ConfigDict(extra="forbid")

    name: str = "AdamW"
    lr: float = 1e-5
    weight_decay: float = 0.0
    betas: list[float] = Field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    # ``None`` lets the model family decide. GLM-MoE leaves FusedAdam off.
    fused: bool | None = None


class TrainingEngineSpec(BaseModel):
    """Engine inputs that do not depend on a training service.

    ``fp32_masters`` and ``zero_bucket_size`` come from the caller's job config.
    ``ds_overrides`` is shallow-merged last, matching a client DeepSpeed
    override block.
    """

    model_config = ConfigDict(extra="forbid")

    dtype: str = "bfloat16"
    world_size: int
    sequence_parallel: int = 1
    gradient_clipping: float = 1.0
    seed: int | None = None
    ds_overrides: dict[str, Any] | None = None
    fp32_masters: bool = False
    zero_bucket_size: int | None = None
    optimizer: OptimizerSettings | None = None


def build_deepspeed_config(spec: TrainingEngineSpec) -> dict[str, Any]:
    """Build the config dict consumed by ``deepspeed.initialize``.

    The caller steps after each microbatch, so the microbatch size and the
    accumulation count are both 1. Sequence parallelism reduces the
    data-parallel degree, and therefore ``train_batch_size``, to
    ``world_size // sequence_parallel``.
    """
    dtype = (spec.dtype or "bfloat16").lower()
    is_bf16 = dtype in ("bfloat16", "bf16")
    is_fp16 = dtype in ("float16", "fp16")
    # bf16 master weights match FusedAdam's training dynamics. fp32 masters
    # are the opt-in that disables them.
    bf16_masters = is_bf16 and not spec.fp32_masters
    ds_config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": _data_parallel_size(spec.world_size, spec.sequence_parallel),
        "gradient_accumulation_steps": 1,
        "gradient_clipping": spec.gradient_clipping,
        "bf16": {
            "enabled": is_bf16,
            "bf16_master_weights_and_grads": bf16_masters,
            "bf16_optimizer_states": bf16_masters,
        },
        "fp16": {"enabled": is_fp16},
        "zero_optimization": {"stage": 2},
        # NCCL bf16 all-reduce is not accumulated in fp32 internally.
        "communication_data_type": "fp32" if is_bf16 else None,
    }
    if spec.seed is not None:
        ds_config["seed"] = int(spec.seed)
    if spec.sequence_parallel > 1:
        # A DeepSpeedConfig built from this dict without the SP mpu reads this key for the dp degree;
        # without it DeepSpeed checks ``train_batch_size`` against the full world size and asserts.
        ds_config["sequence_parallel_size"] = spec.sequence_parallel
    if spec.zero_bucket_size is not None:
        zero_config = ds_config["zero_optimization"]
        zero_config["reduce_bucket_size"] = spec.zero_bucket_size
        zero_config["allgather_bucket_size"] = spec.zero_bucket_size
    if spec.ds_overrides:
        ds_config.update(spec.ds_overrides)
    _require_fp32_gradient_reduction(ds_config)
    _accumulate_fp32_master_gradients_in_fp32(ds_config)
    logger.info(
        "Resolved DeepSpeed precision: bf16=%s fp16=%s communication_data_type=%s",
        ds_config.get("bf16"),
        ds_config.get("fp16"),
        ds_config.get("communication_data_type"),
    )
    return ds_config


def split_expert_param_groups(param_groups: list) -> list:
    """Name optimizer groups and split MoE parameters onto the expert-data-parallel group.

    DeepSpeed only does this split when it builds the optimizer itself. A caller-supplied
    optimizer has to hand it groups that already separate expert parameters.
    """
    from deepspeed.moe.utils import configure_moe_param_groups

    named = []
    for index, group in enumerate(param_groups):
        group.setdefault("name", f"group_{index}")
        named.append(group)
    return configure_moe_param_groups(named)


def initialize_training_engine(
    model,
    ds_config: dict[str, Any],
    *,
    optimizer: OptimizerSettings | None,
    ep_size: int,
    mpu,
    model_path: str,
    rank: int,
):
    """Build the optimizer, keep FP8 storage, and call ``deepspeed.initialize``.

    Returns ``(engine, optimizer, optimizer_choice)``. ``optimizer`` and
    ``optimizer_choice`` are ``None`` when the job has no optimizer.
    ``mpu`` is forwarded only when it is set, so ZeRO reduces sequence-parallel
    gradients over the sequence-data-parallel group.
    """
    built = None
    choice = None
    if optimizer is not None:
        fused = optimizer.fused
        if fused is None:
            fused = not is_glm_moe_model(model_path)
        built = build_training_optimizer(model, optimizer, ds_config, fused=fused)
        choice = optimizer_choice(optimizer, ds_config, fused=fused)
        if ep_size > 1:
            built.param_groups[:] = split_expert_param_groups(list(built.param_groups))
    preserved, marked_scales = preserve_fp8_params_through_ds_cast(model)
    if preserved:
        from deepspeed.runtime.engine import DeepSpeedEngine

        logger.info(
            "rank=%d preserving %d FP8 parameters and %d FP32 block scales through the "
            "DeepSpeed dtype cast (deepspeed_param_cast_guard=%d)",
            rank,
            preserved,
            marked_scales,
            int(bool(getattr(DeepSpeedEngine, "_ap_fp8_param_cast_guard", False))),
        )
    import deepspeed

    patch_zero2_grad_accum_dtype()
    init_kwargs = {"mpu": mpu} if mpu is not None else {}
    if built is not None:
        engine, built, _, _ = deepspeed.initialize(model=model, optimizer=built, config=ds_config, **init_kwargs)
    else:
        engine, *_ = deepspeed.initialize(model=model, config=ds_config, **init_kwargs)
        built = None
    return engine, built, choice


def resolve_optimizer_name(name: str, ds_config: dict) -> str:
    """Map a caller optimizer name onto the implementation DeepSpeed will run."""
    resolved = name.strip().lower().replace("-", "_")
    if resolved in {"", "adam", "adamw"}:
        return "deepspeed_cpu_adam" if _optimizer_cpu_offload_enabled(ds_config) else "fused_adam"
    if resolved not in _OPTIMIZER_ALIASES:
        supported = ", ".join(sorted(set(_OPTIMIZER_ALIASES.values())))
        raise ValueError(f"Unsupported optimizer {name!r}; supported values: {supported}")
    return _OPTIMIZER_ALIASES[resolved]


def optimizer_implementation(optimizer_name: str, *, fused: bool) -> str:
    """The class ``build_training_optimizer`` constructs for a resolved name."""
    if optimizer_name == "fused_adam" and fused:
        return "fused_adam"
    if optimizer_name == "deepspeed_cpu_adam":
        return "deepspeed_cpu_adam"
    if optimizer_name in ("torch_adamw", "fused_adam"):
        return "torch_adamw"
    raise AssertionError(f"Unhandled optimizer {optimizer_name!r}")


def optimizer_choice(optimizer: OptimizerSettings, ds_config: dict, *, fused: bool) -> dict[str, str]:
    """The implementation a job runs and the customer-facing reason."""
    optimizer_name = resolve_optimizer_name(optimizer.name, ds_config)
    implementation = optimizer_implementation(optimizer_name, fused=fused)
    if implementation == "deepspeed_cpu_adam":
        reason = (
            "optimizer state is offloaded to CPU, so the CPU implementation (CPUAdam) ran"
            if _optimizer_cpu_offload_enabled(ds_config)
            else f"you selected '{optimizer.name}'"
        )
    elif implementation == "fused_adam":
        reason = "AdamW ran as the faster fused GPU implementation (FusedAdam)"
    elif optimizer_name == "torch_adamw":
        reason = f"you selected '{optimizer.name}'"
    elif optimizer.fused is False:
        reason = "you set 'optimizer.fused' to false, so standard AdamW ran"
    else:
        reason = "FusedAdam isn't supported for this model, so standard AdamW ran"
    return {"implementation": implementation, "reason": reason}


def cpu_adam_fp32_optimizer_states(ds_config: dict) -> bool:
    bf16_config = ds_config.get("bf16")
    if isinstance(bf16_config, dict) and bf16_config.get("bf16_optimizer_states"):
        return False
    return True


def build_training_optimizer(model, optimizer: OptimizerSettings, ds_config: dict, *, fused: bool):
    """Build AdamW with decay disabled on bias and norm parameters.

    GPU ``FusedAdam`` by default, or ``DeepSpeedCPUAdam`` when ZeRO optimizer
    offload is enabled.
    """
    optimizer_name = resolve_optimizer_name(optimizer.name, ds_config)
    if _optimizer_cpu_offload_enabled(ds_config) and optimizer_name == "deepspeed_cpu_adam":
        logger.info(
            "training_config.ds_config.zero_optimization.offload_optimizer is configured; "
            "using optimizer='deepspeed_cpu_adam'"
        )
    decay_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and not any(token in name.lower() for token in _NO_DECAY)
    ]
    no_decay_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and any(token in name.lower() for token in _NO_DECAY)
    ]
    param_groups = [
        {"params": decay_params, "weight_decay": optimizer.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    param_groups = [group for group in param_groups if group["params"]]
    betas = (optimizer.betas[0], optimizer.betas[1])
    implementation = optimizer_implementation(optimizer_name, fused=fused)
    if implementation == "fused_adam":
        from deepspeed.ops.adam import FusedAdam

        return FusedAdam(
            param_groups,
            lr=optimizer.lr,
            betas=betas,
            eps=optimizer.eps,
            adam_w_mode=True,
        )
    if implementation == "deepspeed_cpu_adam":
        from deepspeed.ops.adam import DeepSpeedCPUAdam

        return DeepSpeedCPUAdam(
            param_groups,
            lr=optimizer.lr,
            betas=betas,
            eps=optimizer.eps,
            adamw_mode=True,
            fp32_optimizer_states=cpu_adam_fp32_optimizer_states(ds_config),
        )
    import torch

    return torch.optim.AdamW(
        param_groups,
        lr=optimizer.lr,
        betas=betas,
        eps=optimizer.eps,
    )


def _data_parallel_size(world_size: int, sequence_parallel: int) -> int:
    if sequence_parallel > 1 and world_size % sequence_parallel != 0:
        raise ValueError(f"world_size ({world_size}) must be divisible by sp_size ({sequence_parallel})")
    return world_size // sequence_parallel


def _optimizer_cpu_offload_enabled(ds_config: dict) -> bool:
    zero_config = ds_config.get("zero_optimization")
    if not isinstance(zero_config, dict):
        return False
    return bool(zero_config.get("offload_optimizer"))


def _accumulate_fp32_master_gradients_in_fp32(ds_config: dict[str, Any]) -> None:
    """Sum a step's model calls in fp32 when ZeRO-2 keeps fp32 masters for bf16 parameters.

    DeepSpeed's default accumulation dtype is the parameter dtype, so the fp32 master gradient would be a bf16
    running sum of every model call of the step (``zero_grad_accum``). At stage 1 the same key would switch
    DeepSpeed to BF16_Optimizer and its full-size fp32 gradient buffer, so only stage 2 is set here.
    """
    bf16_config = ds_config["bf16"]
    if (
        bf16_config.get("enabled")
        and not bf16_config.get("bf16_master_weights_and_grads")
        and ds_config["zero_optimization"].get("stage") == 2
    ):
        ds_config.setdefault("data_types", {"grad_accum_dtype": "fp32"})


def _require_fp32_gradient_reduction(ds_config: dict[str, Any]) -> None:
    """Refuse a bfloat16 gradient reduction while the parameters are bfloat16.

    ``communication_data_type`` is the dtype gradient partials are summed in
    when they cross a rank. A bfloat16 job sets it to fp32 here, and a
    shallow client override can lower it afterwards. DeepSpeed's own default
    is the parameter dtype, so dropping the key is the same hazard as setting
    bfloat16. Anything other than fp32 is refused.
    """
    bf16_config = ds_config.get("bf16")
    if not (isinstance(bf16_config, dict) and bf16_config.get("enabled")):
        return
    reduction_dtype = str(ds_config.get("communication_data_type") or "").strip().lower()
    if reduction_dtype in ("fp32", "float32"):
        return
    raise ValueError(
        "communication_data_type must be fp32 while bf16 is enabled, because the gradient partials that cross "
        "a rank boundary are otherwise summed at bfloat16 granularity -- one step of about 2**-8 relative "
        "against 2**-24, which makes the result depend on how many ranks the token axis is split over; got "
        f"{ds_config.get('communication_data_type')!r}. Remove the training_config.ds_config override, or set "
        "dtype to float32 if the whole step is meant to run in float32."
    )
