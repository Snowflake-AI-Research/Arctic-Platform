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
"""Keep FP32 parameter groups FP32 through a ZeRO-1/2 engine that runs BF16 masters and BF16 gradient staging.

``bf16_master_weights_and_grads`` gives every optimizer group a BF16 master partition, and ZeRO-2 copies each
gradient into a BF16 staging bucket before it reduces. A parameter the model keeps in FP32 (``mark_keep_fp32``) has
a group of its own, and both steps round it: its master loses every update smaller than a BF16 ulp, and its
gradient is rounded before the FP32 reduction. This restores FP32 for those groups only:

* the master partition of a group whose parameters are all FP32 is rebuilt in FP32 from the parameters themselves;
* ZeRO-2's contiguous gradient buckets stage in FP32, which every other gradient fits in exactly;
* the reduced gradient partition is FP32 (``data_types.grad_accum_dtype``), which ``require_fp32_grad_accumulation``
  checks, because the optimizer step reads the partition in the dtype that accumulation used.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)


def _zero_optimizer_class():
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

    return DeepSpeedZeroOptimizer


def requires_fp32_grad_accumulation(ds_config: dict[str, Any]) -> bool:
    """Whether ``ds_config`` is a ZeRO-1/2 job on BF16 parameters, where FP32 groups need FP32 accumulation."""
    bf16_config = ds_config.get("bf16")
    zero_config = ds_config.get("zero_optimization")
    return bool(
        isinstance(bf16_config, dict)
        and bf16_config.get("enabled")
        and isinstance(zero_config, dict)
        and zero_config.get("stage") in (1, 2)
    )


def require_fp32_grad_accumulation(ds_config: dict[str, Any]) -> dict[str, Any]:
    """``ds_config`` with FP32 gradient accumulation, for a model that holds FP32 parameters.

    An unset ``data_types.grad_accum_dtype`` is DeepSpeed's model dtype, which would round an FP32 group's reduced
    gradient to BF16 once more, so it is set to FP32. An explicit lower dtype is refused rather than overridden.
    """
    data_types = dict(ds_config.get("data_types") or {})
    configured = data_types.get("grad_accum_dtype")
    if configured is None:
        data_types["grad_accum_dtype"] = "fp32"
    elif str(configured).lower() not in ("fp32", "float32"):
        raise ValueError(
            f"data_types.grad_accum_dtype={configured!r} would round the model's FP32 parameter gradients to a "
            "lower precision after the FP32 reduction; use fp32 or remove it"
        )
    return {**ds_config, "data_types": data_types}


def fp32_group_indices(optimizer) -> list[int]:
    return [
        index
        for index, group in enumerate(optimizer.bit16_groups)
        if len(group) > 0 and all(param.dtype == torch.float32 for param in group)
    ]


def _promote_master(optimizer, index: int) -> None:
    import deepspeed.comm as dist

    old = optimizer.single_partition_of_fp32_groups[index]
    if old.dtype == torch.float32:
        return
    if optimizer.optimizer.state.get(old):
        raise RuntimeError("an FP32 group's master was already stepped; promote masters right after initialize")
    partition_id = dist.get_rank(group=optimizer.real_dp_process_group[index])
    source = optimizer.parallel_partitioned_bit16_groups[index][partition_id]
    master = source.detach().clone().to(device=old.device, dtype=torch.float32)
    if optimizer.cpu_offload and optimizer.cpu_offload_pin_memory:
        master = master.pin_memory()
    master.requires_grad = True
    if old.grad is not None:
        grad = torch.zeros_like(master)
        master.grad = grad.pin_memory() if optimizer.cpu_offload and optimizer.cpu_offload_pin_memory else grad
    optimizer.single_partition_of_fp32_groups[index] = master
    optimizer.optimizer.param_groups[index]["params"] = [master]


def _stage_gradients_in_fp32(optimizer) -> None:
    if not hasattr(optimizer, "_gradient_reduction_buffer_dtype"):
        raise RuntimeError(
            "this DeepSpeed does not expose ZeRO-2's gradient staging dtype, so FP32 parameter gradients cannot be "
            "kept out of BF16 staging; use a DeepSpeed that does or train without FP32 parameters"
        )
    optimizer._gradient_reduction_buffer_dtype = lambda: torch.float32
    for bucket in optimizer.ipg_buckets.values():
        if any(buffer.dtype != torch.float32 for buffer in bucket.buffer):
            bucket.buffer.clear()
            optimizer.ready_for_gradients = False


def protect_fp32_parameter_groups(engine) -> int:
    """Keep the FP32 groups of ``engine``'s ZeRO-1/2 optimizer FP32; returns how many there are.

    Call right after ``deepspeed.initialize``, before any backward or optimizer step. An engine with no FP32
    group, or whose optimizer is not ZeRO-1/2, is left unchanged.
    """
    optimizer = getattr(engine, "optimizer", None)
    if optimizer is None or not isinstance(optimizer, _zero_optimizer_class()):
        return 0
    indices = fp32_group_indices(optimizer)
    if not indices:
        return 0
    if optimizer.gradient_accumulation_dtype != torch.float32:
        raise RuntimeError(
            f"ZeRO accumulates gradients in {optimizer.gradient_accumulation_dtype}, which rounds the FP32 "
            "parameter groups' gradients; set data_types.grad_accum_dtype to fp32"
        )
    for index in indices:
        _promote_master(optimizer, index)
    optimizer._link_all_hp_params()
    if optimizer.partition_gradients and optimizer.contiguous_gradients:
        _stage_gradients_in_fp32(optimizer)
    logger.info("kept %d FP32 optimizer group(s) in FP32 masters and gradients", len(indices))
    return len(indices)
