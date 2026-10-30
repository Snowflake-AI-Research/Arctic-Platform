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
"""Make DeepSpeed ZeRO-2 sum one step's model calls in ``data_types.grad_accum_dtype``.

DeepSpeed 0.19.7 ignores that dtype at stage 2. Without optimizer offload the first call's bf16 gradient partition
becomes the running sum; with CPU offload a bf16 host buffer per parameter does. Either way fp32 master weights
receive a bf16 sum of every model call of the step. This applies an fp32 accumulation fix on 0.19.7 only;
0.19.8 uses its native fp32 partition accumulator.
It changes nothing unless the accumulation dtype differs from the parameter dtype.
"""

from __future__ import annotations

import torch

_PATCHED = False


def patch_zero2_grad_accum_dtype() -> None:
    import deepspeed

    if deepspeed.__version__ == "0.19.8":
        return  # Native fp32 compact partition accumulation replaces this workaround.
    if deepspeed.__version__ != "0.19.7":
        raise RuntimeError(
            f"ZeRO-2 accumulation support is verified for DeepSpeed 0.19.7 and 0.19.8, got {deepspeed.__version__}; "
            "validate the new version and remove or update the 0.19.7 workaround before training."
        )
    global _PATCHED
    if _PATCHED:
        return
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer as Zero
    from deepspeed.runtime.zero.utils import get_norm_dtype

    epilogue = Zero.independent_gradient_partition_epilogue
    copy_grads_in_partition = Zero.copy_grads_in_partition
    finalize_offload = Zero._finalize_cpu_offload_gradient_accumulation

    def accumulates_in_master_grads(zero) -> bool:
        return (
            zero.cpu_offload
            and zero.dtype != zero.gradient_accumulation_dtype
            and zero.gradient_accumulation_dtype == zero.master_weights_and_grads_dtype
        )

    def master_grad_partition_of(zero, param) -> torch.Tensor:
        [i, _, dest_offset, num_elements] = zero.grad_position[zero.get_param_id(param)]
        return zero.single_partition_of_fp32_groups[i].grad.view(-1).narrow(0, dest_offset, num_elements)

    def independent_gradient_partition_epilogue(zero):
        fresh = [i for i in range(len(zero.bit16_groups)) if zero.all_grad_tensors.get(i) is None]
        epilogue(zero)
        for i in fresh:
            if zero.all_grad_tensors.get(i) is not None:  # a window opened and is still accumulating
                zero.all_grad_tensors[i] = [g.to(zero.gradient_accumulation_dtype) for g in zero.all_grad_tensors[i]]

    def accumulate_grad_in_master_partition(zero, param):
        if not accumulates_in_master_grads(zero):
            return copy_grads_in_partition(zero, param)
        if zero.temp_grad_buffer_for_gpu_offload.dtype != zero.gradient_accumulation_dtype:
            # One partition slice at a time in the accumulation dtype, replacing the whole-parameter bit16 buffer.
            zero.temp_grad_buffer_for_gpu_offload = torch.empty(
                max(position[3] for position in zero.grad_position.values()),
                dtype=zero.gradient_accumulation_dtype,
                device=zero.temp_grad_buffer_for_gpu_offload.device,
            )
        param_id = zero.get_param_id(param)
        [_, source_offset, _, num_elements] = zero.grad_position[param_id]
        master_grad = master_grad_partition_of(zero, param)
        accumulator = zero.temp_grad_buffer_for_gpu_offload.narrow(0, 0, num_elements)
        grad = zero.get_param_gradient_attribute(param).view(-1).narrow(0, source_offset, num_elements)
        if param_id in zero._offload_accumulated_param_ids:
            accumulator.copy_(master_grad, non_blocking=True)
            accumulator.add_(grad)
        else:
            accumulator.copy_(grad)
            zero._offload_accumulated_param_ids.add(param_id)
        if zero.is_gradient_accumulation_boundary():
            zero.norm_for_param_grads[param_id] = accumulator.to(get_norm_dtype()).norm(2)
            zero.update_offload_overflow_tracker(accumulator)
        master_grad.copy_(accumulator, non_blocking=True)
        zero.clear_grad_attribute(param)

    def finalize_cpu_offload_gradient_accumulation(zero):
        if not accumulates_in_master_grads(zero):
            return finalize_offload(zero)
        for group in zero.params_in_partition:
            for param in group:
                if param.requires_grad and zero.get_param_id(param) in zero._offload_accumulated_param_ids:
                    master_grad = master_grad_partition_of(zero, param)
                    # Order the read after the nonblocking D2H copy on the same stream.
                    accumulator = zero.temp_grad_buffer_for_gpu_offload.narrow(0, 0, master_grad.numel())
                    accumulator.copy_(master_grad, non_blocking=True)
                    zero.norm_for_param_grads[zero.get_param_id(param)] = accumulator.to(get_norm_dtype()).norm(2)
                    zero.update_offload_overflow_tracker(accumulator)

    Zero.independent_gradient_partition_epilogue = independent_gradient_partition_epilogue
    Zero.copy_grads_in_partition = accumulate_grad_in_master_partition
    Zero._finalize_cpu_offload_gradient_accumulation = finalize_cpu_offload_gradient_accumulation
    _PATCHED = True
