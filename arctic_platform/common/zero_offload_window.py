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
"""Keep DeepSpeed ZeRO-2's CPU-offload gradient buffer current when several boundaries precede one ``step()``.

With CPU optimizer offload and fp32 master weights for bf16 parameters, DeepSpeed 0.19.7 and 0.19.8 keep each
parameter's running gradient sum in a bf16 host buffer, and a backward adds that buffer whenever it is not the first
backward since ``step()``. The first backward skips writing the buffer when it is also a gradient-accumulation
boundary. DSS marks a boundary on the last model call of every ``/forward-backward`` request, and several requests
may precede one ``/step``, so a one-call request (a zero-advantage keepalive, say) followed by another request makes
that request add whatever an earlier step left in the buffer. Saving every backward keeps the buffer current, so
``/step`` applies the sum of the requests since the last ``/step``.
"""

from __future__ import annotations

_PATCHED = False


def patch_zero2_offload_boundary_window() -> None:
    import deepspeed

    if deepspeed.__version__ not in ("0.19.7", "0.19.8"):
        raise RuntimeError(
            f"The ZeRO-2 offload buffer fix is verified for DeepSpeed 0.19.7 and 0.19.8, got {deepspeed.__version__}; "
            "validate the new version and remove or update this workaround before training."
        )
    global _PATCHED
    if _PATCHED:
        return
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer as Zero

    copy_grads_in_partition = Zero.copy_grads_in_partition

    def uses_host_buffer(zero, param) -> bool:
        # With low-precision masters the buffer is the master gradient itself, which the boundary writes.
        if not zero.cpu_offload or zero.low_precision_master_weights_and_grads:
            return False
        compact = getattr(zero, "_use_compact_grad_partition_buffer", None)  # 0.19.8 fp32 accumulation has no buffer
        return compact is None or not compact(zero._get_param_group_idx(param))

    def save_every_backward(zero, param, *args, **kwargs):
        if zero.micro_step_id == 0 and zero.is_gradient_accumulation_boundary() and uses_host_buffer(zero, param):
            zero.async_accumulate_grad_in_cpu_via_gpu(param)  # the stock path skips this save for a lone backward
            zero._offload_accumulated_param_ids.add(zero.get_param_id(param))
        return copy_grads_in_partition(zero, param, *args, **kwargs)

    Zero.copy_grads_in_partition = save_every_backward
    _PATCHED = True
