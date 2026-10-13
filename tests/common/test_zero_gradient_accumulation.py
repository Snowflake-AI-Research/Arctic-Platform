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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Exercise installed ZeRO accumulation methods without a GPU or process group."""

import pytest
import torch
from deepspeed.runtime.base_optimizer import BackwardHookStateManager
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer


def zero_optimizer(parameter, accumulation_dtype):
    optimizer = object.__new__(DeepSpeedZeroOptimizer)
    optimizer._backward_hook_state = BackwardHookStateManager()
    optimizer.use_grad_accum_attribute = False
    optimizer.gradient_accumulation_dtype = accumulation_dtype
    optimizer.bit16_groups = [[parameter]]
    optimizer.params_in_partition = [[parameter]]
    optimizer.params_already_reduced = [False]
    optimizer.all_grad_tensors = {}
    optimizer.cpu_offload = False
    optimizer.overlap_comm = False
    optimizer.report_ipg_memory_usage = lambda *args: None
    optimizer.reduce_ipg_grads = lambda: None
    optimizer._release_ipg_buffers = lambda: None
    optimizer._clear_param_grad_only = lambda: setattr(parameter, "grad", None)
    optimizer.is_gradient_accumulation_boundary = lambda: False
    return optimizer


@pytest.mark.parametrize("gradient_dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("accumulation_dtype", [torch.bfloat16, torch.float32])
def test_existing_gradient_uses_requested_accumulation_dtype(gradient_dtype, accumulation_dtype):
    parameter = torch.nn.Parameter(torch.ones(4, dtype=gradient_dtype))
    parameter.grad = torch.arange(4, dtype=gradient_dtype)
    original_gradient = parameter.grad.clone()
    optimizer = zero_optimizer(parameter, accumulation_dtype)

    gradient = optimizer.get_all_grad_tensors([parameter], dtype=accumulation_dtype)[0]

    assert gradient.dtype == accumulation_dtype
    torch.testing.assert_close(gradient, original_gradient.to(accumulation_dtype), rtol=0, atol=0)
    torch.testing.assert_close(parameter.grad, original_gradient, rtol=0, atol=0)


@pytest.mark.parametrize("accumulation_dtype", [torch.bfloat16, torch.float32])
def test_missing_gradient_uses_requested_accumulation_dtype(accumulation_dtype):
    parameter = torch.nn.Parameter(torch.ones(4, dtype=torch.bfloat16))
    optimizer = zero_optimizer(parameter, accumulation_dtype)

    gradient = optimizer.get_all_grad_tensors([parameter], dtype=accumulation_dtype)[0]

    assert gradient.dtype == accumulation_dtype
    torch.testing.assert_close(gradient, torch.zeros(4, dtype=accumulation_dtype), rtol=0, atol=0)
    assert parameter.grad is None


@pytest.mark.parametrize(
    ("accumulation_dtype", "expected_sum"),
    [(torch.float32, 259.0), (torch.bfloat16, 256.0)],
)
def test_partition_epilogue_accumulates_in_configured_precision(accumulation_dtype, expected_sum):
    parameter = torch.nn.Parameter(torch.ones(1, dtype=torch.bfloat16))
    optimizer = zero_optimizer(parameter, accumulation_dtype)
    # BF16 can represent each input but rounds each small addition to 256 away.
    for contribution in (256.0, 1.0, 1.0, 1.0):
        parameter.grad = torch.full_like(parameter, contribution)
        optimizer.independent_gradient_partition_epilogue()
        assert parameter.grad is None

    accumulated = optimizer.all_grad_tensors[0][0]
    assert accumulated.item() == expected_sum
    assert accumulated.dtype == accumulation_dtype
