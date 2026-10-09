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

"""Gradients of the token mixers must not depend on how a step's rows are grouped into model calls.

Packing lets one model call carry several rows, so the same step can be run as one wide call or as several
narrow ones. Every token keeps its own weight either way, so exact arithmetic gives one gradient. In bf16 it
does not: regrouping reorders the sums the kernels carry out, and the recurrent delta rule amplifies rounding
sharply -- its gradients with respect to ``A_log`` and ``dt_bias`` move by percent-level amounts under a change
of one ulp in their inputs.

A tolerance chosen to accommodate that would be a number with no meaning. These tests measure the rounding
scale in the same run instead: perturb the input by one bf16 ulp, keep the schedule fixed, and see how far the
gradient moves. Regrouping must not move it further than that. A boundary or weighting defect is not bounded
this way and lands orders of magnitude above, which the last test demonstrates by merging two rows into a
single segment.

Module-level on purpose: one GPU, no gateway and no optimizer, so the measurement is of the kernels alone.
"""

from __future__ import annotations

import pytest

from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeConfig

ROW_LENGTH = 64
ROWS = 2
SEED = 11_000
# One bf16 mantissa step is a relative 2**-8; nudging the input by that much is the smallest change the dtype
# can represent, so the gradient movement it causes is the rounding scale of this path.
ONE_BF16_ULP = 2.0**-8


def _tiny_config():
    return Qwen3_5MoeConfig(
        vocab_size=256,
        hidden_size=512,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        linear_conv_kernel_dim=4,
        linear_key_head_dim=128,
        linear_num_key_heads=4,
        linear_value_head_dim=128,
        linear_num_value_heads=8,
        num_experts=2,
        num_experts_per_tok=1,
        moe_intermediate_size=64,
        shared_expert_intermediate_size=64,
        layer_types=["linear_attention"],
        use_grouped_mm=False,
    )


def _inputs(hidden_size, dtype, device):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    shape = (1, ROWS * ROW_LENGTH, hidden_size)
    hidden = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
    cotangent = torch.randn(shape, generator=generator).to(device=device, dtype=dtype)
    signs = torch.randint(0, 2, shape, generator=generator).float().mul_(2).sub_(1).to(device)
    nudged = (hidden.float() * (1.0 + ONE_BF16_ULP * signs)).to(dtype)
    return hidden, cotangent, nudged


def _gradients(module, hidden, cotangent, *, call_boundaries):
    """Run one step over ``call_boundaries`` and return the gradient of every parameter.

    ``call_boundaries`` lists the row windows that share a model call, so ``[(0, 128)]`` packs both rows into one
    call and ``[(0, 64), (64, 128)]`` gives each its own. Each token is divided by the step's token count rather
    than the call's, which is what makes the schedules comparable.
    """
    import torch

    module.zero_grad(set_to_none=True)
    total_tokens = ROWS * ROW_LENGTH

    for start, end in call_boundaries:
        window = slice(start, end)
        rows_in_call = (end - start) // ROW_LENGTH
        segments = [ROW_LENGTH * index for index in range(rows_in_call + 1)]
        boundaries = torch.tensor(segments, dtype=torch.int32, device=hidden.device)
        output = module(hidden[:, window], cu_seqlens=boundaries)
        ((output * cotangent[:, window]).sum() / total_tokens).backward()

    return {
        name: parameter.grad.detach().float().clone()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }


def _worst_relative_delta(left, right):
    worst, worst_name = 0.0, ""
    for name, value in left.items():
        other = right[name]
        scale = max(float(value.norm()), float(other.norm()), 1e-12)
        relative = float((value - other).norm()) / scale
        if relative > worst:
            worst, worst_name = relative, name
    return worst, worst_name


def _gated_delta_net(tmp_path):
    import torch

    from arctic_platform.model.implementations.qwen35.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeGatedDeltaNet,
    )

    del tmp_path
    config = _tiny_config()
    torch.manual_seed(SEED)
    module = Qwen3_5MoeGatedDeltaNet(config).to(device="cuda", dtype=torch.bfloat16).train()
    return module, config.hidden_size


PACKED = [(0, ROWS * ROW_LENGTH)]
SPLIT = [(row * ROW_LENGTH, (row + 1) * ROW_LENGTH) for row in range(ROWS)]


@pytest.mark.integration
def test_gated_delta_net_gradient_moves_less_under_regrouping_than_under_one_ulp(tmp_path):
    import torch

    if torch.cuda.device_count() < 1:
        pytest.skip("needs a CUDA device")

    module, hidden_size = _gated_delta_net(tmp_path)
    hidden, cotangent, nudged = _inputs(hidden_size, torch.bfloat16, next(module.parameters()).device)

    packed = _gradients(module, hidden, cotangent, call_boundaries=PACKED)
    split = _gradients(module, hidden, cotangent, call_boundaries=SPLIT)
    perturbed = _gradients(module, nudged, cotangent, call_boundaries=PACKED)
    repeated = _gradients(module, hidden, cotangent, call_boundaries=PACKED)

    regrouping, regrouping_name = _worst_relative_delta(packed, split)
    rounding, _ = _worst_relative_delta(packed, perturbed)
    repeat, _ = _worst_relative_delta(packed, repeated)

    assert repeat == 0.0, f"the same schedule twice must give the same gradient, got {repeat:.3e}"
    assert rounding > 0.0, "a one-ulp input change must move the gradient, or the measurement is not calibrating"
    assert regrouping <= rounding, (
        f"regrouping moved {regrouping_name} by {regrouping:.3e}, more than one input ulp does ({rounding:.3e}); "
        "a difference above the rounding scale is a defect in the boundaries or the per-token weights, not "
        "arithmetic noise"
    )


@pytest.mark.integration
def test_gated_delta_net_rejects_a_packed_call_whose_rows_share_one_segment(tmp_path):
    """Merging the rows' boundaries is what a boundary defect looks like, and it must be far outside rounding."""
    import torch

    if torch.cuda.device_count() < 1:
        pytest.skip("needs a CUDA device")

    module, hidden_size = _gated_delta_net(tmp_path)
    hidden, cotangent, nudged = _inputs(hidden_size, torch.bfloat16, next(module.parameters()).device)

    packed = _gradients(module, hidden, cotangent, call_boundaries=PACKED)
    perturbed = _gradients(module, nudged, cotangent, call_boundaries=PACKED)
    rounding, _ = _worst_relative_delta(packed, perturbed)

    module.zero_grad(set_to_none=True)
    merged = torch.tensor([0, ROWS * ROW_LENGTH], dtype=torch.int32, device=hidden.device)
    output = module(hidden, cu_seqlens=merged)
    ((output * cotangent).sum() / (ROWS * ROW_LENGTH)).backward()
    leaked = {
        name: parameter.grad.detach().float().clone()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }

    contamination, contamination_name = _worst_relative_delta(packed, leaked)
    assert contamination > rounding, (
        f"letting the recurrent state cross the row boundary moved {contamination_name} by only "
        f"{contamination:.3e}, within the rounding scale {rounding:.3e}; then this comparison cannot tell a "
        "boundary defect from noise and the test above proves nothing"
    )
