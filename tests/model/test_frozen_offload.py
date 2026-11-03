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
"""Frozen routed-expert codes kept on the host and copied in per layer call, under activation checkpointing."""

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from arctic_platform.model.implementations.fp8 import quantize_to_fp8_blockwise
from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts

DIM, HIDDEN, EXPERTS, RANK, LAYERS = 1024, 512, 8, 8, 4
COUNTS = [3, 0, 17, 9, 1, 30, 4, 0]


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(DIM, DIM, bias=False)
        self.experts = GroupedExperts(DIM, HIDDEN, EXPERTS, use_grouped_mm=True, fp8_block_size=128)
        self.lora = nn.ParameterDict()

    def forward(self, x):
        counts = torch.tensor(COUNTS, dtype=torch.int32, device=x.device)
        scores = torch.linspace(0.1, 1.0, x.shape[0], device=x.device)
        return x + self.experts(self.proj(x), counts, scores)


def _fill(layer: _Layer, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, (rows, cols) in {"w1": (HIDDEN, DIM), "w2": (DIM, HIDDEN), "w3": (HIDDEN, DIM)}.items():
            dense = torch.randn(EXPERTS, rows, cols, generator=g) * 0.05
            codes, scales = zip(*(quantize_to_fp8_blockwise(dense[e].bfloat16()) for e in range(EXPERTS)))
            getattr(layer.experts, name).copy_(torch.stack(codes))
            getattr(layer.experts, f"{name}_scale_inv").copy_(torch.stack(scales))
            layer.lora[f"{name}_a"] = nn.Parameter((torch.randn(EXPERTS * RANK, cols, generator=g) * 0.05).bfloat16())
            layer.lora[f"{name}_b"] = nn.Parameter((torch.randn(rows, RANK * EXPERTS, generator=g) * 0.05).bfloat16())
        layer.proj.weight.copy_(torch.randn(DIM, DIM, generator=g) * 0.02)
    for name in ("w1", "w2", "w3"):
        getattr(layer.experts, name).allreduce = False


def _bind_lora(layer: _Layer) -> None:
    layer.experts._ap_lora_ab = {
        n: (layer.lora[f"{n}_a"], layer.lora[f"{n}_b"], RANK, 0.5) for n in ("w1", "w2", "w3")
    }


def _model(seed: int) -> nn.ModuleList:
    layers = []
    for i in range(LAYERS):
        layer = _Layer()
        _fill(layer, seed + i)
        layer = layer.cuda()
        layer.proj.bfloat16()
        _bind_lora(layer)
        layers.append(checkpoint_wrapper(layer, early_stop=False))
    return nn.ModuleList(layers)


def _run(model: nn.ModuleList, x: torch.Tensor):
    h = x
    for layer in model:
        h = layer(h)
    return h


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_offloaded_expert_codes_match_resident_ones():
    from arctic_platform.model.implementations.moe.frozen_offload import install_frozen_expert_offload

    reference, offloaded = _model(0), _model(0)
    offload = install_frozen_expert_offload(offloaded)
    code_bytes = EXPERTS * HIDDEN * DIM * 3  # E4M3 codes of w1, w2, w3 per layer
    x = torch.randn(sum(COUNTS), DIM, device="cuda", dtype=torch.bfloat16)
    for step in range(2):
        for model in (reference, offloaded):
            model.zero_grad(set_to_none=True)
        xr, xo = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        want = _run(reference, xr)
        want.float().square().sum().backward()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        got = _run(offloaded, xo)
        # The checkpointed forward keeps no layer's copied codes alive (at most the prefetched next one is pending).
        held = torch.cuda.memory_allocated() - before
        assert held < 1.5 * code_bytes, f"step {step}: forward holds {held / code_bytes:.2f} layers of expert codes"
        got.float().square().sum().backward()
        assert torch.equal(got, want), step
        assert torch.equal(xo.grad, xr.grad), step
        for ref, off in zip(reference, offloaded):
            ref_inner, off_inner = ref._checkpoint_wrapped_module, off._checkpoint_wrapped_module
            assert torch.equal(off_inner.proj.weight.grad, ref_inner.proj.weight.grad), step
            for name, param in ref_inner.lora.items():
                assert torch.equal(off_inner.lora[name].grad, param.grad), (step, name)
    for layer in offloaded:
        experts = layer._checkpoint_wrapped_module.experts
        assert (
            experts.w1.numel() == experts.w2.numel() == experts.w3.numel() == 0
        ), "codes live on the host between calls"
        assert experts.w1_scale_inv.numel() > 0, "scales stay on the GPU"
    assert offload.host_bytes() == LAYERS * code_bytes
    # Per step: LAYERS forwards, then LAYERS recomputes in reverse. Every call but the first of each pass was
    # prefetched by the call before it, and no prefetch went unused.
    steps, calls = 2, 2 * 2 * LAYERS
    assert offload.copies == calls
    assert offload.prefetched == calls - 2 * steps, f"{offload.prefetched} of {calls} calls prefetched"
