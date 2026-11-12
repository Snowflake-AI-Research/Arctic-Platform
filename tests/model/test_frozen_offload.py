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
"""Frozen block-FP8 routed-expert codes kept on the host and copied in per experts call (moe/frozen_offload.py)."""

import pytest
import torch
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from arctic_platform.model.implementations.fp8 import quantize_to_fp8_blockwise
from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts

DIM, HIDDEN, EXPERTS, RANK, LAYERS = 1024, 512, 8, 8, 4
COUNTS = [3, 0, 17, 9, 1, 30, 4, 0]
SHAPES = {"w1": (HIDDEN, DIM), "w2": (DIM, HIDDEN), "w3": (HIDDEN, DIM)}
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture
def w8a8_experts():
    """GLM-5.3's experts' forward GEMMs: W8A8 (moe.w8a8_experts, DeepGEMM on Hopper)."""
    from arctic_platform.model.implementations.kernels import fp8_grouped_w8a8

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("W8A8 experts run DeepGEMM's SM90 kernels")
    fp8_grouped_w8a8.configure_from_options(True)
    yield
    fp8_grouped_w8a8.configure_from_options(None)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(DIM, DIM, bias=False)
        self.experts = GroupedExperts(DIM, HIDDEN, EXPERTS, use_grouped_mm=True, fp8_block_size=128)
        self.lora = nn.ParameterDict()

    def forward(self, x):
        # Two calls of the experts per layer call, as DeepEP's dispatch chunks make them.
        counts = torch.tensor(COUNTS, dtype=torch.int32, device=x.device)
        scores = torch.linspace(0.1, 1.0, x.shape[0], device=x.device)
        h = self.proj(x)
        return x + self.experts(h, counts, scores) + 0.5 * self.experts(h.flip(0), counts, scores)


def _fill_fp8_experts(experts: GroupedExperts, g: torch.Generator) -> None:
    with torch.no_grad():
        for name, (rows, cols) in SHAPES.items():
            dense = torch.randn(EXPERTS, rows, cols, generator=g) * 0.05
            codes, scales = zip(*(quantize_to_fp8_blockwise(dense[e].bfloat16()) for e in range(EXPERTS)))
            getattr(experts, name).copy_(torch.stack(codes))
            getattr(experts, f"{name}_scale_inv").copy_(torch.stack(scales))


def _fill(layer: _Layer, seed: int) -> None:
    g = torch.Generator().manual_seed(seed)
    _fill_fp8_experts(layer.experts, g)
    with torch.no_grad():
        for name, (rows, cols) in SHAPES.items():
            layer.lora[f"{name}_a"] = nn.Parameter((torch.randn(EXPERTS * RANK, cols, generator=g) * 0.05).bfloat16())
            layer.lora[f"{name}_b"] = nn.Parameter((torch.randn(rows, RANK * EXPERTS, generator=g) * 0.05).bfloat16())
        layer.proj.weight.copy_(torch.randn(DIM, DIM, generator=g) * 0.02)
    for name in SHAPES:
        getattr(layer.experts, name).allreduce = False


def _bind_lora(layer: _Layer, names: tuple) -> None:
    layer.experts._ap_lora_ab = {n: (layer.lora[f"{n}_a"], layer.lora[f"{n}_b"], RANK, 0.5) for n in names}


def _model(seed: int) -> nn.ModuleList:
    """``LAYERS`` checkpointed blocks with expert LoRA on w1, w2 and w3."""
    layers = []
    for i in range(LAYERS):
        layer = _Layer()
        _fill(layer, seed + i)
        layer = layer.to("cuda")
        layer.proj.bfloat16()
        _bind_lora(layer, tuple(SHAPES))
        layers.append(checkpoint_wrapper(layer, early_stop=False))
    return nn.ModuleList(layers)


def _run(model: nn.ModuleList, x: torch.Tensor):
    h = x
    for layer in model:
        h = layer(h)
    return h


@requires_cuda
def test_offloaded_expert_codes_match_resident_ones(w8a8_experts):
    """The W8A8 GEMMs read the copied codes in place."""
    from arctic_platform.model.implementations.moe.frozen_offload import install_frozen_expert_offload

    reference, offloaded = _model(0), _model(0)
    assert all(layer._checkpoint_wrapped_module.experts.w8a8 for layer in offloaded)
    offload = install_frozen_expert_offload(offloaded, root=offloaded)
    code_bytes = EXPERTS * HIDDEN * DIM * 3  # E4M3 codes of w1, w2, w3 per layer
    x = torch.randn(sum(COUNTS), DIM, device="cuda", dtype=torch.bfloat16)
    addresses: dict[int, set[int]] = {}

    def record(module, args, i):
        addresses.setdefault(i, set()).add(module.w1.data_ptr())

    for i, layer in enumerate(offloaded):
        # The offload's own pre-hook on the experts runs first (bound at the layer's first call), so this sees the copy.
        layer._checkpoint_wrapped_module.experts.register_forward_pre_hook(lambda m, a, i=i: record(m, a, i))
    trained = {f"{n}_{k}" for n in SHAPES for k in "ab"}
    steps = 3
    for step in range(steps):
        for model in (reference, offloaded):
            model.zero_grad(set_to_none=True)
        xr, xo = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        want = _run(reference, xr)
        want.float().square().sum().backward()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        got = _run(offloaded, xo)
        # The checkpointed forward keeps no copied codes alive: they sit in two persistent buffers (allocated in step 0).
        held = torch.cuda.memory_allocated() - before
        if step:
            assert held < 0.5 * code_bytes, f"step {step}: forward allocates {held / code_bytes:.2f} layers of codes"
        got.float().square().sum().backward()
        assert torch.equal(got, want), step
        assert torch.equal(xo.grad, xr.grad), step
        for ref, off in zip(reference, offloaded):
            ref_inner, off_inner = ref._checkpoint_wrapped_module, off._checkpoint_wrapped_module
            assert torch.equal(off_inner.proj.weight.grad, ref_inner.proj.weight.grad), step
            for name in trained:
                assert torch.equal(off_inner.lora[name].grad, ref_inner.lora[name].grad), (step, name)
    for layer in offloaded:
        experts = layer._checkpoint_wrapped_module.experts
        assert (
            experts.w1.numel() == experts.w2.numel() == experts.w3.numel() == 0
        ), "codes live on the host between calls"
        assert experts.w1_scale_inv.numel() > 0, "scales stay on the GPU"
    assert offload.host_bytes() == LAYERS * code_bytes
    # Per step: LAYERS forwards (two experts calls each), then LAYERS recomputes in reverse. Every first call of a pass
    # but training's very first finds its codes prefetched or still resident; a later step copies 2 * LAYERS - 4 times
    # (the turnaround layers and the two the next pass starts with are still in their buffers).
    assert offload.prefetched == steps * 2 * LAYERS - 1
    assert offload.copies == (2 * LAYERS - 2) + (steps - 1) * (2 * LAYERS - 4), offload.copies
    # Every call of a layer finds its codes at the same address: buffer i % 2 of the two persistent buffers.
    assert all(len(ptrs) == 1 for ptrs in addresses.values()), addresses
    assert addresses[0] != addresses[1] and addresses[0] == addresses[2]


@requires_cuda
def test_the_codes_leave_the_gpu_when_the_first_layer_runs():
    """After a restore that wrote the codes on the GPU (GLM-5.3's restored LoRA job ran out of memory in layer 0)."""
    from arctic_platform.model.implementations.moe.frozen_held import whole_frozen_weights
    from arctic_platform.model.implementations.moe.frozen_offload import install_frozen_expert_offload
    from arctic_platform.model.implementations.moe.frozen_offload import offloaded_code_stack

    restored, saved = _model(0), _model(1)
    offload = install_frozen_expert_offload(restored, root=restored)
    experts = [layer._checkpoint_wrapped_module.experts for layer in restored]
    with whole_frozen_weights(restored):
        restored.load_state_dict(saved.state_dict())
    assert not offload.offloaded and offloaded_code_stack(experts[0], "w2") is None
    x = torch.randn(sum(COUNTS), DIM, device="cuda", dtype=torch.bfloat16)
    xr, xs = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
    code_bytes = EXPERTS * HIDDEN * DIM * 3
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    seen = []

    def probe(module, args):  # layer 0's own computation, ahead of its experts
        seen.append(
            (before - torch.cuda.memory_allocated(), [e.w1.numel() + e.w2.numel() + e.w3.numel() for e in experts])
        )

    restored[0]._checkpoint_wrapped_module.proj.register_forward_pre_hook(probe)
    got, want = _run(restored, xr), _run(saved, xs)
    freed, sizes = seen[0]
    # Every layer's codes left the GPU; the offload's two persistent buffers hold two layers' worth.
    assert freed >= (LAYERS - 2) * code_bytes - 4096 and sizes == [0] * LAYERS, (freed / code_bytes, sizes)
    like, w2 = offloaded_code_stack(experts[0], "w2"), saved[0]._checkpoint_wrapped_module.experts.w2
    assert like.device.type == "meta" and like.shape == w2.shape and like.dtype == w2.dtype
    got.float().square().sum().backward()
    want.float().square().sum().backward()
    assert torch.equal(got, want) and torch.equal(xr.grad, xs.grad)
