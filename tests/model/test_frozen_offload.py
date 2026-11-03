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
        # Two calls of the experts per layer call, as DeepEP's dispatch chunks make them.
        counts = torch.tensor(COUNTS, dtype=torch.int32, device=x.device)
        scores = torch.linspace(0.1, 1.0, x.shape[0], device=x.device)
        h = self.proj(x)
        return x + self.experts(h, counts, scores) + 0.5 * self.experts(h.flip(0), counts, scores)


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


def _model(seed: int, plain: tuple = ()) -> nn.ModuleList:
    """``LAYERS`` blocks, each checkpointed except those in ``plain`` (not recomputed)."""
    layers = []
    for i in range(LAYERS):
        layer = _Layer()
        _fill(layer, seed + i)
        layer = layer.cuda()
        layer.proj.bfloat16()
        _bind_lora(layer)
        layers.append(layer if i in plain else checkpoint_wrapper(layer, early_stop=False))
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
    addresses: dict[int, set[int]] = {}

    def record(module, args, i):
        addresses.setdefault(i, set()).add(module.w1.data_ptr())

    for i, layer in enumerate(offloaded):
        # Registered after the offload's own pre-hook on the experts, so it sees the installed copy.
        layer._checkpoint_wrapped_module.experts.register_forward_pre_hook(lambda m, a, i=i: record(m, a, i))
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
            for name, param in ref_inner.lora.items():
                assert torch.equal(off_inner.lora[name].grad, param.grad), (step, name)
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_offload_refuses_to_overwrite_a_buffer_in_use():
    from arctic_platform.model.implementations.moe.frozen_offload import install_frozen_expert_offload

    model = _model(0)
    install_frozen_expert_offload(model)
    x = torch.randn(sum(COUNTS), DIM, device="cuda", dtype=torch.bfloat16)
    _run(model, x.clone().requires_grad_(True)).float().sum().backward()
    # Layer 0's copy held by a live graph (no checkpoint: its forward saves the codes) while layer 2 wants its buffer.
    inner = [layer._checkpoint_wrapped_module for layer in model]
    xs = x.clone().requires_grad_(True)
    out = inner[0](xs)
    with pytest.raises(RuntimeError, match="still in use"):
        inner[2](out)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_layers_not_recomputed_keep_their_codes_on_the_gpu():
    """``ac_config.layers`` leaves blocks out of checkpointing: their saved tensors hold the codes until backward."""
    from arctic_platform.model.implementations.moe.frozen_offload import install_frozen_expert_offload

    plain = (LAYERS - 1,)
    reference, offloaded = _model(0, plain), _model(0, plain)
    offload = install_frozen_expert_offload(offloaded)
    assert len(offload.layers) == LAYERS - 1
    x = torch.randn(sum(COUNTS), DIM, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        for model in (reference, offloaded):
            model.zero_grad(set_to_none=True)
        xr, xo = x.clone().requires_grad_(True), x.clone().requires_grad_(True)
        want = _run(reference, xr)
        want.float().square().sum().backward()
        got = _run(offloaded, xo)
        got.float().square().sum().backward()
        assert torch.equal(got, want) and torch.equal(xo.grad, xr.grad)
    assert offloaded[LAYERS - 1].experts.w1.numel() > 0
    assert offloaded[0]._checkpoint_wrapped_module.experts.w1.numel() == 0


class _Bf16Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = GroupedExperts(DIM, HIDDEN, EXPERTS, use_grouped_mm=True)

    def forward(self, x):
        counts = torch.tensor(COUNTS, dtype=torch.int32, device=x.device)
        return x + self.experts(x, counts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_offload_under_peft_fused_expert_lora():
    """A BF16 base with expert LoRA: PEFT's ParamWrapper registers W + BA over the weights around each call, so the
    copy must be in place before the wrapper runs."""
    from arctic_platform.model.implementations.moe.frozen_offload import install_frozen_expert_offload
    from arctic_platform.model.patches.peft import apply_peft

    def build():
        torch.manual_seed(1)
        blocks = nn.ModuleList([_Bf16Block() for _ in range(LAYERS)])
        with torch.no_grad():
            for p in blocks.parameters():
                p.normal_(0, 0.05)
        blocks = blocks.cuda().bfloat16()
        peft = {
            "peft_type": "LORA",
            "r": 4,
            "lora_alpha": 4,
            "target_parameters": ["experts.w1", "experts.w2", "experts.w3"],
        }
        wrapped = apply_peft(nn.Sequential(*blocks), peft)
        with torch.no_grad():
            for name, p in wrapped.named_parameters():
                if ".lora_B" in name:
                    p.normal_(0, 0.05)
        layers = wrapped.base_model.model
        for i in range(LAYERS):
            layers[i] = checkpoint_wrapper(layers[i], early_stop=False)
        return wrapped, layers

    reference, ref_layers = build()
    offloaded, off_layers = build()
    offload = install_frozen_expert_offload(nn.ModuleList(list(off_layers)))
    assert len(offload.layers) == LAYERS
    x = torch.randn(sum(COUNTS), DIM, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        outs = []
        for model in (reference, offloaded):
            model.zero_grad(set_to_none=True)
            xi = x.clone().requires_grad_(True)
            out = model(xi)
            out.float().square().sum().backward()
            outs.append((out, xi.grad, {n: p.grad for n, p in model.named_parameters() if ".lora_" in n}))
        assert torch.equal(outs[0][0], outs[1][0]) and torch.equal(outs[0][1], outs[1][1])
        assert outs[0][2].keys() == outs[1][2].keys() and all(
            torch.equal(outs[0][2][k], outs[1][2][k]) for k in outs[0][2]
        )
    base = off_layers[0]._checkpoint_wrapped_module.experts
    while hasattr(base, "base_layer"):
        base = base.base_layer
    assert base.w1.numel() == 0, "codes live on the host between calls"
