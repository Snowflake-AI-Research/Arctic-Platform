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
"""Frozen block-FP8 routed-expert codes kept on the host and copied in per experts call, under activation
checkpointing (moe/frozen_offload.py); layers that are not recomputed keep theirs. The install checks and
``fp8.stash_frozen`` run on CPU; the rest needs a GPU."""

import gc
import weakref

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


@pytest.fixture(params=[False, True], ids=["w8a16", "w8a8"])
def expert_gemms(request):
    """The experts' forward GEMMs: decoded to BF16, or W8A8 (moe.w8a8_experts, DeepGEMM on Hopper)."""
    from arctic_platform.model.implementations.kernels import fp8_grouped_w8a8

    if request.param and (not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9):
        pytest.skip("W8A8 experts run DeepGEMM's SM90 kernels")
    fp8_grouped_w8a8.configure_from_options(request.param)
    yield request.param
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


def _model(seed: int, lora: tuple = ("w1", "w2", "w3"), plain: tuple = (), device: str = "cuda") -> nn.ModuleList:
    """``LAYERS`` blocks with expert LoRA on ``lora``, each checkpointed except those in ``plain`` (not recomputed)."""
    layers = []
    for i in range(LAYERS):
        layer = _Layer()
        _fill(layer, seed + i)
        layer = layer.to(device)
        layer.proj.bfloat16()
        _bind_lora(layer, lora)
        layers.append(layer if i in plain else checkpoint_wrapper(layer, early_stop=False))
    return nn.ModuleList(layers)


def _run(model: nn.ModuleList, x: torch.Tensor):
    h = x
    for layer in model:
        h = layer(h)
    return h


@requires_cuda
@pytest.mark.parametrize("lora", [("w1", "w2", "w3"), ("w1", "w3")], ids=["all", "gate-up"])
def test_offloaded_expert_codes_match_resident_ones(lora, expert_gemms):
    """Without down-projection LoRA the routing weights fold into the down projection (``scored_swiglu_down``). The
    W8A8 GEMMs read the copied codes in place."""
    from arctic_platform.model.implementations.moe.frozen_offload import install_frozen_expert_offload

    reference, offloaded = _model(0, lora), _model(0, lora)
    assert all(layer._checkpoint_wrapped_module.experts.w8a8 == expert_gemms for layer in offloaded)
    offload = install_frozen_expert_offload(offloaded)
    code_bytes = EXPERTS * HIDDEN * DIM * 3  # E4M3 codes of w1, w2, w3 per layer
    x = torch.randn(sum(COUNTS), DIM, device="cuda", dtype=torch.bfloat16)
    addresses: dict[int, set[int]] = {}

    def record(module, args, i):
        addresses.setdefault(i, set()).add(module.w1.data_ptr())

    for i, layer in enumerate(offloaded):
        # The offload's own pre-hook on the experts runs first (bound at the layer's first call), so this sees the copy.
        layer._checkpoint_wrapped_module.experts.register_forward_pre_hook(lambda m, a, i=i: record(m, a, i))
    trained = {f"{n}_{k}" for n in lora for k in "ab"}
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


def test_offload_leaves_layers_that_are_not_recomputed_on_the_gpu():
    """``ac_config.layers`` leaves blocks out of checkpointing (here the last two): they are not managed."""
    from arctic_platform.model.implementations.moe.frozen_offload import experts_recomputed
    from arctic_platform.model.implementations.moe.frozen_offload import install_frozen_expert_offload

    model = _model(0, plain=(LAYERS - 2, LAYERS - 1), device="cpu")
    assert [experts_recomputed(layer) for layer in model] == [True] * (LAYERS - 2) + [False] * 2
    offload = install_frozen_expert_offload(model)
    assert offload.layers == [layer._checkpoint_wrapped_module for layer in model[: LAYERS - 2]]


@requires_cuda
def test_layers_not_recomputed_keep_their_codes_on_the_gpu(expert_gemms):
    """Their saved tensors hold the codes until backward; the recomputed layers still match resident experts."""
    from arctic_platform.model.implementations.moe.frozen_offload import install_frozen_expert_offload

    plain = (LAYERS - 1,)
    reference, offloaded = _model(0, plain=plain), _model(0, plain=plain)
    assert offloaded[LAYERS - 1].experts.w8a8 == expert_gemms
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


def test_offload_manages_only_layers_with_routed_experts():
    """Dense layers (GLM's first three) have nothing to offload and need not be checkpointed."""
    from arctic_platform.model.implementations.moe.frozen_offload import install_frozen_expert_offload

    moe_layers = list(_model(0, device="cpu"))
    offload = install_frozen_expert_offload(nn.ModuleList([nn.Linear(DIM, DIM), *moe_layers]))
    assert offload.layers == [layer._checkpoint_wrapped_module for layer in moe_layers]
    assert not offload.offloaded, "the codes move at the first call"


class _Block(nn.Module):
    def __init__(self, fp8: bool):
        super().__init__()
        self.experts = GroupedExperts(DIM, HIDDEN, EXPERTS, use_grouped_mm=True, fp8_block_size=128 if fp8 else None)

    def forward(self, x):
        counts = torch.tensor(COUNTS, dtype=torch.int32, device=x.device)
        return x + self.experts(x, counts)


@requires_cuda
@pytest.mark.parametrize("fp8", [True, False], ids=["fp8", "bf16"])
@pytest.mark.parametrize("peft_after_install", [False, True])
def test_offload_under_peft_expert_lora(fp8, peft_after_install):
    """Expert LoRA through PEFT's ParamWrapper. On a block-FP8 base (GLM-5.3's release) the wrapper only carries the
    factors, which the grouped GEMMs add unfused; on a BF16 base it registers W + BA over the weights around each call,
    so the copy must be in place before the wrapper runs. The trainer applies LoRA after the model (and the offload)
    is loaded, so the hooks must find PEFT's wrapper then too."""
    from peft.tuners.lora.layer import ParamWrapper

    from arctic_platform.model.implementations.moe.frozen_offload import install_frozen_expert_offload
    from arctic_platform.model.patches.peft import apply_peft
    from arctic_platform.model.patches.peft import attach_unfused_expert_lora_factors
    from arctic_platform.model.patches.peft import install_unfused_expert_lora_activation

    def build(install=False):
        torch.manual_seed(1)
        blocks = nn.ModuleList([_Block(fp8) for _ in range(LAYERS)])
        if fp8:
            g = torch.Generator().manual_seed(1)
            for block in blocks:
                _fill_fp8_experts(block.experts, g)
            blocks = blocks.cuda()
        else:
            with torch.no_grad():
                for p in blocks.parameters():
                    p.normal_(0, 0.05)
            blocks = blocks.cuda().bfloat16()
        offload = None
        if install:
            for i in range(LAYERS):
                blocks[i] = checkpoint_wrapper(blocks[i], early_stop=False)
            offload = install_frozen_expert_offload(blocks)
        peft = {
            "peft_type": "LORA",
            "r": 4,
            "lora_alpha": 4,
            "target_parameters": ["experts.w1", "experts.w2", "experts.w3"],
        }
        wrapped = apply_peft(nn.Sequential(*blocks), peft)
        if fp8:
            # As the peft patch does for FP8 bases: unfused factors on the experts, no fused delta.
            install_unfused_expert_lora_activation()
            assert attach_unfused_expert_lora_factors(wrapped) == 3 * LAYERS
        with torch.no_grad():
            for name, p in wrapped.named_parameters():
                if ".lora_B" in name:
                    p.normal_(0, 0.05)
        layers = wrapped.base_model.model
        if not install:
            for i in range(LAYERS):
                layers[i] = checkpoint_wrapper(layers[i], early_stop=False)
        return wrapped, layers, offload

    original_activate = ParamWrapper._activate_lora
    try:
        reference, ref_layers, _ = build()
        offloaded, off_layers, offload = build(install=peft_after_install)
        if not peft_after_install:
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
                g is not None and torch.equal(g, outs[1][2][k]) for k, g in outs[0][2].items()
            )
        base = off_layers[0]._checkpoint_wrapped_module.experts
        while hasattr(base, "base_layer"):
            base = base.base_layer
        assert base.w1.numel() == 0, "codes live on the host between calls"
    finally:
        ParamWrapper._activate_lora = original_activate
        for attr in ("_ap_activate_lora_original", "_ap_skip_fused_expert_delta"):
            if hasattr(ParamWrapper, attr):
                delattr(ParamWrapper, attr)


def test_offload_hooks_the_peft_wrapper_on_a_block_fp8_base():
    """With expert LoRA on a block-FP8 base PEFT's ParamWrapper is the module the layer calls and only carries the
    factors (attached to the experts, added unfused): the offload hooks the wrapper and moves the base's codes."""
    from peft.tuners.lora.layer import ParamWrapper

    from arctic_platform.model.implementations.moe.frozen_offload import _expert_call_sites
    from arctic_platform.model.implementations.moe.frozen_offload import _expert_code_params
    from arctic_platform.model.implementations.moe.frozen_offload import install_frozen_expert_offload
    from arctic_platform.model.patches.peft import apply_peft
    from arctic_platform.model.patches.peft import attach_unfused_expert_lora_factors
    from arctic_platform.model.patches.peft import install_unfused_expert_lora_activation

    blocks = nn.ModuleList([checkpoint_wrapper(_Block(fp8=True), early_stop=False) for _ in range(LAYERS)])
    offload = install_frozen_expert_offload(blocks)
    peft = {
        "peft_type": "LORA",
        "r": 4,
        "lora_alpha": 4,
        "target_parameters": ["experts.w1", "experts.w2", "experts.w3"],
    }
    original_activate = ParamWrapper._activate_lora
    try:
        wrapped = apply_peft(nn.Sequential(*blocks), peft)
        install_unfused_expert_lora_activation()
        assert attach_unfused_expert_lora_factors(wrapped) == 3 * LAYERS
    finally:
        ParamWrapper._activate_lora = original_activate
        for attr in ("_ap_activate_lora_original", "_ap_skip_fused_expert_delta"):
            if hasattr(ParamWrapper, attr):
                delattr(ParamWrapper, attr)
    assert offload.layers == [block._checkpoint_wrapped_module for block in blocks]
    for layer in offload.layers:
        (site,) = _expert_call_sites(layer)
        assert isinstance(site, ParamWrapper)
        base = site
        while hasattr(base, "base_layer"):
            base = base.base_layer
        assert base is next(m for m in layer.modules() if isinstance(m, GroupedExperts))
        assert sorted(base._ap_lora_ab) == ["w1", "w2", "w3"]
        codes = [(name, param.dtype) for _, name, param in _expert_code_params(layer)]
        assert codes == [(name, torch.float8_e4m3fn) for name in SHAPES]


class _Ctx:
    pass


def test_stash_frozen_keeps_parameters_on_ctx_and_saves_other_stacks():
    from arctic_platform.model.implementations.fp8 import stash_frozen
    from arctic_platform.model.implementations.fp8 import unstash_frozen

    codes = nn.Parameter(torch.zeros(4, 3, dtype=torch.float8_e4m3fn), requires_grad=False)
    scale = nn.Parameter(torch.ones(4, 1), requires_grad=False)
    ctx = _Ctx()
    assert stash_frozen(ctx, (codes, codes), (scale, scale)) == []
    (weights, scales), rest = unstash_frozen(ctx, (torch.ones(1),))
    assert weights[1] is codes and scales[0] is scale and len(rest) == 1

    # frozen_offload's per-call copies are parameters too, but transient: they go through save_for_backward.
    transient = nn.Parameter(torch.zeros(4, 3, dtype=torch.float8_e4m3fn), requires_grad=False)
    transient._ap_transient = True
    ctx = _Ctx()
    saved = stash_frozen(ctx, (codes, transient), (scale, scale))
    assert [id(t) for t in saved] == [id(codes), id(transient), id(scale), id(scale)]
    (weights, scales), rest = unstash_frozen(ctx, (torch.ones(1), *saved))
    assert weights[1] is transient and scales[1] is scale and len(rest) == 1

    # A BF16 stack has no scale (scored_swiglu_down); the pair comes back in its own structure.
    stack = torch.zeros(4, 3)
    ctx = _Ctx()
    saved = stash_frozen(ctx, stack, None)
    (weight, no_scale), rest = unstash_frozen(ctx, (torch.ones(1), torch.ones(2), *saved))
    assert weight is stack and no_scale is None and len(rest) == 2
    # Nothing but the saved tensors keeps the stacks alive: the rebuild closure holds only their structure.
    ref = weakref.ref(stack)
    del stack, weight, saved, rest
    gc.collect()
    assert ref() is None
