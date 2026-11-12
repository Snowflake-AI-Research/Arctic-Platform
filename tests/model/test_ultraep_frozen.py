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
"""UltraEP spares for frozen block-FP8 experts with LoRA (moe/ultraep_frozen.py): copies compute with their owners'
codes, block scales and LoRA factors, and the copies' LoRA gradients return to the owners.

Multi-GPU, run directly:  torchrun --nproc-per-node 8 tests/model/test_ultraep_frozen.py [offload] [w8a8]

With ``offload``, three activation-checkpointed layers run under the frozen expert offload (moe/frozen_offload.py):
their codes exist only in its two persistent buffers by layer parity, which UltraEP's pointer pools read in place,
through the forward and the reverse recompute; they run first, so the group's UltraEP state is built after the offload
took the codes to the host (when the first layer ran), from the stacks' shapes. With ``w8a8`` the experts' forward
GEMMs run W8A8 (moe.w8a8_experts) on the native + spare stacks, and the reference emulates its activation quantization
(and the BF16 roundings ahead of each one), straight through. Under pytest (CPU) only the LoRA staging round trip runs.
"""

import os
import sys

import torch
import torch.distributed as dist
from torch import nn

L, S, DIM, HIDDEN, RANK, TOKENS, TOPK, MAX_ROWS = 4, 2, 512, 256, 8, 2048, 4, 8
SHAPES = {"w1": (HIDDEN, DIM), "w2": (DIM, HIDDEN), "w3": (HIDDEN, DIM)}
_W8A8 = False  # set by main from argv


def _w8a8_quant(t):
    """The W8A8 forward's 1 x 128 E4M3 activation quantization (FP32 emulation), straight through."""
    groups = t.reshape(t.shape[0], t.shape[1] // 128, 128)
    scale = groups.detach().abs().amax(-1, keepdim=True).clamp(min=1e-10) / 448.0
    q = ((groups.detach() / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).reshape(t.shape)
    return t + (q - t).detach()


def _bf16(t):
    return t + (t.bfloat16().float() - t).detach()


def test_lora_staging_round_trip():
    """``_stage`` lays out each expert's ``A_e``, then its ``B_e`` (PEFT keeps ``B`` as ``[out, r * E]``, the expert
    index fastest)."""
    from arctic_platform.model.implementations.moe.ultraep_frozen import _stage
    from arctic_platform.model.implementations.moe.ultraep_frozen import _unstage

    experts, rank, width_in, width_out = 3, 2, 5, 7
    a, b = torch.randn(experts * rank, width_in), torch.randn(width_out, rank * experts)
    staged = _stage(a, b, rank, experts)
    assert staged.shape == (experts, rank * (width_in + width_out))
    for e in range(experts):
        assert torch.equal(staged[e, : rank * width_in], a.view(experts, rank, width_in)[e].flatten())
        assert torch.equal(staged[e, rank * width_in :], b.view(width_out, rank, experts)[:, :, e].flatten())
    a2, b2 = _unstage(staged, rank, width_in, width_out)
    assert torch.equal(a2, a) and torch.equal(b2, b)


def _gather(t):
    out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t.contiguous())
    return torch.stack(out)


def _experts(seed: int, rank: int):
    """This rank's ``L`` block-FP8 experts with LoRA on every projection, and every rank's logical references."""
    from arctic_platform.model.implementations.fp8 import dequantize_from_fp8_blockwise
    from arctic_platform.model.implementations.fp8 import quantize_to_fp8_blockwise
    from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts

    torch.manual_seed(seed + rank)
    experts = GroupedExperts(DIM, HIDDEN, L, use_grouped_mm=True, fp8_block_size=128).cuda()
    assert experts.w8a8 == _W8A8
    for name, (rows, cols) in SHAPES.items():
        q, sc = zip(
            *(quantize_to_fp8_blockwise((torch.randn(rows, cols, device="cuda") * 0.05).bfloat16()) for _ in range(L))
        )
        getattr(experts, name).data.copy_(torch.stack(q))
        getattr(experts, f"{name}_scale_inv").data.copy_(torch.stack(sc))
        getattr(experts, name).allreduce = False
    ab = {}
    for name, (out_f, in_f) in SHAPES.items():
        a = nn.Parameter((torch.randn(L * RANK, in_f, device="cuda") * 0.05).bfloat16())
        b = nn.Parameter((torch.randn(out_f, RANK * L, device="cuda") * 0.05).bfloat16())
        ab[name] = (a, b, RANK, 0.5)
    experts._ap_lora_ab = ab

    def decode(name):
        weight, scale = getattr(experts, name).detach(), getattr(experts, f"{name}_scale_inv").detach()
        return torch.stack([dequantize_from_fp8_blockwise(weight[e], scale[e], 128).float() for e in range(L)])

    # Logical reference weights of every expert: [world * L, out, in] FP32, and every rank's factors.
    ref_w = {n: _gather(decode(n)).flatten(0, 1) for n in SHAPES}
    ref_ab = {n: (_gather(ab[n][0].detach()), _gather(ab[n][1].detach())) for n in SHAPES}
    return experts, ab, ref_w, ref_ab


def _skewed_routing(seed: int, rank: int, world: int) -> torch.Tensor:
    """Experts 0 and 1 hot on every rank."""
    g = torch.Generator(device="cuda").manual_seed(seed + rank)
    return torch.cat(
        [
            torch.randint(0, 2, (TOKENS, 1), device="cuda", generator=g),
            torch.randint(2, L * world, (TOKENS, TOPK - 1), device="cuda", generator=g),
        ],
        1,
    )


def _slot_rows(plan, rank: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """This rank's slots' logical experts and the rows each receives (as after dispatch): a few per used slot."""
    p2l = plan.state.gate_up.physical_to_logical_map[plan.lid].long()
    slot_logical = p2l[rank * (L + S) : (rank + 1) * (L + S)]
    g = torch.Generator(device="cuda").manual_seed(seed + rank)
    # Unused spare slots (logical -1) receive no tokens from dispatch.
    counts = torch.randint(0, MAX_ROWS + 1, (L + S,), device="cuda", generator=g) * (slot_logical >= 0)
    return slot_logical, counts


def _reference(x, scores, counts, slot_logical, ref_w, full_ab):
    """FP32 experts with LoRA for rows packed by slot, each slot computing its logical expert. In W8A8 the forward
    GEMMs read quantized activations, and the tensors are rounded to BF16 where the kernels round them (the SwiGLU
    quantization turns those roundings into whole FP8 steps)."""
    q, r = (_w8a8_quant, _bf16) if _W8A8 else ((lambda t: t), (lambda t: t))

    def lora(n, inputs, lid):
        owner, local = divmod(lid, L)
        a = full_ab[n][0][owner].float().view(L, RANK, -1)[local]
        b = full_ab[n][1][owner].float().view(-1, RANK, L)[:, :, local]
        return r(r(inputs @ a.T) * 0.5 @ b.T)

    pieces, start = [], 0
    for j, c in enumerate(counts.tolist()):
        if c:
            xs, lid = x[start : start + c], int(slot_logical[j])
            gate = r(r(q(xs) @ ref_w["w1"][lid].T) + lora("w1", xs, lid))
            up = r(r(q(xs) @ ref_w["w3"][lid].T) + lora("w3", xs, lid))
            h = r(torch.nn.functional.silu(gate) * up)
            pieces.append((r(q(h) @ ref_w["w2"][lid].T) + lora("w2", h, lid)) * scores[start : start + c, None])
        start += c
    return torch.cat(pieces) if pieces else x.new_zeros(0, DIM)


def _check_lora_grads(rank, ab, full_ab, what):
    for n in SHAPES:
        for k in (0, 1):
            dist.all_reduce(full_ab[n][k].grad)  # every rank's rows' contribution to every expert's factors
            expected = full_ab[n][k].grad[rank]
            got = ab[n][k].grad.float()
            err = ((got - expected).norm() / expected.norm().clamp_min(1e-6)).item()
            assert err < 2e-2, f"rank {rank}, {what}: {n} {'AB'[k]} gradient error {err}"


def _relative_error(got, want) -> float:
    return ((got.float() - want).norm() / want.norm().clamp_min(1e-6)).item()


def _single_layer(rank, world, group, owner: nn.Module):
    """One experts call on a planned placement, as the MoE makes it after dispatch. ``owner`` stands for the MoE
    layer: UltraEP indexes its layers by module, so it must outlive the test."""
    from arctic_platform.model.implementations.moe import ultraep_frozen

    experts, ab, ref_w, ref_ab = _experts(100, rank)
    owner.experts = experts  # the pools point at these codes
    selected = _skewed_routing(7, rank, world)
    plan = ultraep_frozen.frozen_ultraep_plan(owner, experts, group, selected, ab, building=True)
    physical = plan.remap(selected)
    p2l = plan.state.gate_up.physical_to_logical_map[plan.lid].long()
    assert torch.equal(p2l[physical], selected), "remap keeps every token on a copy of its expert"
    loads = torch.bincount(physical.flatten(), minlength=plan.num_slots).view(world, L + S).sum(1)
    logical = torch.bincount(selected.flatten(), minlength=L * world).view(world, L).sum(1)
    dist.all_reduce(loads)
    dist.all_reduce(logical)

    slot_logical, counts = _slot_rows(plan, rank, 11)
    rows = int(counts.sum())
    g = torch.Generator(device="cuda").manual_seed(13 + rank)
    x = torch.randn(rows, DIM, device="cuda", generator=g).bfloat16().requires_grad_(True)
    scores = torch.rand(rows, device="cuda", generator=g)
    experts._arl_ultraep_call = plan
    out = experts(x, counts, scores)
    experts._arl_ultraep_call = None

    full_ab = {n: tuple(t.clone().requires_grad_(True) for t in ref_ab[n]) for n in SHAPES}
    want = _reference(x.detach().float(), scores, counts, slot_logical, ref_w, full_ab)
    error = _relative_error(out, want)
    assert error < 2e-2, f"rank {rank}: output error {error}"
    upstream = torch.randn(rows, DIM, device="cuda", generator=g)
    out.float().backward(upstream)
    want.backward(upstream)
    _check_lora_grads(rank, ab, full_ab, "one call")
    return (logical.max() / logical.float().mean()).item(), (loads.max() / loads.float().mean()).item()


class _Block(nn.Module):
    """One MoE layer's routed experts as the MoE runs them: the call's plan (reused by the recompute), then the experts
    on this rank's rows."""

    def __init__(self, experts, selected, group, seed):
        super().__init__()
        self.experts = experts
        self.selected, self.group, self.seed = selected, group, seed
        self.slot_logical: torch.Tensor | None = None
        self.counts: torch.Tensor | None = None

    def forward(self, x, scores):
        from arctic_platform.model.implementations.gpu.router_replay_recompute import _in_recompute
        from arctic_platform.model.implementations.gpu.router_replay_recompute import _install_recompute_detector
        from arctic_platform.model.implementations.moe import ultraep_frozen

        _install_recompute_detector()
        experts = self.experts
        plan = ultraep_frozen.frozen_ultraep_plan(
            self, experts, self.group, self.selected, experts._ap_lora_ab, building=not _in_recompute()
        )
        plan.remap(self.selected)
        if self.counts is None:
            self.slot_logical, self.counts = _slot_rows(plan, dist.get_rank(), self.seed)
        experts._arl_ultraep_call = plan
        out = experts(x, self.counts, scores)
        experts._arl_ultraep_call = None
        return out


def _offloaded_layers(rank, world, group, layers=3):
    """Checkpointed layers whose codes live in the frozen offload's two buffers: layers 0 and 2 share one."""
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

    from arctic_platform.model.implementations.moe.frozen_offload import install_frozen_expert_offload

    built = [_experts(200 + 10 * i, rank) for i in range(layers)]
    blocks = [_Block(e, _skewed_routing(17 + i, rank, world), group, 21 + i) for i, (e, *_) in enumerate(built)]
    model = nn.ModuleList([checkpoint_wrapper(b, early_stop=False) for b in blocks])
    offload = install_frozen_expert_offload(model, root=model)
    g = torch.Generator(device="cuda").manual_seed(31 + rank)
    rows = (L + S) * MAX_ROWS  # rows past a layer's last slot are padding and come out zero
    xs = [torch.randn(rows, DIM, device="cuda", generator=g).bfloat16().requires_grad_(True) for _ in blocks]
    scores = [torch.rand(rows, device="cuda", generator=g) for _ in blocks]
    outs = [layer(x, s) for layer, x, s in zip(model, xs, scores)]
    upstream = [torch.randn(rows, DIM, device="cuda", generator=g) for _ in blocks]
    # One backward: the recomputes run in reverse layer order, as the offload's prefetch expects.
    sum((out.float() * up).sum() for out, up in zip(outs, upstream)).backward()
    assert all(b.experts.w1.numel() == 0 for b in blocks), "codes live on the host between calls"
    for i, (block, (experts, ab, ref_w, ref_ab)) in enumerate(zip(blocks, built)):
        used = int(block.counts.sum())
        full_ab = {n: tuple(t.clone().requires_grad_(True) for t in ref_ab[n]) for n in SHAPES}
        x_ref = xs[i].detach().float()[:used]
        want = _reference(x_ref, scores[i][:used], block.counts, block.slot_logical, ref_w, full_ab)
        error = _relative_error(outs[i][:used], want)
        assert error < 2e-2, f"rank {rank}, layer {i}: output error {error}"
        assert not outs[i][used:].any(), f"rank {rank}, layer {i}: padding rows"
        want.backward(upstream[i][:used])
        _check_lora_grads(rank, ab, full_ab, f"layer {i}")
    return offload


def main():
    global _W8A8
    offload = "offload" in sys.argv[1:]
    _W8A8 = "w8a8" in sys.argv[1:]
    os.environ["ARL_ULTRAEP_SPARES"] = str(S)
    if _W8A8:
        from arctic_platform.model.implementations.kernels.fp8_grouped_w8a8 import configure_from_options

        configure_from_options(True)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    group = dist.new_group(list(range(world)))
    owner = nn.Module()
    if offload:
        # First, as in a job: the UltraEP state is built at the first MoE call, after the offload took the codes.
        from arctic_platform.model.implementations.moe import ultraep_frozen

        assert id(group) not in ultraep_frozen._STATES
        state = _offloaded_layers(rank, world, group)
        assert state.offloaded and state.copies >= len(state.layers), state.copies
    before, after = _single_layer(rank, world, group, owner)
    if rank == 0:
        print(
            f"ok (fp8{' + w8a8' if _W8A8 else ''}{' + offload' if offload else ''}): EP{world}, {S} spares: copies"
            " compute with their owners'"
            f" weights and LoRA, LoRA gradients return; rank load max/mean {before:.3f} -> {after:.3f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
