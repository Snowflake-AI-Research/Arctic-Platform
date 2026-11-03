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
"""UltraEP spares for packed-4-bit experts with LoRA (moe/ultraep_packed4.py): copies compute with their owners'
codes, scales and LoRA factors, and the copies' LoRA gradients return to the owners.

Multi-GPU, run directly:  torchrun --nproc-per-node 8 tests/model/test_ultraep_packed4.py [mxfp4]
"""

import os
import sys

import torch
import torch.distributed as dist

L, S, DIM, HIDDEN, RANK, TOKENS, TOPK = 4, 2, 512, 256, 8, 2048, 4


def _gather(t):
    out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t.contiguous())
    return torch.stack(out)


def main():
    mxfp4 = "mxfp4" in sys.argv[1:]
    os.environ["ARL_ULTRAEP_SPARES"] = str(S)
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    from arctic_platform.model.implementations.kernels.packed4_dequant import dequantize_int4_groups_reference
    from arctic_platform.model.implementations.kernels.packed4_dequant import dequantize_mxfp4_reference
    from arctic_platform.model.implementations.moe import ultraep_packed4
    from arctic_platform.model.implementations.moe.layers.moe import GroupedExperts

    group = dist.new_group(list(range(world)))
    torch.manual_seed(100 + rank)
    experts = GroupedExperts(
        DIM, HIDDEN, L, use_grouped_mm=True, fp8_block_size=128, packed4="mxfp4" if mxfp4 else "int4"
    ).cuda()
    shapes = {"w1": (HIDDEN, DIM), "w2": (DIM, HIDDEN), "w3": (HIDDEN, DIM)}
    for name, (rows, cols) in shapes.items():
        codes = torch.randint(-7, 8, (L, rows, cols), dtype=torch.int16, device="cuda")
        getattr(experts, name).data.copy_(
            ((codes[..., 0::2] & 15) | ((codes[..., 1::2] & 15) << 4)).to(torch.uint8).view(torch.int8)
        )
        if mxfp4:  # FP32 powers of two, compacted to E8M0 bytes as after loading
            exponent = torch.randint(118, 123, (L, rows, cols // 32), dtype=torch.uint8, device="cuda")
            getattr(experts, f"{name}_scale_inv").data.copy_(exponent.view(torch.float8_e8m0fnu).float())
        else:
            getattr(experts, f"{name}_scale_inv").data.copy_(
                (torch.rand(L, rows, cols // 128, device="cuda") * 0.01 + 1e-3).bfloat16()
            )
    if mxfp4:
        experts.compact_mxfp4_scales()
        assert experts.w1_scale_inv.dtype == torch.uint8
    ab = {}
    for name, (out_f, in_f) in shapes.items():
        a = torch.nn.Parameter((torch.randn(L * RANK, in_f, device="cuda") * 0.05).bfloat16())
        b = torch.nn.Parameter((torch.randn(out_f, RANK * L, device="cuda") * 0.05).bfloat16())
        ab[name] = (a, b, RANK, 0.5)
    experts._ap_lora_ab = ab
    # Logical reference weights of every expert: [world * L, out, in] FP32, and every rank's factors.
    raw_scales = {n: getattr(experts, f"{n}_scale_inv").detach().clone() for n in shapes}
    decode = dequantize_mxfp4_reference if mxfp4 else dequantize_int4_groups_reference
    ref_w = {n: _gather(decode(getattr(experts, n).detach(), raw_scales[n])).flatten(0, 1) for n in shapes}
    ref_ab = {n: (_gather(ab[n][0].detach()), _gather(ab[n][1].detach())) for n in shapes}

    # Skewed routing: experts 0 and 1 hot on every rank.
    E = L * world
    g = torch.Generator(device="cuda").manual_seed(7 + rank)
    selected = torch.cat(
        [
            torch.randint(0, 2, (TOKENS, 1), device="cuda", generator=g),
            torch.randint(2, E, (TOKENS, TOPK - 1), device="cuda", generator=g),
        ],
        1,
    )
    module = torch.nn.Module()
    plan = ultraep_packed4.packed4_ultraep_plan(module, experts, group, selected, ab, building=True)
    physical = plan.remap(selected)
    p2l = plan.state.gate_up.physical_to_logical_map[plan.lid].long()
    assert torch.equal(p2l[physical], selected), "remap keeps every token on a copy of its expert"
    loads = torch.bincount(physical.flatten(), minlength=plan.num_slots).view(world, L + S).sum(1)
    logical = torch.bincount(selected.flatten(), minlength=E).view(world, L).sum(1)
    dist.all_reduce(loads)
    dist.all_reduce(logical)

    # This rank's slots receive a few rows each (as after dispatch): rows packed by slot.
    slot_logical = p2l[rank * (L + S) : (rank + 1) * (L + S)]
    # Unused spare slots (logical -1) receive no tokens from dispatch.
    counts = torch.randint(0, 9, (L + S,), device="cuda", generator=g) * (slot_logical >= 0)
    rows = int(counts.sum())
    x = (torch.randn(rows, DIM, device="cuda", generator=g)).bfloat16().requires_grad_(True)
    scores = torch.rand(rows, device="cuda", generator=g)
    experts._arl_ultraep_call = plan
    out = experts(x, counts, scores)
    experts._arl_ultraep_call = None

    x_ref = x.detach().float().requires_grad_(True)
    full_ab = {
        n: (ref_ab[n][0].clone().requires_grad_(True), ref_ab[n][1].clone().requires_grad_(True)) for n in shapes
    }

    def lora(n, inputs, lid):
        owner, local = divmod(lid, L)
        a = full_ab[n][0][owner].float().view(L, RANK, -1)[local]
        b = full_ab[n][1][owner].float().view(-1, RANK, L)[:, :, local]
        return (inputs @ a.T) @ b.T * 0.5

    pieces, start = [], 0
    for j, c in enumerate(counts.tolist()):
        xs = x_ref[start : start + c]
        lid = int(slot_logical[j]) if c else 0
        if c:
            gate = xs @ ref_w["w1"][lid].T + lora("w1", xs, lid)
            up = xs @ ref_w["w3"][lid].T + lora("w3", xs, lid)
            h = torch.nn.functional.silu(gate) * up
            pieces.append((h @ ref_w["w2"][lid].T + lora("w2", h, lid)) * scores[start : start + c, None])
        start += c
    want = torch.cat(pieces) if pieces else x_ref.new_zeros(0, DIM)
    error = ((out.float() - want).norm() / want.norm().clamp_min(1e-6)).item()
    assert error < 2e-2, f"rank {rank}: output error {error}"

    upstream = torch.randn(rows, DIM, device="cuda", generator=g)
    out.float().backward(upstream)
    want.backward(upstream)
    for n in shapes:
        for k in (0, 1):
            dist.all_reduce(full_ab[n][k].grad)  # every rank's rows' contribution to every expert's factors
            expected = full_ab[n][k].grad[rank]
            got = ab[n][k].grad.float()
            err = ((got - expected).norm() / expected.norm().clamp_min(1e-6)).item()
            assert err < 2e-2, f"rank {rank}: {n} {'AB'[k]} gradient error {err}"
    if rank == 0:
        print(
            f"ok ({'mxfp4' if mxfp4 else 'int4'}): EP{world}, {S} spares: copies compute with their owners' weights"
            " and LoRA, LoRA gradients return; rank load max/mean"
            f" {(logical.max() / logical.float().mean()).item():.3f} ->"
            f" {(loads.max() / loads.float().mean()).item():.3f}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
