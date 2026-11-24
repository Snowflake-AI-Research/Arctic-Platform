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
"""UltraEP spares for trainable BF16 experts (moe/ultraep_replicas.py) hold their owners' weights and return their
gradients to them, with the real UltraEP over 8 GPUs: this file under torchrun, which the test launches."""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from arctic_platform.model.implementations.moe import ultraep_replicas

L, S, FFN, DIM, TOKENS, TOPK = 4, 2, 64, 128, 4096, 4


class _Experts(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        g = torch.Generator(device="cuda").manual_seed(100 + rank)
        bf16 = dict(device="cuda", dtype=torch.bfloat16)
        self.w1 = torch.nn.Parameter(torch.randn(L, FFN, DIM, generator=g, **bf16))
        self.w2 = torch.nn.Parameter(torch.randn(L, DIM, FFN, generator=g, **bf16))
        self.w3 = torch.nn.Parameter(torch.randn(L, FFN, DIM, generator=g, **bf16))


def _gather(t):
    out = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(out, t.contiguous())
    return torch.stack(out)


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    ultraep_replicas.configure_from_options(S)
    group = dist.new_group(list(range(world)))
    experts = _Experts(rank)
    module = torch.nn.Module()
    E = L * world
    g = torch.Generator(device="cuda").manual_seed(7 + rank)
    # Skewed routing: experts 0 and 1 are hot on every rank.
    hot = torch.randint(0, 2, (TOKENS, 1), device="cuda", generator=g)
    rest = torch.randint(2, E, (TOKENS, TOPK - 1), device="cuda", generator=g)
    selected = torch.cat([hot, rest], dim=1)

    plan = ultraep_replicas.ultraep_plan(module, experts, group, selected, building=True)
    physical = plan.remap(selected)
    p2l = plan.state.gate_up.physical_to_logical_map[plan.lid].clone()
    assert torch.equal(p2l[physical].long(), selected), "remap must keep every token on a copy of its expert"

    full = plan.fetch((experts.w1, experts.w2, experts.w3), group)
    all_w = [_gather(w.detach()) for w in (experts.w1, experts.w2, experts.w3)]  # [world, L, ...]
    for j in range(L, L + S):
        lid = p2l[rank * (L + S) + j].item()
        if lid >= 0:
            owner, local = divmod(lid, L)
            for got, ref in zip(full, all_w):
                assert torch.equal(got[j], ref[owner, local]), f"spare {j} != expert {lid}"

    gen = torch.Generator(device="cuda").manual_seed(1000 + rank)
    upstream = [torch.randn(t.shape, device="cuda", dtype=torch.bfloat16, generator=gen) for t in full]
    torch.autograd.backward(full, upstream)

    all_up = [_gather(u.float()) for u in upstream]  # [world, L + S, ...]
    p2l_all = p2l.view(world, L + S)
    for k, w in enumerate((experts.w1, experts.w2, experts.w3)):
        expected = upstream[k][:L].float().clone()
        for r in range(world):
            for j in range(L, L + S):
                lid = p2l_all[r, j].item()
                if lid >= 0 and lid // L == rank:
                    expected[lid % L] += all_up[k][r, j]
        torch.testing.assert_close(w.grad.float(), expected.to(torch.bfloat16).float(), rtol=1e-2, atol=1e-2)
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 8, reason="needs 8 GPUs for UltraEP's EP group")
def test_spares_hold_their_owners_weights_and_return_their_gradients():
    paths = [str(Path(__file__).resolve().parents[2]), os.environ.get("PYTHONPATH", "")]
    completed = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc-per-node=8", __file__],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(p for p in paths if p)},
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


if __name__ == "__main__":
    main()
