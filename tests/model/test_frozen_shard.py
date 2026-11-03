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
"""Frozen replicated layer weights sharded across ranks and gathered per call, under activation checkpointing."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = nn.Linear(64, 64, bias=False).requires_grad_(False)
        self.scale = nn.Parameter(torch.rand(64, dtype=torch.float32), requires_grad=False)
        self.lora = nn.Linear(64, 64, bias=False)
        self.expert = nn.Parameter(torch.randn(3, 5), requires_grad=False)
        self.expert.allreduce = False

    def forward(self, x):
        return (self.frozen(x) * self.scale.to(x.dtype) + self.lora(x)).relu()


def _run(rank, world, port, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    from arctic_platform.model.implementations.moe.frozen_shard import install_frozen_layer_shards

    torch.manual_seed(0)
    reference = nn.ModuleList([_Layer(), _Layer()]).cuda().bfloat16()
    for layer in reference:
        layer.scale.data = layer.scale.data.float()
    sharded = nn.ModuleList([checkpoint_wrapper(_Layer(), early_stop=False) for _ in range(2)]).cuda().bfloat16()
    for ref, layer in zip(reference, sharded):
        params = dict(layer._checkpoint_wrapped_module.named_parameters())
        for name, param in ref.named_parameters():
            params[name].data = param.data.clone()
    install_frozen_layer_shards(sharded)

    x = torch.randn(4, 64, device="cuda", dtype=torch.bfloat16)
    want = reference[1](reference[0](x))
    want.float().sum().backward()
    for step in range(2):
        for layer in sharded:
            layer.zero_grad(set_to_none=True)
        got = sharded[1](sharded[0](x))
        got.float().sum().backward()
        assert torch.equal(got, want), step
        for ref, layer in zip(reference, sharded):
            assert torch.equal(layer._checkpoint_wrapped_module.lora.weight.grad, ref.lora.weight.grad), step
    inner = sharded[0]._checkpoint_wrapped_module
    results[rank] = (
        inner.frozen.weight.numel(),
        inner.scale.numel(),
        inner.expert.numel(),
        inner._ap_frozen_shard.shard.numel(),
    )
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_sharded_frozen_weights_match_replicated_ones():
    world = 2
    results = mp.Manager().dict()
    mp.spawn(_run, args=(world, 29511, results), nprocs=world, join=True)
    for rank in range(world):
        frozen, scale, expert, shard = results[rank]
        assert frozen == 0 and scale == 0, "frozen replicated weights keep only their shard"
        assert expert == 15, "expert-parallel parameters stay whole"
        assert shard < 64 * 64 * 2 + 64 * 4
