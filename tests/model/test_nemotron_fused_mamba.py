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
"""Nemotron-H's fused Mamba-2 mixer (conv + scan + gated norm in one node, one all-to-all under CP) against the
per-op path it replaces."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from arctic_platform.model.implementations.layers import cp_mamba

CU = [0, 100, 101, 300, 512]  # packed rows, one of a single token


def _fused_or_skip():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("mamba_ssm's Triton kernels need a CUDA GPU")
    if not cp_mamba.fused_mamba_available():
        pytest.skip("needs mamba_ssm and causal_conv1d")


def _mixer(heads=8, groups=2):
    from arctic_platform.model.implementations.nemotron_h.configuration_nemotron_h import NemotronHConfig
    from arctic_platform.model.implementations.nemotron_h.modeling_nemotron_h import _ensure_zamba2_compat
    from arctic_platform.model.implementations.nemotron_h.modeling_nemotron_h import _nemotron_h_mamba2_mixer_cls

    config = NemotronHConfig(
        hidden_size=256,
        layers_block_type=["mamba"],
        ssm_state_size=32,
        mamba_num_heads=heads,
        mamba_n_groups=groups,
        mamba_head_dim=32,
        mamba_chunk_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=128,
        n_routed_experts=4,
        num_experts_per_tok=2,
    )
    _ensure_zamba2_compat(config)
    torch.manual_seed(0)
    mixer = _nemotron_h_mamba2_mixer_cls()(config, layer_idx=0)
    with torch.no_grad():
        mixer.A_log.uniform_(-1, 1)
        mixer.dt_bias.uniform_(-3, -1)
        mixer.D.normal_()
        mixer.norm.weight.normal_(1.0, 0.1)
    return mixer.cuda().bfloat16()


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _cp_run(rank, world, port, results):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world)
    group = dist.group.WORLD
    mixer = _mixer()
    torch.manual_seed(1)
    hidden = torch.randn(1, CU[-1], mixer.hidden_size, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(1, CU[-1], mixer.hidden_size, device="cuda", dtype=torch.bfloat16)
    local_len = CU[-1] // world
    cu = torch.tensor(CU, dtype=torch.int32, device="cuda")
    out = {}
    for name, forward in (("fused", cp_mamba.mamba_cp_forward_fused), ("reference", cp_mamba.mamba_cp_forward)):
        mixer.zero_grad(set_to_none=True)
        x = hidden[:, rank * local_len : (rank + 1) * local_len].clone().requires_grad_(True)
        y = forward(mixer, x, group, rank, world, cu_seqlens=cu)
        y.backward(dy[:, rank * local_len : (rank + 1) * local_len])
        grads = [p.grad.detach().clone() for p in (mixer.in_proj.weight, mixer.conv1d.weight, mixer.A_log)]
        for g in grads:  # each rank holds its heads' part of the gradient; DeepSpeed's reduction sums them
            dist.all_reduce(g)
        out[name] = [y.detach(), x.grad] + grads
    results[rank] = [_rel(g, w) for g, w in zip(out["fused"], out["reference"])]
    dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_context_parallel_fused_forward_matches_the_per_op_path():
    _fused_or_skip()
    world = 2
    results = mp.Manager().dict()
    mp.spawn(_cp_run, args=(world, 29533, results), nprocs=world, join=True)
    for rank in range(world):
        out_rel, grad_x_rel, *param_rels = results[rank]
        assert out_rel < 1e-2 and grad_x_rel < 2e-2, (rank, results[rank])
        assert max(param_rels) < 2e-2, (rank, results[rank])
