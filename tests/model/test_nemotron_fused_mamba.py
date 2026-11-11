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


def _reference_mixer(mixer, zxbcdt, cu_seqlens):
    """The per-op path of mamba_cp_forward (cp 1): per-row conv1d + SiLU, the chunked scan, the FP32 gated norm."""
    from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

    inter, state = mixer.intermediate_size, mixer.n_groups * mixer.ssm_state_size
    gate, xbc, dt = zxbcdt.split([inter, inter + 2 * state, mixer.num_heads], dim=-1)
    xbc = cp_mamba._causal_conv_packed(xbc, mixer.conv1d.weight, mixer.conv1d.bias, xbc.shape[-1], cu_seqlens)
    x, b, c = xbc.split([inter, state, state], dim=-1)
    length = zxbcdt.shape[1]
    out, _ = mamba_chunk_scan_combined(
        x.view(1, length, mixer.num_heads, mixer.head_dim),
        dt,
        -torch.exp(mixer.A_log.float()),
        b.view(1, length, mixer.n_groups, -1),
        c.view(1, length, mixer.n_groups, -1),
        chunk_size=mixer.chunk_size,
        D=mixer.D,
        z=None,
        seq_idx=cp_mamba._seq_idx_from_cu_seqlens(cu_seqlens, length, zxbcdt.device),
        return_final_states=True,
        dt_bias=mixer.dt_bias,
        dt_softplus=True,
        dt_limit=tuple(mixer.time_step_limit),
    )
    out = out.view(1, length, inter)
    return cp_mamba._gated_rms_norm(out, gate, mixer.norm.weight, mixer.norm.variance_epsilon, mixer.norm.group_size)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_packed_seq_idx_matches_the_row_loop_and_is_built_once():
    cu = torch.tensor(CU, dtype=torch.int32)
    got = cp_mamba.packed_seq_idx(cu, CU[-1])
    torch.testing.assert_close(got, cp_mamba._seq_idx_from_cu_seqlens(cu, CU[-1], cu.device), rtol=0, atol=0)
    assert cp_mamba.packed_seq_idx(cu, CU[-1]) is got
    other = cu.clone()
    assert cp_mamba.packed_seq_idx(other, CU[-1]) is not got


def test_fused_mixer_matches_the_per_op_path():
    _fused_or_skip()
    mixer = _mixer()
    cu = torch.tensor(CU, dtype=torch.int32, device="cuda")
    width = 2 * mixer.intermediate_size + 2 * mixer.n_groups * mixer.ssm_state_size + mixer.num_heads
    zxbcdt = torch.randn(1, CU[-1], width, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn(1, CU[-1], mixer.intermediate_size, device="cuda", dtype=torch.bfloat16)
    params = [mixer.conv1d.weight, mixer.conv1d.bias, mixer.dt_bias, mixer.A_log, mixer.D, mixer.norm.weight]

    def run(fn):
        z = zxbcdt.clone().requires_grad_(True)
        out = fn(z)
        grads = torch.autograd.grad(out, [z] + params, dy)
        return out, grads

    got, got_grads = run(lambda z: cp_mamba.fused_mamba_mixer(mixer, z, cp_mamba.packed_seq_idx(cu, CU[-1])))
    want, want_grads = run(lambda z: _reference_mixer(mixer, z, cu))
    assert _rel(got, want) < 1e-2
    for g, w in zip(got_grads, want_grads):
        assert _rel(g, w) < 2e-2


def test_fused_mixer_on_a_ranks_heads_is_that_slice_of_the_full_mixer():
    """What each context-parallel rank computes on its share of z, x, B, C and dt."""
    _fused_or_skip()
    mixer = _mixer()
    size = 2
    cu = torch.tensor(CU, dtype=torch.int32, device="cuda")
    seq_idx = cp_mamba.packed_seq_idx(cu, CU[-1])
    inter, state, heads = mixer.intermediate_size, mixer.n_groups * mixer.ssm_state_size, mixer.num_heads
    widths = (inter, inter, state, state, heads)
    zxbcdt = torch.randn(1, CU[-1], sum(widths), device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        full = cp_mamba.fused_mamba_mixer(mixer, zxbcdt, seq_idx)
        parts = zxbcdt.split(list(widths), dim=-1)
        for rank in range(size):
            local = torch.cat([p.view(1, CU[-1], size, -1)[:, :, rank] for p in parts], dim=-1)
            got = cp_mamba.fused_mamba_mixer(mixer, local, seq_idx, rank, size)
            share = inter // size
            assert _rel(got, full[..., rank * share : (rank + 1) * share]) < 2e-3


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
