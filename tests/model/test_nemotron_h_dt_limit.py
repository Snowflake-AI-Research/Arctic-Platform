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
"""CPU: Nemotron-H's Mamba-2 mixers train with the time step vLLM samples with, dt = softplus(dt + dt_bias) unfloored
(dt_limit (0, inf), vllm mamba_mixer2.py:704, :883), not Transformers' floor at time_step_min = 1e-3.

Every path that reads the limit hands (0, inf) to its kernel: the fused varlen node (cp_mamba.fused_mamba_mixer), the
per-op varlen path, the per-op context-parallel path (cp_mamba.mamba_cp_forward) and the mixer's own forward for calls
without packed rows. The kernels are recorded, or (for the scan) replaced by a float64 stand-in with mamba_ssm's
semantics, so the two per-op paths also run end to end on CPU against a float64 unfloored mixer.
"""

import inspect
import math

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from arctic_platform.model.implementations.layers import cp_mamba
from arctic_platform.model.implementations.nemotron_h import modeling_nemotron_h as nemotron
from arctic_platform.model.implementations.nemotron_h.configuration_nemotron_h import NemotronHConfig

UNFLOORED = (0.0, math.inf)
CU = [0, 37, 38, 100]  # packed rows, one of a single token, none a multiple of the 16-token chunk
FLOORED_HEADS = [0, 1]  # softplus(dt_bias) of 6e-6 and 1.7e-5: under 1e-3 at every token


def _config():
    config = NemotronHConfig(
        hidden_size=64,
        layers_block_type=["mamba"],
        ssm_state_size=16,
        mamba_num_heads=4,
        mamba_n_groups=2,
        mamba_head_dim=16,
        mamba_chunk_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=64,
        moe_intermediate_size=64,
        moe_shared_expert_intermediate_size=64,
        n_routed_experts=4,
        num_experts_per_tok=2,
    )
    nemotron._ensure_zamba2_compat(config)
    return config


def _mixer():
    """The training mixer (build_mamba2_mixer). Heads 0 and 1, one gated-norm group, sit under 1e-3 at every token
    with no D skip (as Nemotron-3-Nano layer 0's floored heads), so the scan carries their output; O(1) projections."""
    config = _config()
    mixer = nemotron.build_mamba2_mixer(config, 0)
    gen = torch.Generator().manual_seed(0)
    with torch.no_grad():
        mixer.in_proj.weight.copy_(torch.randn(mixer.in_proj.weight.shape, generator=gen) * config.hidden_size**-0.5)
        mixer.conv1d.weight.copy_(torch.randn(mixer.conv1d.weight.shape, generator=gen) * 0.3)
        mixer.conv1d.bias.copy_(torch.randn(mixer.conv1d.bias.shape, generator=gen) * 0.1)
        mixer.dt_bias.copy_(torch.tensor([-12.0, -11.0, 0.5, 2.0]))
        mixer.A_log.copy_(torch.rand(4, generator=gen) * 2 - 1)
        mixer.D.copy_(torch.tensor([0.0, 0.0, 1.0, -0.5]))
        mixer.norm.weight.copy_(torch.rand(mixer.norm.weight.shape, generator=gen) + 0.5)
        mixer.out_proj.weight.copy_(torch.randn(mixer.out_proj.weight.shape, generator=gen) * 64**-0.5)
    return mixer


def _scan_float64(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, seq_idx=None, dt_softplus=False, **kwargs):
    """mamba_chunk_scan_combined's semantics in float64, token by token: dt + dt_bias, softplus below 20, clamped to
    dt_limit (no gradient where it binds), the state reset where seq_idx changes, the D skip."""
    _RECORDED.append(kwargs.get("dt_limit", (0.0, math.inf)))
    batch, length, heads, head_dim = x.shape
    groups = B.shape[2]
    dt = dt.double() + (0 if dt_bias is None else dt_bias.double())
    if dt_softplus:
        dt = torch.where(dt <= 20.0, F.softplus(dt), dt)
    low, high = kwargs.get("dt_limit", (0.0, math.inf))
    dt = dt.clamp(low, high)
    group = torch.arange(heads) // (heads // groups)
    state = x.new_zeros(batch, heads, head_dim, B.shape[-1], dtype=torch.float64)
    ys = []
    for t in range(length):
        if seq_idx is not None and t > 0:
            state = state * (seq_idx[:, t] == seq_idx[:, t - 1]).double()[:, None, None, None]
        decay = torch.exp(A.double() * dt[:, t])[:, :, None, None]
        state = state * decay + (dt[:, t, :, None] * x[:, t].double())[..., None] * B[:, t, group].double()[:, :, None]
        y = torch.einsum("bhpn,bhn->bhp", state, C[:, t, group].double())
        ys.append(y if D is None else y + D.double()[:, None] * x[:, t].double())
    out = torch.stack(ys, 1).to(x.dtype)
    return (out, None) if kwargs.get("return_final_states") else out


_RECORDED = []


@pytest.fixture
def recorded(monkeypatch):
    """The dt_limit every scan call receives; the CPU scan stand-in in place of mamba_ssm's Triton one."""
    ssd_combined = pytest.importorskip("mamba_ssm.ops.triton.ssd_combined")

    _RECORDED.clear()
    monkeypatch.setattr(ssd_combined, "mamba_chunk_scan_combined", _scan_float64)
    return _RECORDED


@pytest.fixture
def patched(monkeypatch, recorded):
    """Arctic's patched mixer forward and its varlen paths, built on CPU (the patch wants CUDA and mamba_ssm), with
    the mixer class restored afterwards."""
    cls = nemotron._nemotron_h_mamba2_mixer_cls()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(nemotron, "_patch_applied", False)
    monkeypatch.setattr(cls, "forward", cls.forward)
    monkeypatch.setattr(cls, "_arl_stock_forward", getattr(cls, "_arl_stock_forward", cls.forward), raising=False)
    nemotron._patch_mamba2_use_triton_ssd()
    return inspect.getclosurevars(cls.forward).nonlocals


def test_the_mamba_layer_builds_its_mixer_with_the_samplers_dt_limit():
    config = _config()
    stock = nemotron._nemotron_h_mamba2_mixer_cls()(config, layer_idx=0)
    assert stock.time_step_limit == (config.time_step_min, math.inf) == (1e-3, math.inf)  # what is overridden
    layer = nemotron.NemotronHMambaLayer(config, 0)
    assert layer.mamba.time_step_limit == UNFLOORED
    assert nemotron.SAMPLER_TIME_STEP_LIMIT == UNFLOORED


def test_a_mixer_without_time_step_limit_is_refused(monkeypatch):
    class Mixer(nn.Module):
        def __init__(self, config, layer_idx):
            super().__init__()

    monkeypatch.setattr(nemotron, "_nemotron_h_mamba2_mixer_cls", lambda: Mixer)
    with pytest.raises(ValueError, match="has no time_step_limit"):
        nemotron.build_mamba2_mixer(_config(), 0)


def test_the_fused_varlen_node_gets_the_unfloored_dt_limit(monkeypatch):
    ssd_combined = pytest.importorskip("mamba_ssm.ops.triton.ssd_combined")

    calls = []

    def fused(zxbcdt, *args, **kwargs):
        calls.append(kwargs["dt_limit"])
        return zxbcdt.new_zeros(*zxbcdt.shape[:2], 64)

    monkeypatch.setattr(ssd_combined, "mamba_split_conv1d_scan_combined", fused)
    mixer = _mixer()
    zxbcdt = mixer.in_proj(torch.randn(1, CU[-1], 64))
    cp_mamba.fused_mamba_mixer(mixer, zxbcdt, cp_mamba.packed_seq_idx(torch.tensor(CU), CU[-1]))
    assert calls == [UNFLOORED]


def test_the_mixers_own_forward_gets_the_unfloored_dt_limit(monkeypatch, patched):
    """Calls without packed rows take the mixer's own forward (Transformers' Zamba2MambaMixer.forward), whose fused
    training call reads time_step_limit (modeling_zamba2.py:737)."""
    from transformers.models.zamba2 import modeling_zamba2

    calls = []

    def fused(projected_states, *args, **kwargs):
        calls.append(kwargs["dt_limit"])
        return projected_states.new_zeros(*projected_states.shape[:2], 64)

    monkeypatch.setattr(modeling_zamba2, "mamba2_split_conv1d_scan_combined", fused)
    mixer = _mixer().train()
    mixer(torch.randn(1, CU[-1], 64))
    assert calls == [UNFLOORED]


def _float64_reference(mixer, hidden, dy):
    """The mixer in float64 torch ops, one packed row at a time, at the sampler's dt = softplus(dt + dt_bias)."""
    p = {n: q.detach().double().requires_grad_() for n, q in mixer.named_parameters()}
    x_in = hidden[0].detach().double()
    heads, head_dim, groups, state = mixer.num_heads, mixer.head_dim, mixer.n_groups, mixer.ssm_state_size
    dim = heads * head_dim
    z, xBC, dt_raw = (x_in @ p["in_proj.weight"].t()).split([dim, mixer.conv_dim, heads], dim=-1)
    group = torch.arange(heads) // (heads // groups)
    outs = []
    for s, e in zip(CU[:-1], CU[1:]):
        conv = F.conv1d(xBC[s:e].t()[None], p["conv1d.weight"], p["conv1d.bias"], padding=3, groups=mixer.conv_dim)
        x, B, C = F.silu(conv[0, :, : e - s]).t().split([dim, groups * state, groups * state], dim=-1)
        dt = F.softplus(dt_raw[s:e] + p["dt_bias"])
        h, ys = torch.zeros(heads, head_dim, state, dtype=torch.float64), []
        for t in range(e - s):
            xt = x[t].view(heads, head_dim)
            h = h * torch.exp(-torch.exp(p["A_log"]) * dt[t])[:, None, None]
            h = h + dt[t][:, None, None] * xt[:, :, None] * B[t].view(groups, state)[group][:, None, :]
            ys.append(torch.einsum("hpn,hn->hp", h, C[t].view(groups, state)[group]) + p["D"][:, None] * xt)
        outs.append(torch.stack(ys).reshape(e - s, dim))
    y = (torch.cat(outs) * F.silu(z)).view(-1, groups, dim // groups)
    y = (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + mixer.norm.variance_epsilon)).view(-1, dim)
    out = (y * p["norm.weight"]) @ p["out_proj.weight"].t()
    out.backward(dy[0].double())
    return out.detach(), {n: t.grad for n, t in p.items()}


def _per_op_paths(patched, single_rank_group):
    import torch.distributed as dist

    group = dist.new_group(ranks=[0], backend="gloo")  # CPU tensors, whatever the default group's backend

    def context_parallel(mixer, hidden, cu):  # one rank: the all-to-alls are identities
        return cp_mamba.mamba_cp_forward(mixer, hidden, group, 0, 1, cu_seqlens=cu)

    return {"per-op varlen": patched["_varlen_mamba_forward"], "per-op context parallel": context_parallel}


@pytest.mark.parametrize("path", ["per-op varlen", "per-op context parallel"])
def test_the_per_op_paths_match_a_float64_unfloored_mixer(patched, recorded, single_rank_group, path):
    """Both per-op paths hand the scan (0, inf) and, with it, match a float64 mixer at dt = softplus(dt + dt_bias) on
    the two heads under 1e-3. With Transformers' floor put back, the output moves by percents and the floored heads'
    dt_bias gets no gradient."""
    forward = _per_op_paths(patched, single_rank_group)[path]
    assert _mixer().time_step_limit == UNFLOORED  # as built for training
    torch.manual_seed(1)
    hidden = torch.randn(1, CU[-1], 64)
    dy = torch.randn(1, CU[-1], 64)
    exact, exact_grads = _float64_reference(_mixer(), hidden, dy)
    cu = torch.tensor(CU, dtype=torch.int32)
    results = {}
    for limit in (UNFLOORED, (1e-3, math.inf)):
        mixer = _mixer()
        mixer.time_step_limit = limit
        recorded.clear()
        out = forward(mixer, hidden, cu)
        out.backward(dy)
        assert recorded == [limit]
        results[limit] = (out.detach().double()[0], {n: q.grad.double() for n, q in mixer.named_parameters()})

    def rel(a, b):
        return ((a - b).norm() / b.norm()).item()

    out, grads = results[UNFLOORED]
    assert rel(out, exact) < 1e-5, rel(out, exact)  # FP32 conv / norm / projections against float64
    for name in ("dt_bias", "A_log", "conv1d.weight", "in_proj.weight", "norm.weight"):
        assert rel(grads[name], exact_grads[name]) < 1e-4, (name, rel(grads[name], exact_grads[name]))
    floored_out, floored_grads = results[(1e-3, math.inf)]
    assert rel(floored_out, exact) > 1e-2
    assert torch.all(floored_grads["dt_bias"][FLOORED_HEADS] == 0)
    assert torch.all(exact_grads["dt_bias"][FLOORED_HEADS].abs() > 0)
