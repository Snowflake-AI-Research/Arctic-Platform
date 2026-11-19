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
per-op path it replaces, and both varlen paths' gradients against a float64 reference."""

import inspect
import math
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

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
    from arctic_platform.model.implementations.nemotron_h.modeling_nemotron_h import build_mamba2_mixer

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
    mixer = build_mamba2_mixer(config, 0)
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


# Packed documents for the non-CP varlen path: one of a single token, none a multiple of the 128-token chunk. The
# upstream gradient reaches only a prefix of each (the rest is padding, as in the correctness suite's rows).
VARLEN_CU = [0, 700, 701, 1900, 3072]
VARLEN_ACTIVE = [700, 1, 600, 1000]
# Bounds against float64, per gradient: |norm(grad) / norm(float64 grad) - 1| and norm(grad - float64 grad) /
# norm(float64 grad). Over 10 seeds on an H200 (at the sampler's unfloored dt), BF16 rounding put conv1d.weight's
# relative errors at 4.4e-3 to 4.7e-3 (fused) and 5.7e-3 to 6.2e-3 (per-op) and every weight's norm ratio within 7e-4;
# the per-head scalars (dt_bias, A_log, D) sum fewer terms (ratios to 6.1e-3, errors to 9.8e-3). So a 1% conv-weight
# error fails: a 1% scale moves the ratio by 1e-2, a 1% error in a random direction puts the error at 1.1e-2 or more.
RATIO_TOL = {"dt_bias": 1e-2, "A_log": 1e-2, "D": 1e-2}
RATIO_TOL_DEFAULT = 3e-3
REL_TOL = {"conv1d.weight": 9e-3, "conv1d.bias": 9e-3}
REL_TOL_DEFAULT = 2e-2


def _layer0_like_mixer(seed, hidden_size=512):
    """Nemotron-3-Nano's mixer shapes (64 heads of 64, 8 groups of 128 states, width-4 conv, 128-token chunks) in the
    regimes its layer 0 puts the kernels in: dt under 1e-3 at every token on 6 heads (where Transformers' mixer would
    floor it; the training mixer, build_mamba2_mixer, runs the sampler's unfloored dt) and crossing 1e-3 from token to
    token on 2, past the softplus threshold of 20 on 4, A down to -1800 where dt is small, D of 26 on 4 heads.

    The rest is kept well conditioned, so that the bounds measure the mixer rather than the layer: each head's decay
    per token (dt * |A|) within 1e-2 to 2.5, its gain sqrt(dt / |A|) under 3, small conv biases. Random layers with
    O(1) conv biases (layer 0's reach 25.5), long high-gain memories, or (with a dt floor) heads whose dt crosses it
    from token to token let BF16 rounding of the forward alone move conv1d's gradients by 5-100% on both paths alike (a
    float64 emulation of the BF16 rounding points shows it). The trained layer on its real input does not.
    """
    from arctic_platform.model.implementations.nemotron_h.configuration_nemotron_h import NemotronHConfig
    from arctic_platform.model.implementations.nemotron_h.modeling_nemotron_h import _ensure_zamba2_compat
    from arctic_platform.model.implementations.nemotron_h.modeling_nemotron_h import build_mamba2_mixer

    config = NemotronHConfig(
        hidden_size=hidden_size,
        layers_block_type=["mamba"],
        ssm_state_size=128,
        mamba_num_heads=64,
        mamba_n_groups=8,
        mamba_head_dim=64,
        mamba_chunk_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=128,
        n_routed_experts=4,
        num_experts_per_tok=2,
    )
    _ensure_zamba2_compat(config)
    mixer = build_mamba2_mixer(config, 0)
    gen = torch.Generator().manual_seed(seed)

    def uniform(n, low, high):
        return torch.rand(n, generator=gen) * (high - low) + low

    heads, channels = mixer.num_heads, mixer.conv_dim
    dt_bias = uniform(heads, math.log(0.02), math.log(21.0)).exp().expm1().log()  # softplus(dt_bias) from 0.02 to 21
    which = torch.randperm(heads, generator=gen)
    dt_bias[which[:6]] = uniform(6, -12.0, -10.0)  # dt of 6e-6 to 4.5e-5, under 1e-3 at every token
    dt_bias[which[6:8]] = uniform(2, -7.5, -6.5)  # dt around 1e-3, crossing it from token to token
    dt_bias[which[8:12]] = uniform(4, 20.5, 21.25)  # past softplus's threshold of 20
    dt = F.softplus(dt_bias).clamp(min=1e-3)  # A from the decay at 1e-3 for the smallest dt: A to -1800 there
    decay = uniform(heads, math.log(1e-2), math.log(2.5)).exp()
    A = torch.maximum(decay / dt, dt / 9.0).clamp(max=1800.0)
    D = uniform(heads, -1.0, 4.0)
    D[torch.randperm(heads, generator=gen)[:4]] = 25.9
    with torch.no_grad():
        mixer.in_proj.weight.copy_(torch.randn(mixer.in_proj.weight.shape, generator=gen) * hidden_size**-0.5)
        mixer.conv1d.weight.copy_((torch.randn(mixer.conv1d.weight.shape, generator=gen) * 0.064).clamp(-0.6, 0.6))
        mixer.conv1d.bias.copy_(torch.randn(channels, generator=gen) * 0.1)
        mixer.dt_bias.copy_(dt_bias)
        mixer.A_log.copy_(A.log())
        mixer.D.copy_(D)
        mixer.norm.weight.copy_(
            (torch.randn(mixer.intermediate_size, generator=gen) * 0.79 + 0.61).abs().clamp(max=9.25)
        )
        mixer.out_proj.weight.copy_(
            torch.randn(mixer.out_proj.weight.shape, generator=gen) * mixer.intermediate_size**-0.5
        )
    return mixer.cuda().bfloat16()


def _varlen_forwards(mixer):
    """The patched mixer's two varlen forwards, per-op and fused (``_patch_mamba2_use_triton_ssd``'s closure)."""
    from arctic_platform.model.implementations.nemotron_h.modeling_nemotron_h import _patch_mamba2_use_triton_ssd

    _patch_mamba2_use_triton_ssd()
    found = inspect.getclosurevars(type(mixer).forward).nonlocals
    return {"per-op": found["_varlen_mamba_forward"], "fused": found["_fused_varlen_mamba_forward"]}


def _segsum(x):
    T = x.size(-1)
    x = x[..., None].expand(*x.shape, T)
    x = x.masked_fill(~torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=-1), 0)
    x = torch.cumsum(x, dim=-2)
    return x.masked_fill(~torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool)), -torch.inf)


def _ssd(X, A, B, C, chunk):
    """mamba_ssm's ssd_minimal_discrete from a zero state: X = x * dt, A = A * dt, B and C per head, length a multiple
    of ``chunk``."""
    length, heads, head_dim = X.shape
    X, A, B, C = (t.reshape(length // chunk, chunk, *t.shape[1:]) for t in (X, A, B, C))
    A = A.permute(2, 0, 1)  # heads, chunks, positions
    A_cumsum = torch.cumsum(A, dim=-1)
    y_diag = torch.einsum("clhn,cshn,hcls,cshp->clhp", C, B, torch.exp(_segsum(A)), X)
    states = torch.einsum("clhn,hcl,clhp->chpn", B, torch.exp(A_cumsum[..., -1:] - A_cumsum), X)
    states = torch.cat([torch.zeros_like(states[:1]), states])
    states = torch.einsum("hzc,chpn->zhpn", torch.exp(_segsum(F.pad(A_cumsum[..., -1], (1, 0)))), states)[:-1]
    y_off = torch.einsum("clhn,chpn,hcl->clhp", C, states, torch.exp(A_cumsum))
    return (y_diag + y_off).reshape(length, heads, head_dim)


def _float64_mixer(mixer, hidden, cu):
    """The mixer in float64 torch ops, one document at a time: in_proj, causal conv1d and SiLU, the scan with the D
    skip at the sampler's time step (vLLM's MambaMixer2: softplus(dt + dt_bias), never floored), the gated group
    RMSNorm, out_proj. Returns the input, the parameters and the output, all float64 leaves / results for backward."""
    p = {n: q.detach().double().requires_grad_() for n, q in mixer.named_parameters()}
    x_in = hidden.detach()[0].double().requires_grad_()
    heads, head_dim, groups, state = mixer.num_heads, mixer.head_dim, mixer.n_groups, mixer.ssm_state_size
    dim, width, chunk = heads * head_dim, p["conv1d.weight"].shape[-1], mixer.chunk_size
    z, xBC, dt_raw = (x_in @ p["in_proj.weight"].t()).split([dim, mixer.conv_dim, heads], dim=-1)
    A = -torch.exp(p["A_log"])
    group = torch.arange(heads, device=x_in.device) // (heads // groups)
    outs = []
    for s, e in zip(cu[:-1], cu[1:]):
        conv = F.conv1d(
            xBC[s:e].t()[None], p["conv1d.weight"], p["conv1d.bias"], padding=width - 1, groups=xBC.shape[1]
        )
        x, B, C = F.silu(conv[0, :, : e - s]).t().split([dim, groups * state, groups * state], dim=-1)
        dt = dt_raw[s:e] + p["dt_bias"]
        dt = torch.where(dt <= 20.0, F.softplus(dt), dt)
        x = x.reshape(e - s, heads, head_dim)
        B, C = (t.reshape(e - s, groups, state)[:, group] for t in (B, C))
        pad = -(e - s) % chunk
        padded = [F.pad(t, (0, 0) * (t.dim() - 1) + (0, pad)) for t in (x * dt[..., None], A * dt, B, C)]
        y = _ssd(*padded, chunk)[: e - s] + x * p["D"][:, None]
        outs.append(y.reshape(e - s, dim))
    y = (torch.cat(outs) * F.silu(z)).reshape(-1, groups, dim // groups)
    y = (y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + mixer.norm.variance_epsilon)).reshape(-1, dim)
    return x_in, p, (y * p["norm.weight"]) @ p["out_proj.weight"].t()


def varlen_gradient_errors(seed):
    """Per varlen path and per gradient (parameters and the input): (norm ratio, relative error) against float64, at
    one fixed upstream gradient."""
    mixer = _layer0_like_mixer(seed)
    assert mixer.time_step_limit == (0.0, float("inf"))  # the sampler's, as built for training
    assert (F.softplus(mixer.dt_bias.float()) < 1e-3).sum() >= 6  # heads Transformers' 1e-3 floor would raise
    gen = torch.Generator(device="cuda").manual_seed(seed)
    total = VARLEN_CU[-1]
    hidden = torch.randn(1, total, mixer.hidden_size, device="cuda", generator=gen).bfloat16()
    mask = torch.zeros(1, total, 1, device="cuda")
    for start, active in zip(VARLEN_CU[:-1], VARLEN_ACTIVE):
        mask[:, start : start + active] = 1
    dy = (torch.randn(1, total, mixer.hidden_size, device="cuda", generator=gen) * mask).bfloat16()
    x64, p64, y64 = _float64_mixer(mixer, hidden, VARLEN_CU)
    y64.backward(dy[0].double())
    reference = {n: t.grad for n, t in p64.items()} | {"input": x64.grad}
    cu = torch.tensor(VARLEN_CU, dtype=torch.int32, device="cuda")
    errors = {}
    for path, forward in _varlen_forwards(mixer).items():
        mixer.zero_grad(set_to_none=True)
        x = hidden.clone().requires_grad_(True)
        forward(mixer, x, cu).backward(dy)
        grads = {n: t.grad for n, t in mixer.named_parameters()} | {"input": x.grad[0]}
        errors[path] = {
            n: (
                (g.double().norm() / reference[n].norm()).item(),
                ((g.double() - reference[n]).norm() / reference[n].norm()).item(),
            )
            for n, g in grads.items()
        }
    return errors


@pytest.mark.parametrize("seed", [0, 1])
def test_varlen_gradients_match_float64(seed):
    """Both non-CP varlen paths (one packed call, seq_idx boundaries) on layer-0-like parameters: every gradient's norm
    and error against float64's at the sampler's unfloored dt within its bounds, at one fixed upstream gradient, with
    8 heads' dt under 1e-3 at some or every token. The correctness suite's whole-model gaps on these tensors
    (Nemotron-3-Nano layer 0's dt_bias, A_log, D, conv1d) move with the upstream gradient, which BF16 rounding anywhere
    above layer 0 changes; at a fixed one the mixer itself is checked."""
    _fused_or_skip()
    errors = varlen_gradient_errors(seed)
    for path, per_tensor in errors.items():
        for name, (ratio, rel) in per_tensor.items():
            bounds = RATIO_TOL.get(name, RATIO_TOL_DEFAULT), REL_TOL.get(name, REL_TOL_DEFAULT)
            assert abs(ratio - 1) < bounds[0] and rel < bounds[1], (path, name, ratio, rel, bounds)
