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
"""The ``deepep_v2`` EP backend (moe/distributed/deepep_v2.py) against a dense reference.

Under pytest (CPU) a one-rank stand-in for DeepEP V2's ``EPBuffer`` delivers each expert's rows in a fresh random
order on every dispatch, as the EFA forks' unordered kernels do. It checks the backend's dispatch -> experts ->
combine and its gradients, that a checkpointed recompute (a bare one, and GLM's MoE layer under a whole-block
checkpoint with its CombineReplay) dispatches with its forward's handle and so rebuilds the same rows, that a
recompute without a replay is refused, the buffer bound, the SM option and the import's NCCL runtime check.

On GPUs with the DeepEP V2 fork (EP = the whole world, internode over NCCL GIN), run directly:
    torchrun --nproc-per-node 8 tests/model/test_deepep_v2_backend.py [--checkpoint] [--num-sms 0]
"""

from __future__ import annotations

import importlib
import os
import sys
import types
from dataclasses import dataclass
from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from arctic_platform.model.implementations.moe.distributed.combine_replay import CombineReplay

MODULE = "arctic_platform.model.implementations.moe.distributed.deepep_v2"
T, H, I, E, K = 12, 8, 4, 4, 2  # one rank: every expert is local


@dataclass
class _FakeHandle:
    tokens: torch.Tensor  # each received row's token
    slots: torch.Tensor  # and its top-k slot
    num_tokens: int
    num_recv_tokens_per_expert_list: list
    num_sms: int


class _FakeBuffer:
    """One rank's DeepEP V2 ``EPBuffer`` in the expanded layout: rows grouped by expert, one per (token, expert)."""

    def __init__(self, seed: int = 0):
        self.generator = torch.Generator().manual_seed(seed)
        self.fresh_dispatches = 0

    def dispatch(self, x, topk_idx=None, topk_weights=None, num_experts=None, handle=None, num_sms=0, **_):
        if handle is None:
            self.fresh_dispatches += 1
            tokens, slots, counts = [], [], []
            for e in range(num_experts):
                t, k = (topk_idx == e).nonzero(as_tuple=True)
                order = torch.randperm(t.numel(), generator=self.generator)  # arrival order
                tokens.append(t[order])
                slots.append(k[order])
                counts.append(t.numel())
            handle = _FakeHandle(torch.cat(tokens), torch.cat(slots), x.shape[0], counts, num_sms)
        weights = None if topk_weights is None else topk_weights[handle.tokens, handle.slots]
        return x[handle.tokens], None, weights, handle, None

    def combine(self, x, handle, topk_weights=None):
        out = x.new_zeros(handle.num_tokens, x.shape[1]).index_add_(0, handle.tokens, x)
        weights = None
        if topk_weights is not None:
            weights = topk_weights.new_zeros(handle.num_tokens, K)
            weights[handle.tokens, handle.slots] = topk_weights
        return out, weights, None


def _fake_deep_ep(nccl_root: str = "/nonexistent") -> types.ModuleType:
    deep_ep = types.ModuleType("deep_ep")
    deep_ep.EPBuffer = _FakeBuffer
    deep_ep.topk_idx_t = torch.int64
    utils = types.ModuleType("deep_ep.utils")
    find_pkgs = types.ModuleType("deep_ep.utils.find_pkgs")
    find_pkgs.find_nccl_root = lambda: nccl_root
    utils.find_pkgs = find_pkgs
    deep_ep.utils = utils
    return deep_ep


def _install(monkeypatch, nccl_root: str = "/nonexistent"):
    deep_ep = _fake_deep_ep(nccl_root)
    for name, module in (
        ("deep_ep", deep_ep),
        ("deep_ep.utils", deep_ep.utils),
        ("deep_ep.utils.find_pkgs", deep_ep.utils.find_pkgs),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    # Record the module's entry and the package attribute as they are, so that teardown drops this import again.
    package = importlib.import_module(MODULE.rsplit(".", 1)[0])
    monkeypatch.setitem(sys.modules, MODULE, None)
    monkeypatch.setattr(package, "deepep_v2", None, raising=False)
    del sys.modules[MODULE]
    return importlib.import_module(MODULE)


@pytest.fixture
def ep(monkeypatch):
    monkeypatch.setenv("EP_SUPPRESS_NCCL_CHECK", "1")
    return _install(monkeypatch)


def _inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(T, H, generator=g, dtype=torch.float64, requires_grad=True)
    top_w, top_i = torch.randn(T, E, generator=g, dtype=torch.float64).softmax(-1).topk(K, dim=-1)
    top_w = top_w.detach().requires_grad_(True)
    ws = [
        (torch.randn(E, *shape, generator=g, dtype=torch.float64) / 2).requires_grad_(True)
        for shape in ((I, H), (I, H), (H, I))
    ]
    dy = torch.randn(T, H, generator=g, dtype=torch.float64)
    return x, top_w, top_i, ws, dy


def _experts(rows, counts, w1, w3, w2):
    outs, start = [], 0
    for e, c in enumerate(counts):
        r = rows[start : start + c]
        start += c
        outs.append((F.silu(r @ w1[e].T) * (r @ w3[e].T)) @ w2[e].T)
    return torch.cat(outs)


def _reference(x, top_w, top_i, ws, score_before):
    y = torch.zeros_like(x)
    for e in range(E):
        tok, slot = (top_i == e).nonzero(as_tuple=True)
        w = top_w[tok, slot, None]
        r = x[tok] * w if score_before else x[tok]
        out = (F.silu(r @ ws[0][e].T) * (r @ ws[1][e].T)) @ ws[2][e].T
        y = y.index_add(0, tok, out if score_before else out * w)
    return y


def _grads(fn, x, top_w, ws, dy):
    for t in (x, top_w, *ws):
        t.grad = None
    y = fn()
    (y * dy).sum().backward()
    return [y.detach()] + [t.grad.clone() for t in (x, top_w, *ws)]


def _backend(ep, group, buffer):
    ep._buffers[(id(group), H, K)] = (buffer, T)


def _run_moe(ep, group, x, top_w, top_i, ws, score_before, replay):
    if replay is not None:
        replay.start()
    pending = ep.dispatch_tokens_async(
        x, top_i, top_w, num_experts=E, group=group, score_before_experts=score_before, replay=replay
    )
    rows, counts, state = ep.finalize_dispatch_tokens(pending)
    out = _experts(rows, counts.tolist(), *ws)
    if state.permuted_scores is not None:
        out = out * state.permuted_scores[:, None].to(out.dtype)
    y = ep.combine_tokens(out, replace(state, permuted_scores=None), replay=replay)
    ep.sync_combine()
    return y


@pytest.mark.parametrize("score_before", [True, False])
@pytest.mark.parametrize("checkpointed", [False, True])
def test_the_backend_matches_a_dense_reference(ep, score_before, checkpointed):
    """Outputs and the gradients of the tokens, routing weights and experts. Checkpointed, the recompute dispatches
    with its forward's handle: one fresh dispatch, so the recompute's rows line up with the forward graph's."""
    x, top_w, top_i, ws, dy = _inputs()
    group, buffer = object(), _FakeBuffer()
    _backend(ep, group, buffer)
    want = _grads(lambda: _reference(x, top_w, top_i, ws, score_before), x, top_w, ws, dy)
    if checkpointed:
        replay = CombineReplay()

        def run():
            return torch.utils.checkpoint.checkpoint(
                lambda *a: _run_moe(ep, group, a[0], a[1], top_i, a[2:], score_before, replay),
                x,
                top_w,
                *ws,
                use_reentrant=False,
            )

    else:

        def run():
            return _run_moe(ep, group, x, top_w, top_i, ws, score_before, None)

    got = _grads(run, x, top_w, ws, dy)
    for g, w in zip(got, want):
        torch.testing.assert_close(g, w, rtol=1e-5, atol=1e-6)  # the backend carries the routing weights in FP32
    assert buffer.fresh_dispatches == 1


def test_a_recompute_without_a_replay_is_refused(ep):
    x, top_w, top_i, ws, dy = _inputs()
    group = object()
    _backend(ep, group, _FakeBuffer())
    y = torch.utils.checkpoint.checkpoint(
        lambda *a: _run_moe(ep, group, a[0], a[1], top_i, a[2:], True, None), x, top_w, *ws, use_reentrant=False
    )
    with pytest.raises(RuntimeError, match="without a CombineReplay"):
        (y * dy).sum().backward()


def _glm_moe(ep, group):
    """GLM-5.2/5.3's MoE layer (glm_moe_dsa/modeling_glm_moe_dsa.py's arguments) on the deepep_v2 backend."""
    from arctic_platform.model.implementations.moe.layers.moe import MoE
    from arctic_platform.model.implementations.moe.layers.moe import MoEArgs

    torch.manual_seed(3)
    args = MoEArgs(
        num_experts=E,
        num_shared_experts=1,
        score_func="sigmoid",
        route_norm=True,
        route_scale=2.5,
        score_before_experts=False,
        top_k=K,
        use_grouped_mm=False,
    )
    moe = MoE(args, dim=H, hidden_dim=I).double()
    with torch.no_grad():
        for parameter in moe.parameters():
            parameter.normal_(0.0, 0.5)
    moe.set_ep_comm_backend("deepep_v2")
    moe.experts._ep_group = group
    return moe


def test_glm_moe_whole_block_recompute_replays_the_dispatch(ep):
    """A checkpointed GLM MoE layer with its CombineReplay (ac_config.combine_replay) gives the gradients of an
    uncheckpointed run although every fresh dispatch orders the rows differently."""
    from arctic_platform.model.implementations.deepseek_v41.moe import _integer_histc_on_cpu
    from arctic_platform.model.implementations.moe.layers.moe import install_combine_replay

    group = object()
    g = torch.Generator().manual_seed(5)
    x = torch.randn(1, T, H, generator=g, dtype=torch.float64)
    dy = torch.randn(1, T, H, generator=g, dtype=torch.float64)
    runs = []
    for checkpointed in (False, True):
        buffer = _FakeBuffer(seed=11 if checkpointed else 7)
        _backend(ep, group, buffer)
        moe = _glm_moe(ep, group)
        xs = x.clone().requires_grad_(True)
        with _integer_histc_on_cpu():  # the router's histc has no CPU integer kernel
            if checkpointed:
                install_combine_replay(moe)
                y = torch.utils.checkpoint.checkpoint(moe, xs, use_reentrant=False)
            else:
                y = moe(xs)
            (y * dy).sum().backward()
        runs.append([y.detach(), xs.grad] + [p.grad for p in moe.parameters() if p.requires_grad])
        assert buffer.fresh_dispatches == 1
    for got, want in zip(*reversed(runs)):
        torch.testing.assert_close(got, want)


def test_the_buffer_refuses_a_call_larger_than_it_was_sized_for(ep):
    group = object()
    ep._buffers[(id(group), H, K)] = (_FakeBuffer(), T - 1)
    x, top_w, top_i, _, _ = _inputs()
    with pytest.raises(ValueError, match="deepep_v2_max_tokens"):
        ep.dispatch_tokens_async(x, top_i, top_w, num_experts=E, group=group)


def test_the_sm_count_takes_the_option_then_the_environment_then_deepep_num_sms(ep, monkeypatch):
    from arctic_platform.model.implementations.moe.runtime_options import MoeRuntimeOptions
    from arctic_platform.model.implementations.moe.runtime_options import apply_moe_runtime_options

    monkeypatch.delenv("ARL_DEEPEP_V2_NUM_SMS", raising=False)
    try:
        apply_moe_runtime_options(MoeRuntimeOptions())
        ep.configure_num_sms(20)
        assert ep._num_sms == 20
        monkeypatch.setenv("ARL_DEEPEP_V2_NUM_SMS", "24")
        ep.configure_num_sms(20)
        assert ep._num_sms == 24
        apply_moe_runtime_options(MoeRuntimeOptions(deepep_v2_num_sms=0))
        ep.configure_num_sms(20)
        assert ep._num_sms == 0  # DeepEP V2 picks its own
    finally:
        apply_moe_runtime_options(MoeRuntimeOptions())


def _mapped_nccl() -> str | None:
    with open("/proc/self/maps") as maps:
        found = {line.split()[-1] for line in maps if os.path.basename(line.split()[-1]).startswith("libnccl.so")}
    return next(iter(found)) if len(found) == 1 else None


def test_the_import_checks_the_nccl_runtime_against_deep_eps_own(monkeypatch, tmp_path):
    """deep_ep's own import check counts aws-ofi-nccl's plugins as NCCL runtimes, so the backend skips it and
    compares the mapped ``libnccl.so*`` alone with the one deep_ep was linked with; its environment is restored."""
    mapped = _mapped_nccl()
    if mapped is None:
        pytest.skip("needs exactly one NCCL runtime mapped (torch's)")
    monkeypatch.delenv("EP_SUPPRESS_NCCL_CHECK", raising=False)
    same = tmp_path / "same" / "lib"
    same.mkdir(parents=True)
    (same / "libnccl.so.2").symlink_to(mapped)
    _install(monkeypatch, str(same.parent))
    assert "EP_SUPPRESS_NCCL_CHECK" not in os.environ
    other = tmp_path / "other" / "lib"
    other.mkdir(parents=True)
    (other / "libnccl.so.2").write_bytes(b"not the mapped runtime")
    with pytest.raises(RuntimeError, match="is not the one deep_ep was linked with"):
        _install(monkeypatch, str(other.parent))


def main():
    """EP = the whole world against a dense FP32 reference, with the DeepEP V2 fork (the bench of 073d123/d0e7124)."""
    import argparse

    import torch.distributed as dist

    from arctic_platform.model.implementations.moe.distributed import deepep_v2

    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=2560)
    parser.add_argument("--inter", type=int, default=256)
    parser.add_argument("--experts", type=int, default=512)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--num-sms", type=int, default=0)
    parser.add_argument("--checkpoint", action="store_true", help="a non-reentrant checkpoint with a CombineReplay")
    args = parser.parse_args()

    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    group = dist.new_group(list(range(world)))
    tokens, hidden, inter, experts, topk = args.tokens, args.hidden, args.inter, args.experts, args.topk
    local_experts = experts // world
    dev, bf16 = torch.device("cuda"), torch.bfloat16
    gen = torch.Generator(device=dev).manual_seed(1000 + rank)
    x = torch.randn(tokens, hidden, device=dev, generator=gen).to(bf16).requires_grad_()
    top_w, top_i = torch.randn(tokens, experts, device=dev, generator=gen).softmax(-1).topk(topk, dim=-1)
    top_w = top_w.detach().requires_grad_()
    dy = torch.randn(tokens, hidden, device=dev, generator=gen).to(bf16)
    shared = torch.Generator(device=dev).manual_seed(7)
    full = [
        (torch.randn(experts, inter, hidden, device=dev, generator=shared) / hidden**0.5).to(bf16),
        (torch.randn(experts, inter, hidden, device=dev, generator=shared) / hidden**0.5).to(bf16),
        (torch.randn(experts, hidden, inter, device=dev, generator=shared) / inter**0.5).to(bf16),
    ]
    local = slice(rank * local_experts, (rank + 1) * local_experts)
    ws = [w[local].clone().requires_grad_() for w in full]

    def gather(t):
        out = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(out, t.contiguous(), group=group)
        return torch.cat(out)

    every = [gather(t.detach()) for t in (x, top_w, top_i, dy)]
    mine = slice(rank * tokens, (rank + 1) * tokens)

    def reference(score_before):
        xr = every[0].float().requires_grad_()
        twr = every[1].clone().requires_grad_()
        wr = [w.float().requires_grad_() for w in full]
        y = torch.zeros_like(xr)
        for e in every[2].unique().tolist():
            tok, slot = (every[2] == e).nonzero(as_tuple=True)
            w = twr[tok, slot, None]
            r = xr[tok] * w if score_before else xr[tok]
            out = (F.silu(r @ wr[0][e].T) * (r @ wr[1][e].T)) @ wr[2][e].T
            y = y.index_add(0, tok, out if score_before else out * w)
        (y * every[3].float()).sum().backward()
        return [y[mine].detach(), xr.grad[mine], twr.grad[mine]] + [w.grad[local] for w in wr]

    deepep_v2.configure_num_sms(args.num_sms)
    worst = 0.0
    for score_before in (True, False):
        want = reference(score_before)
        replay = CombineReplay() if args.checkpoint else None

        def moe(x, top_w, *ws, score_before=score_before, replay=replay):
            if replay is not None:
                replay.start()
            pending = deepep_v2.dispatch_tokens_async(
                x, top_i, top_w, num_experts=experts, group=group, score_before_experts=score_before, replay=replay
            )
            rows, counts, state = deepep_v2.finalize_dispatch_tokens(pending)
            out = _experts(rows, counts.tolist(), *ws)
            if state.permuted_scores is not None:
                out = out * state.permuted_scores[:, None].to(out.dtype)
            y = deepep_v2.combine_tokens(out, replace(state, permuted_scores=None), replay=replay)
            deepep_v2.sync_combine()
            return y

        for t in (x, top_w, *ws):
            t.grad = None
        if args.checkpoint:
            y = torch.utils.checkpoint.checkpoint(moe, x, top_w, *ws, use_reentrant=False)
        else:
            y = moe(x, top_w, *ws)
        (y.float() * dy.float()).sum().backward()
        got = [y.detach(), x.grad, top_w.grad] + [w.grad for w in ws]
        errors = [float((g.float() - w).norm() / w.norm().clamp_min(1e-30)) for g, w in zip(got, want)]
        worst = max(worst, *errors)
    worst_t = torch.tensor([worst], device=dev)
    dist.all_reduce(worst_t, op=dist.ReduceOp.MAX)
    if rank == 0:
        verdict = "PASSED" if worst_t.item() < 2e-2 else "FAILED"
        print(
            f"CHECK {verdict} EP{world}{' checkpointed' if args.checkpoint else ''} worst rel err {worst_t.item():.2e}"
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
