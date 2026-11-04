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

"""GLM sparse MLA under context parallelism equals the full-sequence computation."""

from __future__ import annotations

import pytest
import torch
import torch.multiprocessing as mp

from arctic_platform.testing_utils import get_unique_port_number
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import require_torch_multi_gpu
from arctic_platform.testing_utils import reserve_free_port
from arctic_platform.testing_utils import set_seed

_S = 4096


def _inputs(hidden_size: int):
    set_seed(0)
    x = (torch.randn(1, _S, hidden_size, device="cuda") * 0.5).bfloat16()
    grad = torch.randn(1, _S, hidden_size, device="cuda").bfloat16()
    # Two packed documents, so ``ks`` exercises a sequence start inside the shard boundary.
    pos = torch.cat([torch.arange(1500), torch.arange(_S - 1500)]).cuda()
    freqs = 1.0 / (10000 ** (torch.arange(0, 64, 2, device="cuda").float() / 64))
    angles = (pos[:, None].float() * freqs[None]).repeat_interleave(2, dim=-1)[None]
    ks = (torch.arange(_S, device="cuda") - pos).to(torch.int32)
    ke = torch.arange(1, _S + 1, device="cuda", dtype=torch.int32)
    return x, grad, (angles.cos().bfloat16(), angles.sin().bfloat16()), ks, ke


def _cp_worker(rank: int, world: int, port: int, backend: str, results) -> None:
    import torch.distributed as dist

    from arctic_platform.model.implementations.glm52.models.glm_moe_dsa.sparse_mla_attention import GlmMoeDsaAttention
    from arctic_platform.model.implementations.glm52.models.glm_moe_dsa.sparse_mla_attention import (
        SparseMlaAttentionArgs,
    )
    from arctic_platform.model.implementations.glm52.models.kernels.sparse_mla_flashmla import set_sparse_mla_backend

    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=world)
    set_sparse_mla_backend(backend)
    args = SparseMlaAttentionArgs(
        hidden_size=1024,
        num_attention_heads=16,
        kv_lora_rank=512,
        q_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        qk_head_dim=192,
        v_head_dim=128,
        attention_bias=False,
        rms_norm_eps=1e-6,
        index_n_heads=8,
        index_head_dim=128,
        index_topk=256,
    )
    set_seed(0)
    attn = GlmMoeDsaAttention(args).cuda().bfloat16()
    x, grad, pe, ks, ke = _inputs(args.hidden_size)

    x_ref = x.clone().requires_grad_(True)
    out_ref, _ = attn(x_ref, pe, ks, ke)
    out_ref.backward(grad)
    ref_grads = {n: p.grad.float().clone() for n, p in attn.named_parameters() if p.grad is not None}
    attn.zero_grad(set_to_none=True)

    local = slice(rank * _S // world, (rank + 1) * _S // world)
    x_loc = x[:, local].clone().requires_grad_(True)
    out, _ = attn(x_loc, pe, ks, ke, cp=(dist.group.WORLD, rank, world))
    out.backward(grad[:, local])

    def rel(a, b):
        return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-12)).item()

    worst = 0.0
    for name, param in attn.named_parameters():
        if param.grad is not None:
            summed = param.grad.float().clone()
            dist.all_reduce(summed)
            worst = max(worst, rel(summed, ref_grads[name]))
    results[rank] = (torch.equal(out, out_ref[:, local]), rel(x_loc.grad, x_ref.grad[:, local]), worst)
    dist.destroy_process_group()


@require_torch_multi_gpu
@pytest.mark.parametrize("backend", ["tilelang", "flashmla"])
def test_context_parallel_attention_matches_full_sequence(backend):
    port = reserve_free_port(get_unique_port_number() + 1, span=7)
    with mp.Manager() as manager:
        results = manager.dict()
        mp.spawn(_cp_worker, args=(2, port, backend, results), nprocs=2, join=True)
        results = dict(results)
    for rank, (out_equal, dx_rel, param_rel) in results.items():
        # Local query rows read the same gathered KV through the same kernel: bit-identical.
        assert out_equal, f"rank {rank}: context-parallel output differs from the full sequence"
        # Gradients differ only in BF16 accumulation order (atomic dKV adds, reduce-scatter of gathered keys).
        assert dx_rel < 1e-2 and param_rel < 1e-2, f"rank {rank}: dx {dx_rel:.2e}, params {param_rel:.2e}"


@require_torch_gpu
def test_chunked_rectangular_indexer_selects_the_same_keys():
    from arctic_platform.model.implementations.glm52.models.kernels import fp8_indexer as indexer

    set_seed(1)
    S, H, D, topk = 4096, 8, 128, 512
    q = torch.randn(S, H, D, device="cuda").bfloat16()
    k = torch.randn(S, D, device="cuda").bfloat16()
    w = torch.randn(S, H, device="cuda").bfloat16()
    pos = torch.cat([torch.arange(1000), torch.arange(S - 1000)]).cuda()
    ks = (torch.arange(S, device="cuda") - pos).to(torch.int32)
    ke = torch.arange(1, S + 1, device="cuda", dtype=torch.int32)
    whole = indexer.fp8_indexer(q, k, w, ks, ke, topk)
    budget = indexer._SCORE_BUDGET_BYTES
    try:
        indexer._SCORE_BUDGET_BYTES = 4 * S * 256  # 256-row chunks
        chunked = indexer.fp8_indexer(q, k, w, ks, ke, topk)
    finally:
        indexer._SCORE_BUDGET_BYTES = budget
    shard = slice(S // 2, S)
    rectangular = indexer.fp8_indexer(q[shard], k, w[shard], ks[shard], ke[shard], topk)

    def as_sets(t):
        return t.sort(dim=-1).values

    # top-k order may differ between equal scores; the selected key sets must not.
    assert torch.equal(as_sets(chunked), as_sets(whole))
    assert torch.equal(as_sets(rectangular), as_sets(whole[shard]))
    # Early rows have fewer than ``topk`` selectable keys; their remaining slots hold the sentinel.
    assert int(whole[:topk].max()) == S


@require_torch_gpu
def test_fast_indexer_path_selects_the_portable_paths_keys(monkeypatch):
    """DeepGEMM scores + FlashInfer top-k (when installed) pick the keys of Triton scores + torch.topk."""
    from arctic_platform.model.implementations.glm52.models.kernels import fp8_indexer as indexer

    set_seed(2)
    S, H, D, topk = 5000, 32, 128, 2048
    q = torch.randn(S, H, D, device="cuda").bfloat16()
    k = torch.randn(S, D, device="cuda").bfloat16()
    w = torch.randn(S, H, device="cuda").bfloat16()
    pos = torch.cat([torch.arange(1700), torch.arange(S - 1700)]).cuda()
    ks = (torch.arange(S, device="cuda") - pos).to(torch.int32)
    ke = torch.arange(1, S + 1, device="cuda", dtype=torch.int32)
    shard = slice(1234, S)  # an unaligned query range, as under context parallelism
    fast = indexer.fp8_indexer(q[shard], k, w[shard], ks[shard], ke[shard], topk)
    monkeypatch.setattr(indexer, "_deep_gemm_mqa_logits", lambda capability_major: None)
    monkeypatch.setattr(indexer, "_flashinfer_top_k", None)
    portable = indexer.fp8_indexer(q[shard], k, w[shard], ks[shard], ke[shard], topk)

    assert torch.equal((fast == S).sum(-1), (portable == S).sum(-1))
    # Scores agree to ~1e-7, so only near-ties at the top-k boundary may swap.
    rows_equal = (fast.sort(-1).values == portable.sort(-1).values).all(-1).float().mean().item()
    assert rows_equal > 0.995
