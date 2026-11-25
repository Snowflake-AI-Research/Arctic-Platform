"""Interpret vLLM's expansion kernels on CPU with two emission orders for one selection."""

from types import SimpleNamespace

import pytest
import torch
from triton.runtime.interpreter import InterpretedFunction
from vllm.models.glm5next.nvidia import sparse_indexer
from vllm.models.qwen4_exp.nvidia.ops import qsa_indexer

from arctic_platform.inference.vllm.required_patches import apply_required_vllm_patches


def selections():
    # The long row drops pool 1 of 513; the short row keeps its only complete pool.
    chosen = torch.cat([torch.tensor([0]), torch.arange(2, 513)])
    short = torch.full_like(chosen, -1)
    short[0] = 0
    for order in (chosen.flip([0]), chosen.roll(1)):
        yield torch.stack([order, short])


def expected_tokens():
    # Both rows have a partial final pool: two tokens on the long row and one on the short row.
    return torch.cat([torch.arange(4), torch.arange(8, 2054), torch.tensor([-1])]).int()


def interpret(monkeypatch, module, name):
    monkeypatch.setattr(module, name, InterpretedFunction(getattr(module, name).fn))


def glm_metadata(decode):
    # Rows at positions 2053 and 4: one prefill chunk, or two one-token decodes.
    chunk = SimpleNamespace(
        token_start=0,
        token_end=2,
        skip_kv_gather=True,
        local_cu_seq_lens=torch.tensor([0, 514]),
        local_total_seq_lens=514,
        max_local_total_seq_lens=514,
        max_seqlen_k=514,
        cu_seqlen_ks=torch.zeros(2, dtype=torch.int32),
        cu_seqlen_ke=torch.tensor([513, 1], dtype=torch.int32),
    )
    decodes = SimpleNamespace(
        decode_lens=torch.ones(2, dtype=torch.int32),
        requires_padding=False,
        seq_lens=torch.tensor([[513], [1]]),
        block_table=None,
        schedule_metadata=None,
        indices=None,
    )
    return SimpleNamespace(
        slot_mapping=torch.zeros(2),
        max_seq_len=2054,
        num_decodes=2 * decode,
        num_decode_tokens=2 * decode,
        num_prefills=2 * (not decode),
        decode=decodes if decode else None,
        prefill=None if decode else SimpleNamespace(max_prefill_seq_len=2054, chunks=[chunk]),
    )


@pytest.mark.parametrize("dcp", [1, 8], ids=["dcp1", "dcp8"])
@pytest.mark.parametrize("decode", [False, True], ids=["prefill", "decode"])
def test_glm_pool_order(monkeypatch, decode, dcp):
    apply_required_vllm_patches()
    interpret(monkeypatch, sparse_indexer.kpool_ops, "_expand_pools_and_append_tail_kernel")
    emitted = []

    def emit(dst):
        dst.copy_(emitted[-1])

    # GPU kernels: zero scores; the local top-k emits the pool list, and at dcp > 1 the cross-rank merge re-emits it.
    monkeypatch.setattr(sparse_indexer, "fp8_fp4_mqa_logits", lambda *args: torch.zeros(2, 514))
    monkeypatch.setattr(sparse_indexer, "fp8_fp4_paged_mqa_logits", lambda *args, **kwargs: torch.zeros(2, 514))
    monkeypatch.setattr(torch.ops._C, "top_k_per_row_prefill", lambda _, lo, hi, dst, *a: emit(dst), raising=False)
    monkeypatch.setattr(sparse_indexer, "get_indexer_topk", lambda backend: lambda logits, lens, n, dst, *a: emit(dst))
    monkeypatch.setattr(sparse_indexer, "_merge_dcp_topk_global", lambda logits, dst, *a, **kw: dcp > 1 and emit(dst))
    workspace = SimpleNamespace(get_simultaneous=lambda *specs: [torch.empty(s, dtype=d) for s, d in specs])
    monkeypatch.setattr(sparse_indexer, "current_workspace_manager", lambda: workspace)
    monkeypatch.setattr(sparse_indexer, "DeepseekV32IndexerMetadata", SimpleNamespace)
    context = SimpleNamespace(attn_metadata={"indexer": glm_metadata(decode)})
    monkeypatch.setattr(sparse_indexer, "get_forward_context", lambda: context)
    short = torch.full((2051,), -1, dtype=torch.int32)
    short[:4] = torch.arange(4)
    short[2048] = 4
    for selected in selections():
        emitted.append(selected.int())
        buffer = torch.zeros(2, 2051, dtype=torch.int32)
        sparse_indexer.sparse_attn_indexer_kpool(
            hidden_states=torch.empty(2, 1),
            k_cache_prefix="indexer",
            kv_cache=torch.empty(1, 1, 1),
            q_quant=torch.empty(2, 1, 1),
            q_scale=None,
            k=None,
            weights=torch.empty(2, 1),
            quant_block_size=128,
            scale_fmt="ue8m0",
            topk_tokens=2048,
            head_dim=128,
            max_pool_len=514,
            total_seq_lens=514,
            topk_indices_buffer=buffer,
            skip_k_cache_insert=True,
            index_kpool=4,
            positions=torch.tensor([2053, 4]),
            dcp_world_size=dcp,
            cp_kv_cache_interleave_size=4,
        )
        torch.testing.assert_close(buffer, torch.stack([expected_tokens(), short]))


@pytest.mark.parametrize("decode", [False, True], ids=["prefill", "decode"])
def test_qsa_block_order(monkeypatch, decode):
    apply_required_vllm_patches()
    ops = qsa_indexer
    interpret(monkeypatch, ops, "_expand_qsa_indices_kernel")
    score_kernel = ops._qsa_mqa_paged_uniform_kernel if decode else ops._qsa_mqa_paged_prefill_kernel
    monkeypatch.setattr(score_kernel, "run", lambda *args, **kwargs: None)
    monkeypatch.setattr(ops.envs, "VLLM_SPARSE_INDEXER_MAX_LOGITS_MB", 1)
    monkeypatch.setattr(ops.current_platform, "has_device_capability", lambda cap: decode)
    monkeypatch.setattr(ops.current_platform, "is_device_capability_family", lambda cap: False)
    orders = selections()

    def emit(logits, lengths, dst, workspace, k, width):
        dst.copy_(next(orders))

    monkeypatch.setattr(torch.ops._C, "cooperative_topk" if decode else "persistent_topk", emit, raising=False)
    positions = torch.tensor([2053, 4], dtype=torch.int32)
    visible = (positions + 1) // 4
    q = torch.empty(2, 1, 64)
    cache = torch.empty(33, 16, 1, 64)
    pages = torch.arange(33, dtype=torch.int32).expand(2, -1)
    backing = torch.full((4, 512), -99, dtype=torch.int32)
    blocks = backing[1:3]
    out = torch.empty(2, 2052, dtype=torch.int32)
    short = torch.full((2051,), -1, dtype=torch.int32)
    short[:5] = torch.arange(5)
    for _ in range(2):
        if decode:
            ops.qsa_select_paged_decode(q, cache, pages, visible, 2048, 4, 1, blocks)
        else:
            starts = torch.tensor([0, 1, 2], dtype=torch.int32)
            ops.qsa_select_paged_prefill(q, cache, pages, starts, visible, 2048, 4, 1, blocks, 2054)
        ops.expand_qsa_block_indices(blocks, positions, visible, 4, 2048, out)
        torch.testing.assert_close(out[:, :-1], torch.stack([expected_tokens(), short]))
        torch.testing.assert_close(out[:, -1], torch.tensor([2050, 5], dtype=torch.int32))
        assert (backing[[0, 3]] == -99).all()
