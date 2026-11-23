"""Interpret vLLM's expansion kernels on CPU with two emission orders for one selection."""

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


def test_glm_pool_order(monkeypatch):
    apply_required_vllm_patches()
    ops = sparse_indexer.kpool_ops
    interpret(monkeypatch, ops, "_expand_pools_and_append_tail_kernel")
    seq_lens = torch.tensor([2054, 5], dtype=torch.int32)
    short = torch.full((2051,), -1, dtype=torch.int32)
    short[:4] = torch.arange(4)
    short[2048] = 4
    expected = torch.stack([expected_tokens(), short])
    for selected in selections():
        torch.testing.assert_close(ops.expand_pools_and_append_tail(selected, seq_lens, 4), expected)


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
