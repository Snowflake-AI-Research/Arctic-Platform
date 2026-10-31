import sys
import types

import pytest
import torch

from arctic_platform.inference.vllm import sparse_topk_order

# GLM-5.3-Flash and Qwen3.8-Flash-Next both select 2048 tokens as 512 groups of 4.
_POOL = 4
_GROUPS = 512


def _expand_pools_and_append_tail(pool_ids, seq_lens, pool_size):
    # Torch copy of vLLM 0.32 `_expand_pools_and_append_tail_kernel` (glm5next/nvidia/ops/kpool_compress.py).
    rows, groups = pool_ids.shape
    history = (pool_ids.unsqueeze(-1) * pool_size + torch.arange(pool_size)).reshape(rows, groups * pool_size)
    history = torch.where(pool_ids.repeat_interleave(pool_size, dim=1) >= 0, history, -1)
    tail_start = seq_lens // pool_size * pool_size
    offsets = torch.arange(pool_size - 1)
    tail = torch.where(offsets < (seq_lens - tail_start)[:, None], tail_start[:, None] + offsets, -1)
    return torch.cat([history, tail], dim=1).to(torch.int32)


_QSA_SOURCE = '''
def _topk(logits, visible_blocks, token_topk, compress_ratio, block_indices, topk_workspace):
    # Stand-in for cooperative_topk / persistent_topk: the right set, in the order atomic slots hand out.
    k = token_topk // compress_ratio
    for row in range(logits.shape[0]):
        visible = int(visible_blocks[row])
        if visible <= k:
            block_indices[row] = torch.where(torch.arange(k) < visible, torch.arange(k), -1)
        else:
            chosen = logits[row, :visible].topk(k).indices
            block_indices[row] = chosen[torch.randperm(k, generator=topk_workspace)]


def qsa_select(logits, visible_blocks, token_topk, compress_ratio, block_indices, generator):
    # Like qsa_select_paged_decode / _prefill: resolves `_topk` through the module globals.
    _topk(logits, visible_blocks, token_topk, compress_ratio, block_indices, generator)
'''


def _sparse_attn_indexer(hidden_states, scores, generator, topk_indices_buffer):
    """Stand-in for vLLM's DSA indexer: writes each token's top-k into the shared buffer in atomic-slot order."""
    width = topk_indices_buffer.shape[1]
    topk_indices_buffer[: hidden_states.shape[0]] = -1
    for row in range(hidden_states.shape[0]):
        visible = int(scores[row].isfinite().sum())
        if visible <= width:
            topk_indices_buffer[row, :visible] = torch.arange(visible)
        else:
            chosen = scores[row].topk(width).indices
            topk_indices_buffer[row] = chosen[torch.randperm(width, generator=generator)]
    return topk_indices_buffer


@pytest.fixture
def vllm_ops(monkeypatch):
    """Stub vLLM leaf modules the patch imports, without vLLM or a GPU."""
    kpool = types.ModuleType("kpool_compress")
    kpool.expand_pools_and_append_tail = _expand_pools_and_append_tail
    qsa = types.ModuleType("qsa_indexer")
    qsa.torch = torch
    exec(_QSA_SOURCE, qsa.__dict__)
    dsa = types.ModuleType("attention")
    dsa.sparse_attn_indexer = _sparse_attn_indexer
    made = {}
    for name, leaf in {
        "vllm.models.deepseek_v32.attention": dsa,
        "vllm.models.glm5next.nvidia.ops.kpool_compress": kpool,
        "vllm.models.qwen4_exp.nvidia.ops.qsa_indexer": qsa,
    }.items():
        parts = name.split(".")
        for i in range(1, len(parts) + 1):
            prefix = ".".join(parts[:i])
            if prefix not in made:
                made[prefix] = leaf if i == len(parts) else types.ModuleType(prefix)
                monkeypatch.setitem(sys.modules, prefix, made[prefix])
            if i > 1:
                setattr(made[".".join(parts[: i - 1])], parts[i - 1], made[prefix])
    monkeypatch.setattr(sparse_topk_order, "_PATCHED", False)
    sparse_topk_order.apply_sparse_topk_order_fix()
    return kpool, qsa, dsa


def _assert_sorted_rows(ids):
    for row in ids.tolist():
        valid = [i for i in row if i >= 0]
        assert valid == sorted(valid), "selected ids are not ascending"


def test_glm_pool_order_no_longer_depends_on_topk_slot_order(vllm_ops):
    kpool, _, _ = vllm_ops
    seq_lens = torch.tensor([4099, 1003], dtype=torch.int32)  # 1024 pools (sparse) and 250 pools (dense)
    scores = torch.randn(1024, generator=torch.Generator().manual_seed(0))
    chosen = scores.topk(_GROUPS).indices
    dense = torch.where(torch.arange(_GROUPS) < 250, torch.arange(_GROUPS), -1)  # vLLM's ordered short-row path
    runs = []
    for seed in (1, 2):
        order = torch.randperm(_GROUPS, generator=torch.Generator().manual_seed(seed))
        pool_ids = torch.stack([chosen[order], dense]).to(torch.int64)
        runs.append((pool_ids, kpool.expand_pools_and_append_tail(pool_ids, seq_lens, _POOL)))

    (ids_a, out_a), (ids_b, out_b) = runs
    assert not torch.equal(ids_a, ids_b)
    assert torch.equal(out_a, out_b), "same selected set, different key order handed to attention"
    _assert_sorted_rows(out_a)
    unpatched = _expand_pools_and_append_tail(ids_a, seq_lens, _POOL)
    for row in range(2):
        assert sorted(out_a[row].tolist()) == sorted(unpatched[row].tolist())
    assert torch.equal(out_a[1], unpatched[1]), "an already ordered dense row must not change"


def test_qsa_block_order_no_longer_depends_on_topk_slot_order(vllm_ops):
    _, qsa, _ = vllm_ops
    visible = torch.tensor([1024, 300, 513], dtype=torch.int32)
    logits = torch.randn(3, 1024, generator=torch.Generator().manual_seed(0))
    runs = []
    for seed in (1, 2):
        backing = torch.empty(5, _GROUPS, dtype=torch.int32)
        qsa.qsa_select(logits, visible, _GROUPS * _POOL, _POOL, backing[1:4], torch.Generator().manual_seed(seed))
        runs.append(backing[1:4])

    assert torch.equal(runs[0], runs[1]), "same selected set, different key order handed to attention"
    _assert_sorted_rows(runs[0])
    for row in range(3):
        n_valid = min(int(visible[row]), _GROUPS)
        assert bool((runs[0][row, :n_valid] >= 0).all()), "valid ids must lead: the expand kernel reads that prefix"
        assert bool((runs[0][row, n_valid:] == -1).all())
        visible_logits = logits[row, : int(visible[row])]
        expected = torch.arange(n_valid) if n_valid < _GROUPS else visible_logits.topk(_GROUPS).indices
        assert set(runs[0][row, :n_valid].tolist()) == set(expected.tolist())


def test_dsa_token_order_no_longer_depends_on_topk_slot_order(vllm_ops):
    _, _, dsa = vllm_ops
    width = 2048
    scores = torch.randn(3, 4096, generator=torch.Generator().manual_seed(0))
    scores[1, 1000:] = float("-inf")  # a short row: vLLM writes its keys in order
    hidden = torch.empty(3, 8)
    runs = []
    for seed in (1, 2):
        backing = torch.full((5, width), 7, dtype=torch.int32)
        out = dsa.sparse_attn_indexer(hidden, scores, torch.Generator().manual_seed(seed), topk_indices_buffer=backing)
        assert out is backing
        runs.append(backing)

    assert torch.equal(runs[0], runs[1]), "same selected set, different key order handed to attention"
    _assert_sorted_rows(runs[0][:3])
    assert set(runs[0][0].tolist()) == set(scores[0].topk(width).indices.tolist())
    assert runs[0][1, :1000].tolist() == list(range(1000)) and bool((runs[0][1, 1000:] == -1).all())
    assert bool((runs[0][3:] == 7).all()), "rows past this batch's tokens must not change"
