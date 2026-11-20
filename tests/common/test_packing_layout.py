# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""PackedLayout: gather pack/unpack, sharing-off vs pack_microbatch, prefix attention, lifetime, gates.

Attention numerics run on CPU through ``flash_attn_varlen_reference`` (pure torch, bottom-right causal).
Nothing here imports the model stack; ``tests/model/test_packed_layout_continuation.py`` drives the
Arctic-Platform and Hugging Face attention modules.
"""

import sys

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import CheckpointError
from torch.utils.checkpoint import checkpoint

from arctic_platform.common.packing import IGNORE_INDEX
from arctic_platform.common.packing import PackedLayout
from arctic_platform.common.packing import cu_seqlens_from_position_ids
from arctic_platform.common.packing import pack_microbatch
from arctic_platform.common.packing import pack_with_layout
from arctic_platform.common.packing import require_identical_prefixes
from arctic_platform.common.packing import require_no_prefix_sharing
from arctic_platform.common.packing import require_prefix_continuation
from arctic_platform.common.packing import require_token_mean_if_sharing
from arctic_platform.common.packing import shared_prefix_length
from arctic_platform.common.packing import skip_share_if_router_disagree
from arctic_platform.common.packing import using_layout
from arctic_platform.common.packing.layout import LAYOUT_ATTR
from arctic_platform.common.packing.layout import active_layout_requires_token_mean
from arctic_platform.common.packing.layout import get_active_packed_layout
from arctic_platform.common.packing.layout import get_stashed_layout
from arctic_platform.common.packing.layout import prefix_varlen_args
from arctic_platform.common.packing.layout import unshared_groups
from arctic_platform.testing_utils import flash_attn_varlen_reference
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal

# float64 attention on CPU: different segment lengths change only the softmax reduction order.
F64_ATOL = 1e-12


class FlashAttention(nn.Module):
    """Minimal varlen attention under the name the mixer allowlist accepts for Arctic-Platform flash attention."""

    def __init__(self, hidden=16, heads=4, kv_heads=2, sliding_window=None):
        super().__init__()
        self.heads, self.kv_heads, self.head_dim = heads, kv_heads, hidden // heads
        self.q = nn.Linear(hidden, heads * self.head_dim, bias=False)
        self.kv = nn.Linear(hidden, 2 * kv_heads * self.head_dim, bias=False)
        self.sliding_window = sliding_window

    def forward(self, hidden, cu_seqlens=None, max_seqlen=None):
        q = self.q(hidden).view(-1, self.heads, self.head_dim)
        k, v = self.kv(hidden).view(-1, 2 * self.kv_heads, self.head_dim).chunk(2, dim=1)
        k, v, cu_q, cu_k, _, _ = prefix_varlen_args(k, v, cu_seqlens, max_seqlen, get_stashed_layout(self))
        window = (-1, -1) if self.sliding_window is None else (self.sliding_window - 1, 0)
        return flash_attn_varlen_reference(q, k, v, cu_q, cu_k, causal=True, window_size=window).flatten(1)


class Qwen3MoeSparseMoeBlock(nn.Module):
    """Stand-in whose class name marks the model as MoE."""


class MoeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = FlashAttention()
        self.moe = Qwen3MoeSparseMoeBlock()


def _rows(lengths, pad=None, left_pad=0):
    width = (pad or max(lengths)) + left_pad
    batch_size = len(lengths)
    ids = torch.zeros(batch_size, width, dtype=torch.long)
    pos = torch.zeros(batch_size, width, dtype=torch.long)
    labels = torch.full((batch_size, width), IGNORE_INDEX, dtype=torch.long)
    mask = torch.zeros(batch_size, width, dtype=torch.long)
    token = 10
    for row, length in enumerate(lengths):
        start = left_pad
        end = start + length
        mask[row, start:end] = 1
        pos[row, start:end] = torch.arange(length)
        ids[row, start:end] = torch.arange(token, token + length)
        labels[row, start:end] = torch.arange(length)
        token += length
    return {
        "input_ids": ids,
        "position_ids": pos,
        "labels": labels,
        "attention_mask": mask,
    }


def _shared_layout(lengths=(4, 4), shared=3, **build_kwargs):
    batch = _rows(list(lengths))
    for row in range(1, len(lengths)):
        batch["input_ids"][row, :shared] = batch["input_ids"][0, :shared]
    layout = PackedLayout.build(
        batch["attention_mask"],
        [list(range(len(lengths)))],
        [shared],
        batch["position_ids"],
        input_ids=batch["input_ids"],
        **build_kwargs,
    )
    return batch, layout


def test_sharing_off_matches_pack_microbatch():
    batch = _rows([3, 2, 4], pad=5)
    groups, shared = unshared_groups(3)
    layout = PackedLayout.build(batch["attention_mask"], groups, shared, batch["position_ids"])
    packed_layout = pack_with_layout(batch, layout)
    packed_base, metadata = pack_microbatch(batch)
    torch_assert_equal(packed_layout["input_ids"], packed_base["input_ids"])
    torch_assert_equal(packed_layout["position_ids"], packed_base["position_ids"])
    torch_assert_equal(layout.cu_seqlens, metadata.cu_seqlens)


def test_left_padding_is_dropped_on_pack():
    batch = _rows([3, 2], pad=3, left_pad=2)
    groups, shared = unshared_groups(2)
    layout = PackedLayout.build(batch["attention_mask"], groups, shared, batch["position_ids"])
    packed = pack_with_layout(batch, layout)
    assert packed["input_ids"].shape[-1] == 5
    assert (packed["input_ids"] != 0).all()


def test_shared_prefix_is_written_once():
    batch, layout = _shared_layout()
    packed = pack_with_layout(batch, layout)
    # root (3) + two continuations (1 each)
    assert packed["input_ids"].shape[-1] == 5
    torch_assert_equal(packed["input_ids"][0, :3], batch["input_ids"][0, :3])
    assert packed["input_ids"][0, 3].item() == batch["input_ids"][0, 3].item()
    assert packed["input_ids"][0, 4].item() == batch["input_ids"][1, 3].item()
    assert layout.num_roots == 1
    assert layout.has_sharing
    torch_assert_equal(layout.cu_seqlens, torch.tensor([0, 3, 4, 5], dtype=torch.int32))
    torch_assert_equal(layout.cu_seqlens_k, torch.tensor([0, 3, 7, 11], dtype=torch.int32))


def test_unpack_restores_shared_prefix_to_every_row():
    batch, layout = _shared_layout()
    restored = layout.unpack(layout.pack(batch["input_ids"]))
    torch_assert_equal(restored, batch["input_ids"])


def test_unpack_accepts_batch_dim_and_single_token():
    batch, layout = _shared_layout()
    packed = layout.pack(batch["input_ids"])
    torch_assert_equal(layout.unpack(packed.unsqueeze(0)), layout.unpack(packed))
    with pytest.raises(ValueError, match="T=5"):
        layout.unpack(packed[:4])

    one = PackedLayout.build(torch.ones(1, 1, dtype=torch.long), [[0]], [0])
    logits = torch.arange(6.0).view(1, 6)  # [T=1, V=6], not [1, T=6]
    torch_assert_equal(one.unpack(logits), logits.view(1, 1, 6))


def test_pack_gathers_from_one_source_row():
    x = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], requires_grad=True)
    mask = torch.ones(2, 3, dtype=torch.long)
    ids = torch.tensor([[1, 2, 3], [1, 2, 9]])
    layout = PackedLayout.build(mask, [[0, 1]], [2], torch.arange(3).expand(2, -1), input_ids=ids)
    packed = layout.pack(x)
    packed.sum().backward()
    torch_assert_equal(x.grad, torch.tensor([[1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]))


def test_identical_prefix_is_required():
    batch = _rows([4, 4], pad=4)
    with pytest.raises(ValueError, match="not token-identical"):
        PackedLayout.build(batch["attention_mask"], [[0, 1]], [3], batch["position_ids"], input_ids=batch["input_ids"])
    with pytest.raises(ValueError, match="not token-identical"):
        require_identical_prefixes(batch["input_ids"], [[0, 1]], [1], batch["attention_mask"])


def test_router_disagreement_skips_share():
    routed = torch.tensor([[1, 2, 3], [1, 9, 3]])
    mask = torch.ones(2, 3, dtype=torch.long)
    assert skip_share_if_router_disagree([[0, 1]], [2], routed, mask) == [0]
    assert skip_share_if_router_disagree([[0, 1]], [1], routed, mask) == [1]


def test_shared_prefix_length_rounds_down_to_chunk():
    assert shared_prefix_length(70, 64) == 64
    assert shared_prefix_length(63, 64) == 0
    assert shared_prefix_length(70, None) == 70


def test_prompt_len_minus_one_keeps_last_prompt_token_in_each_continuation():
    prompt_len = 4
    batch = _rows([6, 7])
    batch["input_ids"][1, :prompt_len] = batch["input_ids"][0, :prompt_len]
    shared = prompt_len - 1
    layout = PackedLayout.build(
        batch["attention_mask"], [[0, 1]], [shared], batch["position_ids"], input_ids=batch["input_ids"]
    )
    torch_assert_equal(layout.cu_seqlens, torch.tensor([0, 3, 6, 10], dtype=torch.int32))
    # Each continuation starts at the last prompt token, whose logits predict the first response token.
    torch_assert_equal(layout.position_ids, torch.tensor([0, 1, 2, 3, 4, 5, 3, 4, 5, 6]))
    torch_assert_equal(layout.src_col[layout.cu_seqlens[1:3].long()], torch.tensor([shared, shared]))
    torch_assert_equal(layout.row_index[:, :shared], torch.tensor([[0, 1, 2], [0, 1, 2]]))
    torch_assert_equal(layout.row_index[:, shared], torch.tensor([3, 6]))


def test_right_padding_sharing_uses_valid_tokens():
    batch = _rows([5, 3], pad=6)
    batch["input_ids"][1, :2] = batch["input_ids"][0, :2]
    layout = PackedLayout.build(
        batch["attention_mask"], [[0, 1]], [2], batch["position_ids"], input_ids=batch["input_ids"]
    )
    torch_assert_equal(layout.cu_seqlens, torch.tensor([0, 2, 5, 6], dtype=torch.int32))
    assert (layout.row_index[0, 5:] == -1).all() and (layout.row_index[1, 3:] == -1).all()
    restored = layout.unpack(layout.pack(batch["input_ids"]))
    torch_assert_equal(restored, batch["input_ids"])


def test_left_padding_sharing_uses_valid_tokens():
    batch = _rows([4, 4], pad=4, left_pad=2)
    batch["input_ids"][1, 2:5] = batch["input_ids"][0, 2:5]
    layout = PackedLayout.build(
        batch["attention_mask"], [[0, 1]], [3], batch["position_ids"], input_ids=batch["input_ids"]
    )
    assert (layout.src_col[:3] == torch.tensor([2, 3, 4])).all()
    restored = layout.unpack(layout.pack(batch["input_ids"]))
    torch_assert_equal(restored, batch["input_ids"])


def test_left_padding_does_not_treat_pad_as_prefix():
    batch = _rows([4, 4], pad=4, left_pad=2)
    with pytest.raises(ValueError, match="not token-identical"):
        PackedLayout.build(batch["attention_mask"], [[0, 1]], [3], batch["position_ids"], input_ids=batch["input_ids"])


def test_two_groups_get_one_root_each():
    mask = torch.ones(4, 3, dtype=torch.long)
    pos = torch.arange(3).expand(4, -1)
    ids = torch.arange(3).expand(4, -1).contiguous()
    layout = PackedLayout.build(mask, [[0, 1], [2, 3]], [2, 2], pos, input_ids=ids)
    assert layout.num_roots == 2
    assert layout.num_segments == 6
    assert layout.parent.tolist() == [-1, -1, 0, 0, 1, 1]


def test_require_unweighted_shared():
    mask = torch.ones(2, 4, dtype=torch.long)
    ids = torch.tensor([[1, 2, 3, 4], [1, 2, 8, 9]])
    layout = PackedLayout.build(mask, [[0, 1]], [2], torch.arange(4).expand(2, -1), input_ids=ids)
    weights = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.float)
    layout.require_unweighted_shared(weights)
    weights[1, 0] = 1
    with pytest.raises(ValueError, match="shared prefix"):
        layout.require_unweighted_shared(weights)


def test_router_replay_must_be_explicit_for_moe():
    _, unstated = _shared_layout()
    assert unstated.router_replay is None
    _, opted_out = _shared_layout(router_replay=False)
    assert opted_out.router_replay is False
    _, replayed = _shared_layout(routed_experts=torch.zeros(2, 4, 2, dtype=torch.long))
    assert replayed.router_replay is True and replayed.has_sharing
    with pytest.raises(ValueError, match="router_replay=False"):
        _shared_layout(routed_experts=torch.zeros(2, 4, 2, dtype=torch.long), router_replay=False)
    with pytest.raises(ValueError, match="needs routed_experts"):
        _shared_layout(router_replay=True)

    with pytest.raises(ValueError, match="MoE model"):
        with using_layout(MoeModel(), unstated):
            pass
    for layout in (opted_out, replayed):
        with using_layout(MoeModel(), layout):
            pass
    with using_layout(FlashAttention(), unstated):
        pass


def test_token_mean_required_when_sharing():
    require_token_mean_if_sharing(False, "seq-mean-token-sum")
    with pytest.raises(ValueError, match="token-mean"):
        require_token_mean_if_sharing(True, "seq-mean-token-sum")
    _, layout = _shared_layout()
    with using_layout(FlashAttention(), layout):
        active_layout_requires_token_mean("token-mean")
        with pytest.raises(ValueError, match="token-mean"):
            active_layout_requires_token_mean("seq-mean-token-sum")
    active_layout_requires_token_mean("seq-mean-token-sum")


def test_agg_loss_rejects_seq_mean_under_sharing_layout():
    # rl.processors must load before the verl module, which it imports back.
    from arctic_platform.rl.processors.functional import agg_loss
    from arctic_platform.integrations.verl.grpo_loss import agg_loss as verl_agg_loss

    loss_mat, loss_mask = torch.ones(2, 3), torch.ones(2, 3)
    with pytest.raises(ValueError, match="token-mean"):
        agg_loss(loss_mat, loss_mask, "seq-mean-token-sum", prefix_sharing=True)
    _, layout = _shared_layout()
    with using_layout(FlashAttention(), layout):
        for fn in (agg_loss, verl_agg_loss):
            torch_assert_equal(fn(loss_mat, loss_mask, "token-mean"), torch.tensor(1.0))
            with pytest.raises(ValueError, match="token-mean"):
                fn(loss_mat, loss_mask, "seq-mean-token-sum")
    torch_assert_equal(verl_agg_loss(loss_mat, loss_mask, "seq-mean-token-sum"), torch.tensor(3.0))


def test_cu_seqlens_from_position_ids_uses_active_layout():
    batch, layout = _shared_layout()
    packed = pack_with_layout(batch, layout)
    # Positions [0, 1, 2, 3, 3] carry no reset at either continuation; only the layout knows the segments.
    assert cu_seqlens_from_position_ids(packed["position_ids"]).tolist() != layout.cu_seqlens.tolist()
    with using_layout(FlashAttention(), layout):
        torch_assert_equal(cu_seqlens_from_position_ids(packed["position_ids"]), layout.cu_seqlens)
        with pytest.raises(ValueError, match="5 packed tokens"):
            cu_seqlens_from_position_ids(torch.zeros(1, 8, dtype=torch.long))
        with pytest.raises(ValueError, match="position_ids == 0"):
            require_no_prefix_sharing("helper")
    require_no_prefix_sharing("helper")


def test_require_prefix_continuation_rejects_unsupported_mixers():
    class Qwen3NextGatedDeltaNet(nn.Module):
        pass

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = FlashAttention()
            self.gdn = Qwen3NextGatedDeltaNet()

    require_prefix_continuation(FlashAttention())
    with pytest.raises(ValueError, match="Qwen3NextGatedDeltaNet"):
        require_prefix_continuation(Model())

    class Cfg:
        _attn_implementation = "flash_attention_2"

    class Qwen3Attention(nn.Module):
        def __init__(self, impl="flash_attention_2", sliding_window=None):
            super().__init__()
            self.config = Cfg()
            self.config._attn_implementation = impl
            self.sliding_window = sliding_window

    class Qwen2Attention(Qwen3Attention):
        pass

    require_prefix_continuation(Qwen3Attention())
    with pytest.raises(ValueError, match="needs flash"):
        require_prefix_continuation(Qwen3Attention(impl="sdpa"))
    with pytest.raises(ValueError, match="sliding window"):
        require_prefix_continuation(Qwen3Attention(sliding_window=4))
    with pytest.raises(ValueError, match="Qwen2Attention"):
        require_prefix_continuation(Qwen2Attention())
    with pytest.raises(ValueError, match="no sequence mixers"):
        require_prefix_continuation(nn.Linear(1, 1))


def test_using_layout_validates_before_stashing():
    _, layout = _shared_layout()
    model = nn.Linear(1, 1)
    with pytest.raises(ValueError, match="no sequence mixers"):
        with using_layout(model, layout):
            pass
    assert LAYOUT_ATTR not in model.__dict__
    assert get_active_packed_layout() is None
    groups, shared = unshared_groups(2)
    unshared = PackedLayout.build(torch.ones(2, 3, dtype=torch.long), groups, shared)
    with using_layout(model, unshared):
        assert get_stashed_layout(model) is unshared


def test_nested_using_layout_restores_previous():
    _, outer = _shared_layout()
    _, inner = _shared_layout(lengths=(5, 5))
    model, other = FlashAttention(), FlashAttention()
    with using_layout(model, outer):
        with using_layout(other, inner):
            assert get_stashed_layout(model) is outer
            assert get_stashed_layout(other.q) is inner
            assert get_active_packed_layout() is inner
            with using_layout(model, inner):
                assert get_stashed_layout(model.kv) is inner
            assert get_stashed_layout(model.kv) is outer
        assert LAYOUT_ATTR not in other.__dict__ and LAYOUT_ATTR not in other.q.__dict__
        assert get_active_packed_layout() is outer
    assert all(LAYOUT_ATTR not in module.__dict__ for module in model.modules())
    assert get_active_packed_layout() is None


def test_using_layout_restores_after_exception():
    _, outer = _shared_layout()
    _, inner = _shared_layout(lengths=(5, 5))
    model = FlashAttention()
    with using_layout(model, outer):
        with pytest.raises(RuntimeError, match="boom"):
            with using_layout(model, inner):
                raise RuntimeError("boom")
        assert get_stashed_layout(model) is outer
        assert get_active_packed_layout() is outer
    assert get_active_packed_layout() is None


def _attention_batch(left_pad_row=1, prefix=4, lengths=(7, 6, 5), hidden=16):
    """Group rows 0-1 share ``prefix`` tokens (row 1 left-padded); row 2 is a singleton root."""
    width = max(lengths) + 1
    mask = torch.zeros(len(lengths), width, dtype=torch.long)
    for row, length in enumerate(lengths):
        start = 1 if row == left_pad_row else 0
        mask[row, start : start + length] = 1
    x = torch.randn(len(lengths), width, hidden, dtype=torch.float64)
    ids = torch.randint(0, 100, (len(lengths), width))
    cols0 = mask[0].nonzero().flatten()[:prefix]
    cols1 = mask[1].nonzero().flatten()[:prefix]
    x[1, cols1] = x[0, cols0]
    ids[1, cols1] = ids[0, cols0]
    layout = PackedLayout.build(mask, [[0, 1], [2]], [prefix, 0], input_ids=ids)
    unshared = PackedLayout.build(mask, *unshared_groups(len(lengths)))
    return x, mask, layout, unshared


def _per_row(module, x, mask):
    """Each row on its own through ``module``; padding cells stay zero."""
    out = torch.zeros(*x.shape[:2], module.heads * module.head_dim, dtype=x.dtype)
    for row in range(x.shape[0]):
        cols = mask[row].nonzero().flatten()
        cu = torch.tensor([0, cols.numel()], dtype=torch.int32)
        out[row, cols] = module(x[row, cols], cu_seqlens=cu, max_seqlen=cols.numel())
    return out


def _continuation_weights(x, mask, layout):
    """Loss weights on cells owned by exactly one row (shared prefix unweighted)."""
    valid = layout.row_index >= 0
    counts = torch.bincount(layout.row_index[valid], minlength=layout.num_tokens)
    owned = torch.zeros_like(mask, dtype=torch.bool)
    owned[valid] = counts[layout.row_index[valid]] == 1
    return torch.randn(*x.shape[:2], 16, dtype=x.dtype) * owned[..., None]


def _grads(module, x, loss_fn):
    module.zero_grad()
    x = x.clone().requires_grad_(True)
    loss_fn(module, x).backward()
    return x.grad, {name: p.grad.clone() for name, p in module.named_parameters()}


def test_prefix_attention_matches_per_row_forward_and_backward():
    set_seed(0)
    x, mask, layout, _ = _attention_batch()
    module = FlashAttention().double()
    weights = _continuation_weights(x, mask, layout)

    def shared_loss(m, h):
        with using_layout(m, layout):
            out = layout.unpack(m(layout.pack(h)))
            return (out * weights).sum(), out

    def row_loss(m, h):
        out = _per_row(m, h, mask)
        return (out * weights).sum(), out

    (_, out_shared), (_, out_rows) = shared_loss(module, x), row_loss(module, x)
    valid = mask.bool()
    torch_assert_close(out_shared[valid], out_rows[valid], rtol=0, atol=F64_ATOL)

    x_grad, p_grad = _grads(module, x, lambda m, h: shared_loss(m, h)[0])
    x_grad_ref, p_grad_ref = _grads(module, x, lambda m, h: row_loss(m, h)[0])
    for name in p_grad:
        torch_assert_close(p_grad[name], p_grad_ref[name], rtol=0, atol=F64_ATOL, msg=name)
    # The root is gathered from row 0, so it holds the sum of both rows' prefix gradients; row 1's prefix gets none.
    root = (layout.row_index[0] >= 0) & (layout.row_index[0] < layout.root_tokens)
    root1 = (layout.row_index[1] >= 0) & (layout.row_index[1] < layout.root_tokens)
    torch_assert_close(x_grad[0, root], x_grad_ref[0, root] + x_grad_ref[1, root1], rtol=0, atol=F64_ATOL)
    torch_assert_equal(x_grad[1, root1], torch.zeros_like(x_grad[1, root1]))
    rest = valid.clone()
    rest[0, root], rest[1, root1] = False, False
    torch_assert_close(x_grad[rest], x_grad_ref[rest], rtol=0, atol=F64_ATOL)


def test_prefix_gather_disabled_does_not_match(monkeypatch):
    set_seed(0)
    x, mask, layout, _ = _attention_batch()
    module = FlashAttention().double()
    reference = _per_row(module, x, mask)

    def no_gather(k, v, cu_seqlens, max_seqlen, layout):
        cu = layout.cu_seqlens.to(torch.int32)
        return k, v, cu, cu, layout.max_len_q, layout.max_len_q

    monkeypatch.setattr(sys.modules[__name__], "prefix_varlen_args", no_gather)
    with using_layout(module, layout):
        broken = layout.unpack(module(layout.pack(x)))
    continuation = layout.row_index >= layout.root_tokens
    assert (broken[continuation] - reference[continuation]).abs().max() > 1e-3


def test_sliding_window_prefix_matches_per_row():
    set_seed(1)
    x, mask, layout, _ = _attention_batch()
    module = FlashAttention(sliding_window=3).double()
    with using_layout(module, layout):
        out = layout.unpack(module(layout.pack(x)))
    torch_assert_close(out[mask.bool()], _per_row(module, x, mask)[mask.bool()], rtol=0, atol=F64_ATOL)


def test_checkpoint_recompute_under_layout_matches_no_checkpoint():
    set_seed(2)
    x, mask, layout, _ = _attention_batch()
    module = FlashAttention().double()
    weights = _continuation_weights(x, mask, layout)

    segments = (layout.cu_seqlens, layout.max_len_q)

    def loss(m, h, use_checkpoint):
        with using_layout(m, layout):
            packed = layout.pack(h)
            out = checkpoint(m, packed, *segments, use_reentrant=False) if use_checkpoint else m(packed, *segments)
            loss = (layout.unpack(out) * weights).sum()
            loss.backward()
        return loss

    module.zero_grad()
    h = x.clone().requires_grad_(True)
    loss(module, h, use_checkpoint=False)
    expected = [h.grad] + [p.grad.clone() for p in module.parameters()]
    module.zero_grad()
    h = x.clone().requires_grad_(True)
    loss(module, h, use_checkpoint=True)
    for got, want in zip([h.grad] + [p.grad for p in module.parameters()], expected):
        torch_assert_equal(got, want)

    # Backward after the context exits recomputes without the gather; checkpoint refuses the mismatch.
    h = x.clone().requires_grad_(True)
    with using_layout(module, layout):
        out = checkpoint(module, layout.pack(h), *segments, use_reentrant=False)
    with pytest.raises(CheckpointError):
        (layout.unpack(out) * weights).sum().backward()


def test_share_off_layout_matches_pack_microbatch_attention():
    set_seed(3)
    x, mask, _, unshared = _attention_batch(left_pad_row=None)
    module = FlashAttention().double()
    positions = (mask.cumsum(1) - 1).clamp(min=0) * mask
    packed, metadata = pack_microbatch({"x": x, "attention_mask": mask, "position_ids": positions})
    with using_layout(module, unshared):
        out = module(unshared.pack(x))
    baseline = module(packed["x"][0], cu_seqlens=metadata.cu_seqlens, max_seqlen=int(metadata.cu_seqlens.diff().max()))
    torch_assert_equal(out, baseline)
