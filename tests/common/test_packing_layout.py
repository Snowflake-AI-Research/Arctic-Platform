# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""PackedLayout: gather pack/unpack, sharing-off vs pack_microbatch, mixer gate, token-mean."""

import pytest
import torch
from torch import nn

from arctic_platform.common.packing import IGNORE_INDEX
from arctic_platform.common.packing import PackedLayout
from arctic_platform.common.packing import cu_seqlens_from_position_ids
from arctic_platform.common.packing import pack_microbatch
from arctic_platform.common.packing import pack_with_layout
from arctic_platform.common.packing import require_prefix_continuation
from arctic_platform.common.packing import require_token_mean_if_sharing
from arctic_platform.common.packing import shared_prefix_length
from arctic_platform.common.packing import skip_share_if_router_disagree
from arctic_platform.common.packing import using_layout
from arctic_platform.common.packing.layout import active_layout_requires_token_mean
from arctic_platform.common.packing.layout import unshared_groups


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


def test_sharing_off_matches_pack_microbatch():
    batch = _rows([3, 2, 4], pad=5)
    groups, shared = unshared_groups(3)
    layout = PackedLayout.build(batch["attention_mask"], groups, shared, batch["position_ids"])
    packed_layout = pack_with_layout(batch, layout)
    packed_base, metadata = pack_microbatch(batch)
    torch.testing.assert_close(packed_layout["input_ids"], packed_base["input_ids"])
    torch.testing.assert_close(packed_layout["position_ids"], packed_base["position_ids"])
    torch.testing.assert_close(layout.cu_seqlens, metadata.cu_seqlens)


def test_left_padding_is_dropped_on_pack():
    batch = _rows([3, 2], pad=3, left_pad=2)
    groups, shared = unshared_groups(2)
    layout = PackedLayout.build(batch["attention_mask"], groups, shared, batch["position_ids"])
    packed = pack_with_layout(batch, layout)
    assert packed["input_ids"].shape[-1] == 5
    assert (packed["input_ids"] != 0).all()


def test_shared_prefix_is_written_once():
    batch = _rows([4, 4], pad=4)
    batch["input_ids"][1, :3] = batch["input_ids"][0, :3]
    layout = PackedLayout.build(
        batch["attention_mask"], [[0, 1]], [3], batch["position_ids"], input_ids=batch["input_ids"]
    )
    packed = pack_with_layout(batch, layout)
    # root (3) + two continuations (1 each)
    assert packed["input_ids"].shape[-1] == 5
    torch.testing.assert_close(packed["input_ids"][0, :3], batch["input_ids"][0, :3])
    assert packed["input_ids"][0, 3].item() == batch["input_ids"][0, 3].item()
    assert packed["input_ids"][0, 4].item() == batch["input_ids"][1, 3].item()
    assert layout.num_roots == 1
    assert layout.has_sharing
    torch.testing.assert_close(layout.cu_seqlens, torch.tensor([0, 3, 4, 5], dtype=torch.int32))
    torch.testing.assert_close(layout.cu_seqlens_k, torch.tensor([0, 3, 7, 11], dtype=torch.int32))


def test_unpack_restores_shared_prefix_to_every_row():
    batch = _rows([4, 4], pad=4)
    batch["input_ids"][1, :3] = batch["input_ids"][0, :3]
    layout = PackedLayout.build(
        batch["attention_mask"], [[0, 1]], [3], batch["position_ids"], input_ids=batch["input_ids"]
    )
    packed_ids = layout.pack(batch["input_ids"])
    restored = layout.unpack(packed_ids)
    torch.testing.assert_close(restored, batch["input_ids"])


def test_pack_gathers_from_one_source_row():
    x = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]], requires_grad=True)
    mask = torch.ones(2, 3, dtype=torch.long)
    ids = torch.tensor([[1, 2, 3], [1, 2, 9]])
    layout = PackedLayout.build(mask, [[0, 1]], [2], torch.arange(3).expand(2, -1), input_ids=ids)
    packed = layout.pack(x)
    packed.sum().backward()
    assert x.grad[0, 0].item() == 1.0
    assert x.grad[0, 1].item() == 1.0
    assert x.grad[1, 0].item() == 0.0
    assert x.grad[1, 1].item() == 0.0
    assert x.grad[0, 2].item() == 1.0
    assert x.grad[1, 2].item() == 1.0


def test_identical_prefix_is_required():
    batch = _rows([4, 4], pad=4)
    with pytest.raises(ValueError, match="not token-identical"):
        from arctic_platform.common.packing.layout import require_identical_prefixes

        PackedLayout.build(
            batch["attention_mask"], [[0, 1]], [3], batch["position_ids"], input_ids=batch["input_ids"]
        )


def test_router_disagreement_skips_share():
    routed = torch.tensor([[1, 2, 3], [1, 9, 3]])
    mask = torch.ones(2, 3, dtype=torch.long)
    assert skip_share_if_router_disagree([[0, 1]], [2], routed, mask) == [0]
    assert skip_share_if_router_disagree([[0, 1]], [1], routed, mask) == [1]


def test_shared_prefix_length_rounds_down_to_chunk():
    assert shared_prefix_length(70, 64) == 64
    assert shared_prefix_length(63, 64) == 0
    assert shared_prefix_length(70, None) == 70


def test_token_mean_required_when_sharing():
    require_token_mean_if_sharing(False, "seq-mean-token-sum")
    with pytest.raises(ValueError, match="token-mean"):
        require_token_mean_if_sharing(True, "seq-mean-token-sum")
    batch = _rows([4, 4], pad=4)
    batch["input_ids"][1, :3] = batch["input_ids"][0, :3]
    layout = PackedLayout.build(
        batch["attention_mask"], [[0, 1]], [3], batch["position_ids"], input_ids=batch["input_ids"]
    )
    model = nn.Linear(1, 1)
    with using_layout(model, layout):
        with pytest.raises(ValueError, match="token-mean"):
            active_layout_requires_token_mean("seq-mean-token-sum")


def test_cu_seqlens_from_position_ids_uses_stashed_layout():
    batch = _rows([4, 4], pad=4)
    batch["input_ids"][1, :3] = batch["input_ids"][0, :3]
    layout = PackedLayout.build(
        batch["attention_mask"], [[0, 1]], [3], batch["position_ids"], input_ids=batch["input_ids"]
    )
    model = nn.Linear(1, 1)
    packed = pack_with_layout(batch, layout)
    with using_layout(model, layout):
        torch.testing.assert_close(cu_seqlens_from_position_ids(packed["position_ids"]), layout.cu_seqlens)


def test_require_prefix_continuation_rejects_gdn():
    class FlashAttention(nn.Module):
        pass

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
        _attn_implementation = "sdpa"

    class Qwen3Attention(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = Cfg()

    with pytest.raises(ValueError, match="Qwen3Attention"):
        require_prefix_continuation(Qwen3Attention())
    with pytest.raises(ValueError, match="no sequence mixers"):
        require_prefix_continuation(nn.Linear(1, 1))


def test_require_unweighted_shared():
    mask = torch.ones(2, 4, dtype=torch.long)
    ids = torch.tensor([[1, 2, 3, 4], [1, 2, 8, 9]])
    layout = PackedLayout.build(mask, [[0, 1]], [2], torch.arange(4).expand(2, -1), input_ids=ids)
    weights = torch.tensor([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=torch.float)
    layout.require_unweighted_shared(weights)
    weights[1, 0] = 1
    with pytest.raises(ValueError, match="shared prefix"):
        layout.require_unweighted_shared(weights)


def test_split_group_duplicates_roots():
    mask = torch.ones(4, 3, dtype=torch.long)
    pos = torch.arange(3).expand(4, -1)
    ids = torch.arange(3).expand(4, -1).contiguous()
    layout = PackedLayout.build(mask, [[0, 1], [2, 3]], [2, 2], pos, input_ids=ids)
    assert layout.num_roots == 2
    assert layout.num_segments == 6


def test_left_padding_sharing_uses_valid_tokens():
    batch = _rows([4, 4], pad=4, left_pad=2)
    batch["input_ids"][1, 2:5] = batch["input_ids"][0, 2:5]
    layout = PackedLayout.build(
        batch["attention_mask"], [[0, 1]], [3], batch["position_ids"], input_ids=batch["input_ids"]
    )
    restored = layout.unpack(layout.pack(batch["input_ids"]))
    torch.testing.assert_close(restored, batch["input_ids"])


def test_left_padding_does_not_treat_pad_as_prefix():
    batch = _rows([4, 4], pad=4, left_pad=2)
    with pytest.raises(ValueError, match="not token-identical"):
        PackedLayout.build(
            batch["attention_mask"], [[0, 1]], [3], batch["position_ids"], input_ids=batch["input_ids"]
        )
