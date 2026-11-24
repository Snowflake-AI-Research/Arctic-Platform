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

"""PackedLayout: gather pack/unpack, sharing-off vs pack_microbatch, prefix attention, lifetime, gates.

Attention numerics run on CPU through ``flash_attn_varlen_reference`` (pure torch, bottom-right causal).
Nothing here imports the model stack; ``tests/model/test_packed_layout_continuation.py`` drives the
Arctic-Platform and Hugging Face attention modules.
"""

import sys
import threading
from contextlib import ExitStack

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import CheckpointError
from torch.utils.checkpoint import checkpoint
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from arctic_platform.common.packing import IGNORE_INDEX
from arctic_platform.common.packing import PackedLayout
from arctic_platform.common.packing import cu_seqlens_from_position_ids
from arctic_platform.common.packing import layout as layout_module
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
from arctic_platform.common.packing.layout import register_native_prefix_attention
from arctic_platform.common.packing.layout import require_dense_attention_layout
from arctic_platform.common.packing.layout import unshared_groups
from arctic_platform.testing_utils import flash_attn_varlen_reference
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close
from arctic_platform.testing_utils import torch_assert_equal

# float64 attention on CPU: different segment lengths change only the softmax reduction order.
F64_ATOL = 1e-12


@pytest.fixture(autouse=True)
def _reference_flash_attn(monkeypatch):
    """The reference kernel stands in for flash-attn 2.8, FA3 and FA4 here, installed or not."""
    monkeypatch.setattr(
        layout_module, "_installed_version", {"flash_attn": (2, 8), "flash_attn_3": (3, 0), "flash_attn_4": (4, 0)}.get
    )
    monkeypatch.setattr(layout_module, "_fa4_varlen_backward_available", lambda: True)


@register_native_prefix_attention
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
    batch["labels"][:, :shared] = IGNORE_INDEX
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


def test_row_broadcast_keys_match_pack_microbatch():
    batch = _rows([3, 2, 4], pad=5)
    batch["sampler_kl_weights"] = torch.tensor([0.5, 2.0, 3.0])
    batch["context"] = {"sampler_kl_weights": torch.tensor([1.0, 4.0, 9.0]), "labels": batch["labels"]}
    layout = PackedLayout.build(batch["attention_mask"], *unshared_groups(3), batch["position_ids"])
    packed_layout = pack_with_layout(batch, layout)
    packed_base, _ = pack_microbatch(batch)
    assert packed_layout.keys() == packed_base.keys()
    for key in ("sampler_kl_weights", "labels", "input_ids"):
        torch_assert_equal(packed_layout[key], packed_base[key], msg=key)
    for key in ("sampler_kl_weights", "labels"):
        torch_assert_equal(packed_layout["context"][key], packed_base["context"][key], msg=key)
    assert packed_layout["sampler_kl_weights"].shape == (1, layout.num_tokens)


def test_row_broadcast_keys_follow_logical_row_under_sharing():
    batch, layout = _shared_layout(lengths=(5, 4))
    batch["sampler_kl_weights"] = torch.tensor([0.5, 2.0])
    packed = pack_with_layout(batch, layout)
    # Root (3 tokens, gathered from row 0), row 0's continuation (2), row 1's continuation (1).
    torch_assert_equal(packed["sampler_kl_weights"], torch.tensor([[0.5, 0.5, 0.5, 0.5, 0.5, 2.0]]))
    torch_assert_equal(packed["sampler_kl_weights"][0], batch["sampler_kl_weights"][layout.logical_row])


def test_pack_with_layout_checks_routed_experts_against_router_replay():
    routed = torch.arange(16).view(2, 4, 2)
    routed[1, :3] = routed[0, :3]
    batch, replayed = _shared_layout(routed_experts=routed)
    packed = pack_with_layout({**batch, "routed_experts": routed}, replayed)
    torch_assert_equal(packed["routed_experts"][0], replayed.pack(routed))
    with pytest.raises(ValueError, match="carries none"):
        pack_with_layout(batch, replayed)
    other = routed.clone()
    other[1, 0] += 1
    with pytest.raises(ValueError, match="disagree on a shared prefix cell"):
        pack_with_layout({**batch, "routed_experts": other}, replayed)

    _, opted_out = _shared_layout(router_replay=False)
    pack_with_layout(batch, opted_out)
    with pytest.raises(ValueError, match="router_replay=False"):
        pack_with_layout({**batch, "context": {"routed_experts": routed}}, opted_out)
    _, unstated = _shared_layout()
    with pytest.raises(ValueError, match="were not checked"):
        pack_with_layout({**batch, "routed_experts": routed}, unstated)
    unshared = PackedLayout.build(batch["attention_mask"], *unshared_groups(2), batch["position_ids"])
    pack_with_layout({**batch, "routed_experts": routed}, unshared)


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
    # [1, 1, V] with T == 1 reads as a model output [1, T, V] unless told it is [T, 1, V].
    batched = logits.view(1, 1, 6)
    torch_assert_equal(one.unpack(batched), logits.view(1, 1, 6))
    torch_assert_equal(one.unpack(batched, batched=False), logits.view(1, 1, 1, 6))
    with pytest.raises(ValueError, match=r"\[1, T, \.\.\.\]"):
        one.unpack(logits, batched=True)
    torch_assert_equal(layout.unpack(packed, batched=False), layout.unpack(packed))


def test_build_rejects_all_pad_row():
    mask = torch.tensor([[1, 1, 1], [0, 0, 0]])
    with pytest.raises(ValueError, match=r"rows \[1\] have no valid tokens"):
        PackedLayout.build(mask, *unshared_groups(2))


def test_layout_compares_by_identity():
    _, layout = _shared_layout()
    moved = layout.to("cpu")
    assert layout == layout and layout != moved
    assert len({layout, moved}) == 2
    assert moved.origin is layout.origin


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


def _build_shared(batch, shared=3):
    return PackedLayout.build(
        batch["attention_mask"], [[0, 1]], [shared], batch["position_ids"], input_ids=batch["input_ids"]
    )


def test_prefix_positions_must_match_across_rows():
    batch, _ = _shared_layout(lengths=(5, 4))
    _build_shared(batch)
    require_identical_prefixes(batch["input_ids"], [[0, 1]], [3], batch["attention_mask"], batch["position_ids"])

    # Right padding: row 1's positions start at 1, so the root would carry row 0's RoPE for both rows.
    offset = {**batch, "position_ids": batch["position_ids"].clone()}
    offset["position_ids"][1, :4] += 1
    with pytest.raises(ValueError, match="different position_ids"):
        _build_shared(offset)
    # Only the shared prefix has to agree.
    offset["position_ids"][1, :3] -= 1
    _build_shared(offset)

    # Right-aligned rows with column positions arange(S): equal ids, positions 1.. vs 2..
    mask = torch.tensor([[0, 1, 1, 1, 1], [0, 0, 1, 1, 1]])
    ids = torch.tensor([[0, 7, 8, 9, 5], [0, 0, 7, 8, 9]])
    columns = torch.arange(5).expand(2, -1)
    with pytest.raises(ValueError, match="different position_ids"):
        PackedLayout.build(mask, [[0, 1]], [2], columns, input_ids=ids)
    # The same rows with valid-token positions (cumsum - 1, the dummy-batch convention) share.
    valid_positions = (mask.cumsum(1) - 1).clamp(min=0) * mask
    layout = PackedLayout.build(mask, [[0, 1]], [2], valid_positions, input_ids=ids)
    torch_assert_equal(layout.position_ids, torch.tensor([0, 1, 2, 3, 2]))
    # Left padding with equal positions shares.
    left = _rows([4, 4], pad=4, left_pad=2)
    left["input_ids"][1] = left["input_ids"][0]
    assert _build_shared(left).has_sharing


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


def test_pack_with_layout_refuses_loss_weight_on_shared_cells():
    batch, layout = _shared_layout(lengths=(5, 4))
    loss_mask = torch.zeros(2, 5, dtype=torch.long)
    loss_mask[:, 3:] = batch["attention_mask"][:, 3:]
    pack_with_layout({**batch, "loss_mask": loss_mask}, layout)
    for key in ("loss_mask", "response_mask", "sft_mask", "echo_observation_mask"):
        weighted = loss_mask.clone()
        weighted[1, 0] = 1
        with pytest.raises(ValueError, match="shared prefix"):
            pack_with_layout({**batch, "context": {key: weighted}}, layout)

    # labels are the weight only when no mask is sent: a provider may keep prompt labels and mask them out.
    labeled = batch["labels"].clone()
    labeled[1, 0] = 7
    with pytest.raises(ValueError, match="shared prefix"):
        pack_with_layout({**batch, "labels": labeled}, layout)
    pack_with_layout({**batch, "labels": labeled, "loss_mask": loss_mask}, layout)

    unshared = PackedLayout.build(batch["attention_mask"], *unshared_groups(2), batch["position_ids"])
    pack_with_layout({**batch, "labels": labeled, "loss_mask": torch.ones(2, 5, dtype=torch.long)}, unshared)


def test_pack_with_layout_requires_the_batch_it_was_built_from():
    batch, layout = _shared_layout(lengths=(5, 4))
    pack_with_layout(batch, layout)
    mask = batch["attention_mask"].clone()
    mask[1, 3] = 0
    with pytest.raises(ValueError, match="attention_mask"):
        pack_with_layout({**batch, "attention_mask": mask}, layout)
    with pytest.raises(ValueError, match="position_ids"):
        pack_with_layout({**batch, "position_ids": batch["position_ids"] + 1}, layout)
    ids = batch["input_ids"].clone()
    ids[1, 0] += 1
    with pytest.raises(ValueError, match="input_ids"):
        pack_with_layout({**batch, "input_ids": ids}, layout)
    # Continuation tokens are each row's own.
    ids = batch["input_ids"].clone()
    ids[1, 3] += 1
    pack_with_layout({**batch, "input_ids": ids}, layout)


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
    # isort: off
    from arctic_platform.rl.processors.functional import agg_loss
    from arctic_platform.integrations.verl.grpo_loss import agg_loss as verl_agg_loss

    # isort: on

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
        with pytest.raises(ValueError, match="5 packed tokens.*SP >= 2.*pad_packed_microbatch"):
            cu_seqlens_from_position_ids(torch.zeros(1, 8, dtype=torch.long))
        with pytest.raises(ValueError, match="position_ids == 0"):
            require_no_prefix_sharing("helper")
    require_no_prefix_sharing("helper")


def test_nested_unshared_context_keeps_sharing_guards():
    batch, outer = _shared_layout()
    packed = pack_with_layout(batch, outer)
    inner = PackedLayout.build(batch["attention_mask"], *unshared_groups(2), batch["position_ids"])
    model, other = FlashAttention(), FlashAttention()
    with using_layout(model, outer):
        with using_layout(model, outer):
            torch_assert_equal(cu_seqlens_from_position_ids(packed["position_ids"]), outer.cu_seqlens)
        with using_layout(other, inner):
            assert get_active_packed_layout() is inner
            with pytest.raises(ValueError, match="position_ids == 0"):
                require_no_prefix_sharing("helper")
            with pytest.raises(ValueError, match="token-mean"):
                active_layout_requires_token_mean("seq-mean-token-sum")
            # Either layout's segments could be the wrong call's, so neither is returned.
            with pytest.raises(ValueError, match="nested using_layout contexts"):
                cu_seqlens_from_position_ids(packed["position_ids"])
        with using_layout(other, None):
            with pytest.raises(ValueError, match="nested using_layout contexts"):
                cu_seqlens_from_position_ids(packed["position_ids"])
    with using_layout(other, inner):
        require_no_prefix_sharing("helper")
        active_layout_requires_token_mean("seq-mean-token-sum")
        torch_assert_equal(cu_seqlens_from_position_ids(torch.zeros(1, 8, dtype=torch.long)), inner.cu_seqlens)


def test_unstashed_attention_refuses_to_run_inside_another_context():
    set_seed(4)
    x, mask, layout, unshared = _attention_batch()
    model, other = FlashAttention().double(), FlashAttention().double()
    hidden = unshared.pack(x)
    segments = (unshared.cu_seqlens, unshared.max_len_q)
    expected = other(hidden, *segments)
    for active in (layout, unshared):
        with using_layout(model, active):
            with pytest.raises(ValueError, match="no PackedLayout stashed"):
                other(hidden, *segments)
            # Wrapped in its own unshared layout, the other model runs as it does on its own.
            with using_layout(other, unshared):
                torch_assert_equal(other(hidden, *segments), expected)
    torch_assert_equal(other(hidden, *segments), expected)


def test_dense_attention_refuses_packed_layouts():
    _, layout = _shared_layout()
    unshared = PackedLayout.build(torch.ones(2, 4, dtype=torch.long), *unshared_groups(2))
    single = PackedLayout.build(torch.ones(1, 4, dtype=torch.long), *unshared_groups(1))
    require_dense_attention_layout("SDPAAttention", None)
    require_dense_attention_layout("SDPAAttention", single)
    for packed in (layout, unshared):
        with pytest.raises(ValueError, match="no varlen path"):
            require_dense_attention_layout("SDPAAttention", packed)
        with using_layout(FlashAttention(), packed):
            with pytest.raises(ValueError, match="SDPAAttention ran with no PackedLayout stashed"):
                require_dense_attention_layout("SDPAAttention", None)


def test_using_layout_is_owned_by_one_thread():
    _, layout = _shared_layout()
    model = FlashAttention()
    entered, release = threading.Event(), threading.Event()
    seen = {}

    def hold():
        with using_layout(model, layout):
            entered.set()
            release.wait(timeout=30)

    def read():
        seen["active"] = get_active_packed_layout()

    holder = threading.Thread(target=hold)
    holder.start()
    try:
        assert entered.wait(timeout=30)
        with pytest.raises(RuntimeError, match="already active on thread"):
            with using_layout(FlashAttention(), None):
                pass
        assert get_active_packed_layout() is layout
        # Reads are not thread-checked: autograd runs backward on its own threads.
        reader = threading.Thread(target=read)
        reader.start()
        reader.join(timeout=30)
        assert seen["active"] is layout
    finally:
        release.set()
        holder.join(timeout=30)
    assert get_active_packed_layout() is None
    with using_layout(model, layout):
        with using_layout(model, layout):
            assert get_active_packed_layout() is layout


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

    # Native mixers are matched by registered type, not by class name, and not by subclass.
    unregistered = type("FlashAttention", (nn.Module,), {})
    with pytest.raises(ValueError, match="FlashAttention .no prefix-continuation path"):
        require_prefix_continuation(unregistered())
    subclass = type("FlashAttentionVariant", (FlashAttention,), {})
    with pytest.raises(ValueError, match="FlashAttentionVariant"):
        require_prefix_continuation(subclass())

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
    for impl in ("flash_attention_3", "flash_attention_4"):
        require_prefix_continuation(Qwen3Attention(impl=impl))
    with pytest.raises(ValueError, match="needs flash_attention_2, flash_attention_3 or flash_attention_4"):
        require_prefix_continuation(Qwen3Attention(impl="kernels-community/flash-attn"))
    with pytest.raises(ValueError, match="no sequence mixers"):
        require_prefix_continuation(nn.Linear(1, 1))

    for version in (3, 4):
        module = FlashAttention()
        module._flash_attn_version = version
        require_prefix_continuation(module)
    unknown = FlashAttention()
    unknown._flash_attn_version = 5
    with pytest.raises(ValueError, match="flash-attn 5"):
        require_prefix_continuation(unknown)


def test_sharing_refuses_state_space_and_conv_mixers():
    """Nemotron-H's Mamba-2 layers sit next to registered flash attention; their names carry no attention needle."""
    _, shared = _shared_layout()
    unshared = PackedLayout.build(torch.ones(2, 4, dtype=torch.long), *unshared_groups(2))

    def hybrid(mixer_name, **attrs):
        mixer = type(mixer_name, (nn.Module,), {})()
        for name, value in attrs.items():
            setattr(mixer, name, value)
        model = nn.Module()
        model.attn = FlashAttention()
        model.mixer = mixer
        return model

    models = {
        "NemotronHMambaLayer": hybrid("NemotronHMambaLayer"),
        "NemotronHMamba2Mixer": hybrid("NemotronHMamba2Mixer"),
        "CausalTokenBlock": hybrid("CausalTokenBlock", conv1d=nn.Conv1d(4, 4, 3, groups=4)),
    }
    for name, model in models.items():
        with pytest.raises(ValueError, match=name):
            with using_layout(model, shared):
                pass
        assert get_active_packed_layout() is None
        with using_layout(model, unshared):
            pass


def test_sharing_refuses_top_left_causal_flash_attn(monkeypatch):
    batch, layout = _shared_layout()
    unshared = PackedLayout.build(batch["attention_mask"], *unshared_groups(2), batch["position_ids"])
    fa3 = FlashAttention()
    fa3._flash_attn_version = 3

    def installed(**versions):
        monkeypatch.setattr(layout_module, "_installed_version", versions.get)

    installed(flash_attn=(2, 0), flash_attn_3=(3, 0))
    with pytest.raises(ValueError, match="flash_attn >= 2.1"):
        with using_layout(FlashAttention(), layout):
            pass
    assert get_active_packed_layout() is None
    with using_layout(FlashAttention(), unshared):
        pass
    with using_layout(fa3, layout):
        pass

    installed()
    with pytest.raises(ValueError, match="flash_attn package .* not installed"):
        with using_layout(FlashAttention(), layout):
            pass
    with pytest.raises(ValueError, match="flash_attn_3 package .* not installed"):
        with using_layout(fa3, layout):
            pass
    for model in (FlashAttention(), fa3):
        with using_layout(model, unshared):
            pass

    for version in ((2, 1), (2, 8)):
        installed(flash_attn=version)
        with using_layout(FlashAttention(), layout):
            pass
    installed(flash_attn_3=(3, 0))
    with using_layout(fa3, layout):
        pass

    # flash_attn 2.8 bundles an FA4 (flash_attn.cute) with no varlen backward; only the flash-attn-4 package counts.
    fa4 = FlashAttention()
    fa4._flash_attn_version = 4
    installed(flash_attn=(2, 8))
    with pytest.raises(ValueError, match="flash_attn_4 package .* not installed"):
        with using_layout(fa4, layout):
            pass
    installed(flash_attn_4=(4, 0))
    with using_layout(fa4, layout):
        pass
    # Installing flash_attn after flash-attn-4 puts the bundled flash_attn.cute back over it.
    monkeypatch.setattr(layout_module, "_fa4_varlen_backward_available", lambda: False)
    with pytest.raises(ValueError, match="bundled copy"):
        with using_layout(fa4, layout):
            pass
    with using_layout(fa4, unshared):
        pass


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
    """``loss_fn(module, x, backward=True)`` runs its own backward, inside any layout context it opens."""
    module.zero_grad()
    x = x.clone().requires_grad_(True)
    loss_fn(module, x, backward=True)
    return x.grad, {name: p.grad.clone() for name, p in module.named_parameters()}


def test_prefix_attention_matches_per_row_forward_and_backward():
    set_seed(0)
    x, mask, layout, _ = _attention_batch()
    module = FlashAttention().double()
    weights = _continuation_weights(x, mask, layout)

    def shared_loss(m, h, backward=False):
        with using_layout(m, layout):
            out = layout.unpack(m(layout.pack(h)))
            loss = (out * weights).sum()
            if backward:
                loss.backward()
        return loss, out

    def row_loss(m, h, backward=False):
        out = _per_row(m, h, mask)
        loss = (out * weights).sum()
        if backward:
            loss.backward()
        return loss, out

    (_, out_shared), (_, out_rows) = shared_loss(module, x), row_loss(module, x)
    valid = mask.bool()
    torch_assert_close(out_shared[valid], out_rows[valid], rtol=0, atol=F64_ATOL)

    x_grad, p_grad = _grads(module, x, shared_loss)
    x_grad_ref, p_grad_ref = _grads(module, x, row_loss)
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


def _checkpoint_grads(module, x, layout, weights, use_reentrant, through_unpack=True, exit_before_backward=False):
    """Input and parameter grads of a weighted loss over ``module``'s output; ``use_reentrant=None`` skips checkpointing."""
    segments = (layout.cu_seqlens, layout.max_len_q)
    packed_weights = layout.pack(weights)
    module.zero_grad()
    h = x.clone().requires_grad_(True)
    with ExitStack() as stack:
        stack.enter_context(using_layout(module, layout))
        packed = layout.pack(h)
        if use_reentrant is None:
            out = module(packed, *segments)
        else:
            out = checkpoint(module, packed, *segments, use_reentrant=use_reentrant)
        loss = (layout.unpack(out) * weights).sum() if through_unpack else (out * packed_weights).sum()
        if exit_before_backward:
            stack.close()
        loss.backward()
    return [h.grad] + [p.grad.clone() for p in module.parameters()]


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_checkpoint_recompute_under_layout_matches_no_checkpoint(use_reentrant):
    set_seed(2)
    x, mask, layout, _ = _attention_batch()
    module = FlashAttention().double()
    weights = _continuation_weights(x, mask, layout)
    expected = _checkpoint_grads(module, x, layout, weights, use_reentrant=None)
    for through_unpack in (True, False):
        got = _checkpoint_grads(module, x, layout, weights, use_reentrant, through_unpack)
        for g, want in zip(got, expected):
            torch_assert_equal(g, want)

    # Backward after the context exits recomputes without the prefix gather. The hook unpack puts on
    # backward refuses that before any recompute runs, whichever checkpoint flavor is in use.
    with pytest.raises(RuntimeError, match="after its using_layout exited"):
        _checkpoint_grads(module, x, layout, weights, use_reentrant, exit_before_backward=True)


def test_checkpoint_after_context_exit_without_unpack():
    """With the loss off ``unpack``, nothing hooks backward: pin what each checkpoint flavor then does."""
    set_seed(2)
    x, mask, layout, _ = _attention_batch()
    module = FlashAttention().double()
    weights = _continuation_weights(x, mask, layout)
    expected = _checkpoint_grads(module, x, layout, weights, use_reentrant=None)
    exited = {"through_unpack": False, "exit_before_backward": True}
    with pytest.raises(CheckpointError):
        _checkpoint_grads(module, x, layout, weights, use_reentrant=False, **exited)
    # Reentrant recompute silently attends without the root: finite grads that do not match.
    got = _checkpoint_grads(module, x, layout, weights, use_reentrant=True, **exited)
    assert all(torch.isfinite(g).all() for g in got)
    assert max((g - want).abs().max().item() for g, want in zip(got, expected)) > 1e-3


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


def _ulysses_registry_entry(ulysses, full_qkv):
    """Shaped like DeepSpeed's ``register_with_transformers`` closure: the core runs on the gathered sequence."""

    def uattn_wrapper(module, query, key, value, attention_mask, *args, **kwargs):
        assert query.shape[2] * 2 == full_qkv[0].shape[2], "the registry entry sees one rank's window"
        return ulysses.attn(module, *full_qkv, None, *args, **kwargs)

    return uattn_wrapper


class _HFAttentionConfig:
    _attn_implementation = "flash_attention_2"


class Qwen3_5Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = _HFAttentionConfig()
        self.sliding_window = None


@pytest.mark.parametrize("subclass", [False, True], ids=["ulysses", "heterogeneous"])
def test_ulysses_prefix_gather_runs_after_the_all_to_all(monkeypatch, subclass):
    ulysses_sp = pytest.importorskip("deepspeed.runtime.sequence_parallel.ulysses_sp")

    _, layout = _shared_layout(lengths=(4, 5), shared=3)
    tokens = layout.num_tokens
    assert tokens == 6
    calls = []

    def core(module, query, key, value, attention_mask, *args, **kwargs):
        calls.append({"q": query.shape[2], "k": key.shape[2], **kwargs})
        return query.transpose(1, 2), None

    cls = ulysses_sp.UlyssesSPAttentionHF
    if subclass:
        cls = type("HeterogeneousUlyssesSPAttentionHF", (cls,), {})
    ulysses = cls.__new__(cls)
    nn.Module.__init__(ulysses)
    ulysses.attn = core
    full = tuple(torch.randn(1, 2, tokens, 4) for _ in range(3))
    entry = _ulysses_registry_entry(ulysses, full)
    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._global_mapping, "flash_attention_2", entry)
    module = Qwen3_5Attention()
    local = tuple(tensor[:, :, : tokens // 2] for tensor in full)

    if subclass:
        with pytest.raises(ValueError, match="not the subclass HeterogeneousUlyssesSPAttentionHF"):
            with using_layout(module, layout):
                pass
        assert ulysses.attn is core and ALL_ATTENTION_FUNCTIONS["flash_attention_2"] is entry
        return

    with using_layout(module, layout):
        ALL_ATTENTION_FUNCTIONS["flash_attention_2"](module, *local, None, cu_seq_lens_q=torch.tensor([0, 3]))
        assert ulysses.attn is not core
    assert ulysses.attn is core and ALL_ATTENTION_FUNCTIONS["flash_attention_2"] is entry
    assert len(calls) == 1
    assert calls[0]["q"] == tokens and calls[0]["k"] == layout.kv_index.numel()
    torch_assert_equal(calls[0]["cu_seq_lens_q"], layout.cu_seqlens.to(torch.int32))
    torch_assert_equal(calls[0]["cu_seq_lens_k"], layout.cu_seqlens_k.to(torch.int32))


def test_ulysses_entry_still_refuses_an_unstashed_module_in_a_sharing_context(monkeypatch):
    ulysses_sp = pytest.importorskip("deepspeed.runtime.sequence_parallel.ulysses_sp")

    _, layout = _shared_layout(lengths=(4, 5), shared=3)
    ulysses = ulysses_sp.UlyssesSPAttentionHF.__new__(ulysses_sp.UlyssesSPAttentionHF)
    nn.Module.__init__(ulysses)
    ulysses.attn = lambda *args, **kwargs: (None, None)
    full = tuple(torch.randn(1, 2, layout.num_tokens, 4) for _ in range(3))
    monkeypatch.setitem(
        ALL_ATTENTION_FUNCTIONS._global_mapping, "flash_attention_2", _ulysses_registry_entry(ulysses, full)
    )
    local = tuple(tensor[:, :, :3] for tensor in full)
    with using_layout(Qwen3_5Attention(), layout):
        with pytest.raises(ValueError, match="no PackedLayout stashed"):
            ALL_ATTENTION_FUNCTIONS["flash_attention_2"](Qwen3_5Attention(), *local, None)
