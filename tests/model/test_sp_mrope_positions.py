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

"""SP and CP adapters rebuild a packed call's global row boundaries from text or M-RoPE position windows."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
from torch import nn

from arctic_platform.model.implementations.gpu.packing import cu_seqlens_from_position_ids
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_equal

WORLD_SIZE = 4
WINDOW = 4
# Two packed calls of 16 tokens over four 4-token windows. In both, the 5-token row crosses the first window
# boundary and the rows ending at 8 and 12 end exactly on one. "padded" ends in two pad tokens, which
# ``pad_packed_microbatch`` writes at position 0, so each reads as its own one-token row.
LAYOUTS = {
    "rows": ((3, 5, 4, 4), 0, [0, 3, 8, 12, 16]),
    "padded": ((3, 5, 4, 2), 2, [0, 3, 8, 12, 14, 15, 16]),
}


@dataclass(frozen=True)
class PackedCall:
    full: torch.Tensor  # what each rank windows: [1, T] text ids or [4, 1, T] M-RoPE ids
    text: torch.Tensor  # the unsharded call's row positions, [1, T]
    cu_seqlens: list[int]
    max_seqlen: int


def _packed_call(layout: str, kind: str) -> PackedCall:
    row_lengths, pad_tokens, cu_seqlens = LAYOUTS[layout]
    pieces = [torch.arange(length) for length in row_lengths] + [torch.zeros(pad_tokens, dtype=torch.long)]
    text = torch.cat(pieces)[None, :]
    assert text.shape[-1] == WORLD_SIZE * WINDOW
    # Channel 0 holds the row positions; the other channels must not be read as row positions.
    full = text if kind == "text" else torch.stack([text, text + 7, text + 11, text + 13])
    return PackedCall(full=full, text=text, cu_seqlens=cu_seqlens, max_seqlen=max(row_lengths))


@pytest.fixture(params=[(layout, kind) for layout in LAYOUTS for kind in ("text", "mrope")], ids="-".join)
def call(request) -> PackedCall:
    return _packed_call(*request.param)


def _window(positions: torch.Tensor, rank: int) -> torch.Tensor:
    return positions[..., rank * WINDOW : (rank + 1) * WINDOW].contiguous()


def _spy_on_boundary_reads(monkeypatch, module) -> list[torch.Tensor]:
    """Record the positions ``module`` hands to ``cu_seqlens_from_position_ids``."""
    seen = []

    def spy(position_ids):
        seen.append(position_ids)
        return cu_seqlens_from_position_ids(position_ids)

    monkeypatch.setattr(module, "cu_seqlens_from_position_ids", spy)
    return seen


def _install_fake_group(monkeypatch, full: torch.Tensor, rank: int) -> object:
    """Make ``dist`` look like rank ``rank`` of a ``WORLD_SIZE`` group holding windows of ``full``."""
    group = object()

    def all_gather(outputs, tensor, group=None, async_op=False):
        assert len(outputs) == WORLD_SIZE
        if tensor.dtype.is_floating_point:
            # Activations the adapter gathers alongside the positions; their values do not matter here.
            for output in outputs:
                output.copy_(tensor)
            return None
        source = full if tensor.ndim == full.ndim else full[0]
        assert torch.equal(tensor, _window(source, rank)), "a rank must contribute its own window"
        for peer, output in enumerate(outputs):
            output.copy_(_window(source, peer))
        return None

    monkeypatch.setattr(dist, "get_world_size", lambda group=None: WORLD_SIZE)
    monkeypatch.setattr(dist, "get_rank", lambda group=None: rank)
    monkeypatch.setattr(dist, "all_gather", all_gather)
    return group


def test_unsharded_call_has_the_expected_boundaries(call):
    assert cu_seqlens_from_position_ids(call.full).tolist() == call.cu_seqlens


@pytest.mark.parametrize("rank", range(WORLD_SIZE))
def test_gather_packed_positions_rebuilds_the_call_on_the_token_axis(monkeypatch, call, rank):
    from arctic_platform.model.implementations.gpu.sp.collectives import gather_packed_positions

    group = _install_fake_group(monkeypatch, call.full, rank)

    gathered = gather_packed_positions(_window(call.full, rank), group)

    torch_assert_equal(gathered, call.text)


@pytest.mark.parametrize("shape", [(8,), (2, 4, 1, 4), (4, 1, 4, 1)])
def test_gather_packed_positions_refuses_other_shapes(monkeypatch, shape):
    from arctic_platform.model.implementations.gpu.sp.collectives import gather_packed_positions

    monkeypatch.setattr(dist, "all_gather", lambda *_args, **_kwargs: pytest.fail("must refuse before gathering"))

    with pytest.raises(ValueError, match="packed position_ids"):
        gather_packed_positions(torch.zeros(shape, dtype=torch.long), object())


@pytest.mark.parametrize("rank", range(WORLD_SIZE))
def test_gated_delta_net_sp_publishes_global_boundaries(monkeypatch, call, rank):
    from arctic_platform.model.implementations.gpu.sp import gated_delta_net

    group = _install_fake_group(monkeypatch, call.full, rank)
    seen = _spy_on_boundary_reads(monkeypatch, gated_delta_net)
    model = nn.Module()
    model.forward = lambda **kwargs: kwargs
    linear = nn.Module()
    gated_delta_net._install_global_boundary_wrapper(model, [linear], group)
    local = _window(call.full, rank)

    assert model.forward(position_ids=local)["position_ids"] is local
    assert len(seen) == 1
    torch_assert_equal(seen[0], call.text)
    assert getattr(linear, gated_delta_net._GLOBAL_CU_SEQLENS_ATTRIBUTE).tolist() == call.cu_seqlens


@pytest.mark.parametrize("rank", range(WORLD_SIZE))
def test_qwen38_cp_builds_global_boundaries(monkeypatch, call, rank):
    from arctic_platform.model.implementations.qwen38 import context_parallel

    group = _install_fake_group(monkeypatch, call.full, rank)
    seen = _spy_on_boundary_reads(monkeypatch, context_parallel)
    observed = {}

    def build_context(**kwargs):
        observed.update(kwargs)
        return object()

    monkeypatch.setattr(context_parallel, "build_gated_delta_net_cp_context", build_context)

    class Backbone(nn.Module):
        def forward(self, **kwargs):
            return kwargs

    backbone = Backbone()
    linear = nn.Module()
    linear.conv_kernel_size = 4
    context_parallel._wrap_backbone_forward(backbone, [linear], group)

    backbone(position_ids=_window(call.full, rank))

    assert len(seen) == 1
    torch_assert_equal(seen[0], call.text)
    assert observed["global_cu_seqlens"].tolist() == call.cu_seqlens
    assert observed["local_sequence_length"] == WINDOW
    assert getattr(linear, context_parallel._GLOBAL_CU_SEQLENS).tolist() == call.cu_seqlens


@pytest.mark.parametrize("rank", range(WORLD_SIZE))
def test_moe_sp_stashes_global_boundaries(monkeypatch, call, rank):
    from arctic_platform.model.implementations.moe import sequence_parallel

    group = _install_fake_group(monkeypatch, call.full, rank)
    seen = _spy_on_boundary_reads(monkeypatch, sequence_parallel)

    def original_forward(input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None):
        return position_ids

    softmax_attention = nn.Module()
    linear_attention = nn.Module()
    mamba = nn.Module()
    forward = sequence_parallel._make_backbone_sp_forward(
        original_forward, group, [softmax_attention], [linear_attention], [mamba]
    )
    local = _window(call.full, rank)

    assert forward(None, position_ids=local) is local
    assert len(seen) == 1
    torch_assert_equal(seen[0], call.text)
    assert softmax_attention._sp_cu_seqlens.tolist() == call.cu_seqlens
    assert softmax_attention._sp_max_seqlen == call.max_seqlen
    assert linear_attention._dss_sp_global_cu_seqlens.tolist() == call.cu_seqlens
    assert mamba._sp_cu_seqlens.tolist() == call.cu_seqlens


def _row_spans_seen_by_qsa_indexer(monkeypatch, call, rank) -> list[tuple[int, int]]:
    pytest.importorskip("transformers.models.qwen4_exp.modeling_qwen4_exp")
    from arctic_platform.model.implementations.qwen38 import qsa_flex

    group = _install_fake_group(monkeypatch, call.full, rank)
    head_dim = 4
    indexer = nn.Module()
    indexer.index_n_heads = 1
    indexer.index_head_dim = head_dim
    indexer.index_qk_proj = nn.Linear(8, 2 * head_dim, bias=False)
    indexer.q_layernorm = nn.Identity()
    indexer.k_layernorm = nn.Identity()
    indexer.token_budget = 2
    indexer.compress_ratio = 1
    indexer._cp_group = group

    # The indexer visits this rank's pieces of each row in order; each call says how far into its row the piece
    # starts and how long the row is, which locates the row in the packed call.
    spans = []
    cursor = rank * WINDOW
    real_select = qsa_flex.select_qsa_token_ids

    def select(index_queries, compressed_keys, sequence_lengths, **kwargs):
        nonlocal cursor
        start = cursor - kwargs["query_offset"]
        spans.append((start, start + int(sequence_lengths[0])))
        cursor += index_queries.shape[1]
        return real_select(index_queries, compressed_keys, sequence_lengths, **kwargs)

    monkeypatch.setattr(qsa_flex, "select_qsa_token_ids", select)
    cos = torch.ones(1, WINDOW, head_dim)
    sin = torch.zeros(1, WINDOW, head_dim)
    set_seed(0)
    with torch.no_grad():
        qsa_flex._qsa_indexer_forward(
            indexer,
            torch.randn(1, WINDOW, 8),
            (cos, sin),
            None,
            position_ids=_window(call.full, rank),
        )
    assert cursor == (rank + 1) * WINDOW, "every local query must belong to exactly one row"
    return spans


def test_qsa_indexer_finds_the_global_rows(monkeypatch, call):
    spans = set()
    for rank in range(WORLD_SIZE):
        with monkeypatch.context() as patch:
            spans.update(_row_spans_seen_by_qsa_indexer(patch, call, rank))

    assert spans == set(zip(call.cu_seqlens, call.cu_seqlens[1:]))
