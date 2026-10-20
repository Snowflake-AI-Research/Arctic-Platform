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

"""Check bounded selection against the original eager FP32 indexer."""

from unittest.mock import patch

import pytest
import torch
from torch.nn import functional as F

from arctic_platform.model.implementations.glm53 import context_parallel as cp
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_equal


@torch.no_grad()
def _eager_reference(indexer, hidden, residual, mask, boundaries, *, packed, rank):
    batch, length = hidden.shape[:2]
    query = indexer.wq_b(residual).view(batch, length, -1, indexer.head_dim)
    weights = indexer.weights_proj(hidden.to(indexer.weights_proj.weight.dtype)).float()
    weights *= indexer.n_heads**-0.5
    width = indexer.index_topk + (indexer.index_kpool - 1 if indexer.index_kpool_always_select_tail else 0)
    output = torch.full((batch, length, width), -1, device=hidden.device, dtype=torch.int32)
    offset = rank * length
    for start, end in zip(boundaries[:-1].tolist(), boundaries[1:].tolist(), strict=True):
        lo, hi = max(start, offset), min(end, offset + length)
        if lo >= hi:
            continue
        query_slice = slice(lo - offset, hi - offset)
        states = packed[:, start:end]
        valid_keys = states[..., -1].bool()
        visible = (
            torch.arange(end - start, device=hidden.device)[None, None]
            <= torch.arange(lo - start, hi - start, device=hidden.device)[None, :, None]
        ) & valid_keys[:, None]
        keys, indices, valid = indexer.get_pooled_states(packed_states=states)
        scores = torch.matmul(query[:, query_slice].float(), keys.transpose(-1, -2).float().unsqueeze(1))
        scores = F.relu(scores * indexer.softmax_scale)
        scores = torch.matmul(weights[:, query_slice].unsqueeze(-2), scores).squeeze(-2)
        pool_end = indices[..., -1].clamp(0, end - start - 1)
        candidates = visible.gather(-1, pool_end[:, None].expand(batch, hi - lo, -1)) & valid[:, None]
        scores.masked_fill_(~candidates, torch.finfo(scores.dtype).min)
        selected = scores.topk(min(indexer.index_topk // indexer.index_kpool, scores.shape[-1]), dim=-1).indices
        selected_indices = indices[torch.arange(batch, device=hidden.device)[:, None, None], selected]
        selected_indices.masked_fill_(~candidates.gather(-1, selected)[..., None].expand_as(selected_indices), -1)
        selected_indices = selected_indices.flatten(-2)
        if indexer.index_kpool_always_select_tail:
            selected_indices = indexer.append_visible_tail(selected_indices, visible, valid_keys)
        selected_indices = F.pad(selected_indices, (0, width - selected_indices.shape[-1]), value=-1)[..., :width]
        selected_indices = torch.where(selected_indices >= 0, selected_indices + start, selected_indices)
        selected_indices.masked_fill_(~mask[:, query_slice, None], -1)
        output[:, query_slice] = selected_indices.to(torch.int32)
    return output


def _inputs(length, device, tiny_glm_sparse_model):
    model = tiny_glm_sparse_model().to(device)
    indexer = model.model.layers[0].self_attn.indexer
    hidden = torch.randn(2, length, 16, device=device, dtype=torch.bfloat16)
    residual = torch.randn(2, length, 8, device=device, dtype=torch.bfloat16)
    mask = torch.ones(2, length, device=device, dtype=torch.bool)
    mask[0, 1] = False
    key = indexer.k_norm(indexer.wk(hidden))
    gates = F.linear(hidden, indexer.index_kpool_compress_gate)
    packed = torch.cat((key, gates, mask[..., None].to(key.dtype)), dim=-1)
    return indexer, hidden, residual, mask, packed


def _check_chunked_indices(rank, tail, device, tiny_glm_sparse_model):
    set_seed(29)
    indexer, hidden, residual, mask, packed = _inputs(24, device, tiny_glm_sparse_model)
    indexer.index_kpool_always_select_tail = tail
    boundaries = torch.tensor([0, 3, 9, 17, 24], device=device, dtype=torch.int32)
    local = slice(rank * 8, (rank + 1) * 8)
    hidden, residual, mask = hidden[:, local], residual[:, local], mask[:, local]
    expected = _eager_reference(indexer, hidden, residual, mask, boundaries, packed=packed, rank=rank)
    score_sizes = []
    matmul = torch.matmul

    def measured_matmul(left, right):
        result = matmul(left, right)
        if left.shape[-2] == indexer.n_heads and right.shape[-2] == indexer.head_dim:
            score_sizes.append(result.numel() * result.element_size())
        return result

    with patch.object(cp, "_gather_sequence_no_grad", return_value=packed):
        with patch.object(cp, "_INDEXER_SCORE_CHUNK_BYTES", 96), patch.object(torch, "matmul", measured_matmul):
            actual = cp._select_sparse_indices(
                indexer,
                hidden,
                residual,
                mask,
                boundaries,
                cp_group=None,
                cp_rank=rank,
            )
    torch_assert_equal(actual, expected)
    assert score_sizes and max(score_sizes) <= 96


@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("tail", [False, True])
def test_chunked_indices_match_eager_with_packed_boundaries(rank, tail, tiny_glm_sparse_model):
    _check_chunked_indices(rank, tail, "cpu", tiny_glm_sparse_model)


@require_torch_gpu
@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("tail", [False, True])
def test_chunked_cuda_indices_match_eager_with_packed_boundaries(rank, tail, tiny_glm_sparse_model):
    _check_chunked_indices(rank, tail, "cuda", tiny_glm_sparse_model)


def _check_model_parity(device, tiny_glm_sparse_model):
    set_seed(31)
    model = tiny_glm_sparse_model().to(device).train()
    tokens = torch.arange(1, 17, device=device)[None]
    positions = torch.arange(16, device=device)[None]

    def run(selector):
        model.zero_grad(set_to_none=True)
        with patch.object(cp, "_select_sparse_indices", selector):
            output = model(input_ids=tokens, position_ids=positions, use_cache=False).last_hidden_state
            loss = output.float().square().mean()
            loss.backward()
        gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
        return output.detach(), loss.detach(), gradients

    def eager(indexer, hidden, residual, mask, boundaries, *, cp_group, cp_rank):
        key = indexer.k_norm(indexer.wk(hidden))
        gates = F.linear(hidden, indexer.index_kpool_compress_gate)
        packed = torch.cat((key, gates, mask[..., None].to(key.dtype)), dim=-1)
        return _eager_reference(indexer, hidden, residual, mask, boundaries, packed=packed, rank=cp_rank)

    expected, expected_loss, expected_gradients = run(eager)
    with patch.object(cp, "_INDEXER_SCORE_CHUNK_BYTES", 128):
        actual, actual_loss, actual_gradients = run(cp._select_sparse_indices)
    torch_assert_equal(actual, expected)
    torch_assert_equal(actual_loss, expected_loss)
    assert actual_gradients.keys() == expected_gradients.keys()
    for name, gradient in actual_gradients.items():
        torch_assert_equal(gradient, expected_gradients[name], msg=name)


def test_chunking_preserves_model_outputs_and_parameter_gradients(tiny_glm_sparse_model):
    _check_model_parity("cpu", tiny_glm_sparse_model)


@require_torch_gpu
def test_cuda_chunking_preserves_model_outputs_and_parameter_gradients(tiny_glm_sparse_model):
    _check_model_parity("cuda", tiny_glm_sparse_model)


@require_torch_gpu
def test_chunked_cuda_peak_is_lower_with_live_safety_reserve(tiny_glm_sparse_model):
    set_seed(37)
    reserve = torch.empty(20 * 1024**3, dtype=torch.uint8, device="cuda")
    indexer, hidden, residual, mask, packed = _inputs(4096, "cuda", tiny_glm_sparse_model)
    boundaries = torch.tensor([0, 4096], device="cuda", dtype=torch.int32)

    def peak(selector):
        torch.cuda.synchronize()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        output = selector()
        torch.cuda.synchronize()
        return output, torch.cuda.max_memory_allocated() - base

    expected, eager_peak = peak(
        lambda: _eager_reference(indexer, hidden, residual, mask, boundaries, packed=packed, rank=0)
    )
    with patch.object(cp, "_INDEXER_SCORE_CHUNK_BYTES", 1024 * 1024):
        with patch.object(cp, "_gather_sequence_no_grad", return_value=packed):
            actual, bounded_peak = peak(
                lambda: cp._select_sparse_indices(
                    indexer,
                    hidden,
                    residual,
                    mask,
                    boundaries,
                    cp_group=None,
                    cp_rank=0,
                )
            )
    torch_assert_equal(actual, expected)
    assert reserve.numel() == 20 * 1024**3
    assert bounded_peak < eager_peak / 2
