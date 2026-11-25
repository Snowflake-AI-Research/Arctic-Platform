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
"""CPU checks for packed mixer tiling: budget, batch size, and missing packed boundaries."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from arctic_platform.model.config import TiledRematPatch
from arctic_platform.model.patches.packed_seq_tiled_remat import install_packed_seq_tiled_remat

_RUNTIME = "deepspeed.runtime.activation_checkpointing.tiled_rematerialization"


class _Attn(nn.Module):
    def forward(self, hidden_states, position_ids=None, cu_seq_lens_q=None, cu_seq_lens_k=None, seq_idx=None):
        return hidden_states


class _DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = nn.Identity()


class _Model(nn.Module):
    def __init__(self, impl):
        super().__init__()
        self.config = type("Config", (), {"_attn_implementation": impl})()
        self.layer = _DecoderLayer()


@pytest.fixture(autouse=True)
def _restore_attn_forward():
    original = _Attn.forward
    yield
    _Attn.forward = original


def test_token_budget_default_is_4096():
    assert TiledRematPatch().token_budget == 4096


def test_rejects_attention_that_is_not_flash():
    model = _Model("sdpa")
    with pytest.raises(ValueError, match="flash attention"):
        install_packed_seq_tiled_remat(model, 128)


def test_plan_packed_tiles_keeps_whole_samples_under_the_budget():
    runtime = pytest.importorskip(_RUNTIME)
    cu_seqlens = torch.tensor([0, 100, 200, 350], dtype=torch.int32)
    tiles = runtime.plan_packed_tiles(cu_seqlens, token_budget=150)
    assert tiles == [(0, 1, 0, 100), (1, 2, 100, 200), (2, 3, 200, 350)]
    one_sample = torch.tensor([0, 500], dtype=torch.int32)
    assert runtime.plan_packed_tiles(one_sample, token_budget=150) == [(0, 1, 0, 500)]


def test_rejects_a_missing_mixer():
    pytest.importorskip(_RUNTIME)

    class _Bare(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = type("Config", (), {"_attn_implementation": "flash_attention_2"})()

    with pytest.raises(RuntimeError, match="no self_attn/linear_attn"):
        install_packed_seq_tiled_remat(_Bare(), 128)


def test_rejects_micro_batch_above_one():
    pytest.importorskip(_RUNTIME)
    model = _Model("flash_attention_2")
    install_packed_seq_tiled_remat(model, 128)
    hidden = torch.zeros(2, 4, 8)
    cu_seqlens = torch.tensor([0, 4], dtype=torch.int32)
    with pytest.raises(RuntimeError, match="micro batch 1"):
        model.layer.self_attn(hidden, cu_seq_lens_q=cu_seqlens)


def test_rejects_a_forward_without_packed_boundaries():
    pytest.importorskip(_RUNTIME)
    model = _Model("flash_attention_2")
    install_packed_seq_tiled_remat(model, 128)
    hidden = torch.zeros(1, 4, 8)
    with pytest.raises(RuntimeError, match="Packed batches are required"):
        model.layer.self_attn(hidden)


def test_rejects_seq_idx_that_disagrees_with_cu_seqlens():
    pytest.importorskip(_RUNTIME)
    model = _Model("flash_attention_2")
    install_packed_seq_tiled_remat(model, 128)
    hidden = torch.zeros(1, 4, 8)
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)
    seq_idx = torch.zeros(1, 4, dtype=torch.int32)
    with pytest.raises(RuntimeError, match="seq_idx disagrees"):
        model.layer.self_attn(hidden, cu_seq_lens_q=cu_seqlens, seq_idx=seq_idx)


def test_forward_passes_the_configured_token_budget():
    runtime = pytest.importorskip(_RUNTIME)
    seen = {}
    real_compute = runtime.packed_seq_tiled_rematerialize_compute
    real_classify = runtime.classify_packed_kwarg

    def _compute(fn, hidden, cu_seqlens, shards=None, token_budget=None, compute_params=None, **kwargs):
        seen["token_budget"] = token_budget
        seen["cu_seqlens"] = cu_seqlens
        return hidden

    def _classify(name, value, num_samples):
        return "pass"

    runtime.packed_seq_tiled_rematerialize_compute = _compute
    runtime.classify_packed_kwarg = _classify
    try:
        model = _Model("flash_attention_2")
        install_packed_seq_tiled_remat(model, 128)
        hidden = torch.zeros(1, 4, 8)
        cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32)
        out = model.layer.self_attn(hidden, cu_seq_lens_q=cu_seqlens)
    finally:
        runtime.packed_seq_tiled_rematerialize_compute = real_compute
        runtime.classify_packed_kwarg = real_classify
    assert out.shape == hidden.shape
    assert seen["token_budget"] == 128
    assert torch.equal(seen["cu_seqlens"], cu_seqlens)
