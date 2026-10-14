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

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from arctic_platform.model.implementations.layers.cp_mamba import _causal_conv_packed

from arctic_platform.model.implementations.moe import sequence_parallel


class _Attention(nn.Module):
    def forward(self, hidden_states, position_embeddings=None, cu_seqlens=None, max_seqlen=None):
        return hidden_states, None


class _Backbone(nn.Module):
    def __init__(self, model_type: str, layer_types: list[str | None]):
        super().__init__()
        self.config = SimpleNamespace(
            model_type=model_type,
            _attn_implementation="flash_attention_3",
        )
        self.layers = nn.ModuleList()
        for layer_type in layer_types:
            layer = nn.Module()
            layer.self_attn = _Attention()
            if layer_type is not None:
                layer.attention_type = layer_type
            self.layers.append(layer)

    def forward(self, input_ids=None, position_ids=None, inputs_embeds=None, routed_experts=None):
        return inputs_embeds


@pytest.mark.parametrize("layer_type", [None, "full_attention", "sliding_attention"])
def test_generic_sequence_parallel_wraps_softmax_attention(monkeypatch, layer_type):
    backbone = _Backbone("qwen3_moe", [layer_type])
    monkeypatch.setattr(sequence_parallel, "get_language_model", lambda model: model)
    monkeypatch.setattr(sequence_parallel.dist, "get_world_size", lambda group: 2)

    sequence_parallel.apply_sequence_parallelism(backbone, 2, object())

    attention = backbone.layers[0].self_attn
    assert attention._sp_world_size == 2
    assert attention._sp_group is not None


class _MambaLayer(nn.Module):
    def set_context_parallel_attributes(self, cp_group, cp_rank, cp_world_size):
        self.cp_group = cp_group
        self.cp_rank = cp_rank
        self.cp_world_size = cp_world_size


def test_generic_sequence_parallel_wires_nemotron_mamba(monkeypatch):
    backbone = _Backbone("nemotron_h", [None])
    mamba = _MambaLayer()
    backbone.layers.insert(0, mamba)
    group = object()
    monkeypatch.setattr(sequence_parallel, "get_language_model", lambda model: model)
    monkeypatch.setattr(sequence_parallel.dist, "get_world_size", lambda group: 8)
    monkeypatch.setattr(sequence_parallel.dist, "get_rank", lambda group: 3)

    sequence_parallel.apply_sequence_parallelism(backbone, 8, group)

    assert mamba.cp_group is group
    assert mamba.cp_rank == 3
    assert mamba.cp_world_size == 8
    attention = backbone.layers[1].self_attn
    assert attention._sp_group is group
    assert attention._sp_world_size == 8


def test_generic_sequence_parallel_requires_nemotron_mamba_layers(monkeypatch):
    backbone = _Backbone("nemotron_h", [None])
    monkeypatch.setattr(sequence_parallel, "get_language_model", lambda model: model)
    monkeypatch.setattr(sequence_parallel.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(sequence_parallel.dist, "get_rank", lambda group: 0)

    with pytest.raises(TypeError, match="no Mamba layers"):
        sequence_parallel.apply_sequence_parallelism(backbone, 2, object())


def test_nemotron_sp_stashes_global_cu_seqlens(monkeypatch):
    backbone = _Backbone("nemotron_h", [None])
    mamba = _MambaLayer()
    backbone.layers.insert(0, mamba)
    group = object()

    def all_gather(output_list, tensor, group=None):
        del group
        for buf in output_list:
            buf.copy_(tensor)

    monkeypatch.setattr(sequence_parallel, "get_language_model", lambda model: model)
    monkeypatch.setattr(sequence_parallel.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(sequence_parallel.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(sequence_parallel.dist, "all_gather", all_gather)

    sequence_parallel.apply_sequence_parallelism(backbone, 2, group)
    position_ids = torch.tensor([[0, 1, 2, 0, 1]])
    backbone(position_ids=position_ids)

    cu_seqlens = mamba._sp_cu_seqlens
    assert int(cu_seqlens[-1]) == position_ids.shape[1] * 2
    assert cu_seqlens.numel() > 2


def test_causal_conv_packed_resets_at_row_boundaries():
    weight = torch.tensor([[[1.0, 1.0, 1.0]]])
    hidden = torch.tensor([[[1.0], [1.0], [1.0], [9.0], [9.0], [9.0]]])
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)

    packed = _causal_conv_packed(hidden, weight, None, 1, cu_seqlens)
    left = _causal_conv_packed(hidden[:, :3], weight, None, 1, None)
    right = _causal_conv_packed(hidden[:, 3:], weight, None, 1, None)
    whole = _causal_conv_packed(hidden, weight, None, 1, None)

    assert torch.allclose(packed, torch.cat([left, right], dim=1))
    assert not torch.allclose(packed[:, 3:], whole[:, 3:])


def test_generic_sequence_parallel_requires_flash_attention(monkeypatch):
    backbone = _Backbone("qwen3_moe", [None])
    backbone.config._attn_implementation = "sdpa"
    monkeypatch.setattr(sequence_parallel, "get_language_model", lambda model: model)

    with pytest.raises(ValueError, match="requires flash_attention"):
        sequence_parallel.apply_sequence_parallelism(backbone, 2, object())
