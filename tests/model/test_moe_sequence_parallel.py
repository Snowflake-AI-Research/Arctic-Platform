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
from torch import nn

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


def test_generic_sequence_parallel_rejects_nemotron(monkeypatch):
    backbone = _Backbone("nemotron_h", [None])
    monkeypatch.setattr(sequence_parallel, "get_language_model", lambda model: model)

    with pytest.raises(NotImplementedError, match="Mamba context parallelism"):
        sequence_parallel.apply_sequence_parallelism(backbone, 2, object())


def test_generic_sequence_parallel_requires_flash_attention(monkeypatch):
    backbone = _Backbone("qwen3_moe", [None])
    backbone.config._attn_implementation = "sdpa"
    monkeypatch.setattr(sequence_parallel, "get_language_model", lambda model: model)

    with pytest.raises(ValueError, match="requires flash_attention"):
        sequence_parallel.apply_sequence_parallelism(backbone, 2, object())
