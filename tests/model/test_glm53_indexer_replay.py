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

"""The GLM-5.3 trainer attends to exactly the tokens the sampler's indexer selected, carried in the routed-experts
capture: sampler capture rows -> narrowed capture -> packed replay window -> decoder layers -> sparse-MLA indices."""

import types
from unittest.mock import patch

import pytest
import torch

from arctic_platform.model.implementations.glm53 import context_parallel as cp
from arctic_platform.testing_utils import set_seed
from tests.model.test_glm5_next import _tiny_config

ROWS = (23, 6, 19)
LAYERS = 2


def _model():
    from arctic_platform.model.implementations.glm53.context_parallel import apply_context_parallelism
    from arctic_platform.model.implementations.glm53.modeling_glm5_next import Glm5NextForConditionalGenerationPrimeRL

    config = _tiny_config()
    text = config.text_config
    sparse = next(kind for kind in text.layer_types if kind != "linear_attention")
    text.num_hidden_layers = LAYERS
    text.layer_types = [sparse] * LAYERS
    text.indexer_types = ["full"] * LAYERS
    text.mlp_layer_types = ["sparse"] * LAYERS
    config.use_grouped_mm = False
    set_seed(0)
    model = Glm5NextForConditionalGenerationPrimeRL(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.5)
    model = model.to(torch.bfloat16).eval()
    apply_context_parallelism(model, 1, None)
    return model


def _attended(model, tokens, positions, boundaries, routed_experts, **kwargs):
    seen, pad = [], cp._pad_sparse_indices

    def record(indices, *args, **kwargs):
        seen.append([{int(t) for t in row if t >= 0} for row in indices[0]])
        return pad(indices, *args, **kwargs)

    with patch.object(cp, "_pad_sparse_indices", record), torch.no_grad():
        model(
            input_ids=tokens,
            position_ids=positions,
            cu_seq_lens_q=boundaries,
            routed_experts=routed_experts,
            use_cache=False,
            **kwargs,
        )
    return seen


def _record_routes(model):
    """Expert dispatch needs GPUs; each MoE records the routes it receives and adds nothing."""
    seen = []

    def forward(self, hidden_states, routed_experts=None):
        seen.append(routed_experts)
        return torch.zeros_like(hidden_states)

    for layer in model.model.language_model.layers:
        layer.mlp.forward = types.MethodType(forward, layer.mlp)
    return seen


def test_trainer_replays_packed_selections():
    model = _model()
    moe_routes = _record_routes(model)
    text = model.config.text_config
    kpool, top_k = text.index_kpool, text.num_experts_per_tok
    length = sum(ROWS)
    tokens = torch.randint(2, text.vocab_size, (1, length))
    positions = torch.cat([torch.arange(n) for n in ROWS])[None]
    boundaries = torch.tensor([0, *torch.tensor(ROWS).cumsum(0).tolist()], dtype=torch.int32)
    routes = torch.randint(0, text.n_routed_experts, (1, length, LAYERS, top_k))
    own = _attended(model, tokens, positions, boundaries, routes)

    # The tiny model's two pool codes fill one extra routing row per attention layer.
    matching, different = torch.zeros_like(routes), torch.zeros_like(routes)
    expected = [[] for _ in range(LAYERS)]
    for start, n in zip(boundaries.tolist(), ROWS):
        for t in range(n):
            at = start + t
            visible = (t + 1) // kpool
            for layer in range(LAYERS):
                kept = {(token - start) // kpool for token in own[layer][at] if token < start + visible * kpool}
                matching[0, at, layer, : len(kept)] = torch.tensor(sorted(p + 1 for p in kept))
                pools = sorted(range(visible), key=kept.__contains__)[:top_k]
                if layer == 0 and at in (18, 20):  # cache miss and best-effort fresh row
                    different[0, at, layer].fill_(0 if at == 18 else -1)
                    expected[layer].append(own[layer][at])
                    continue
                if layer == 0 and at == 15:
                    pools = pools[:1]  # fewer pools than the trainer's own choice
                different[0, at, layer, : len(pools)] = torch.tensor([p + 1 for p in pools])
                expected[layer].append(
                    {start + p * kpool + o for p in pools for o in range(kpool)}
                    | set(range(start + visible * kpool, at + 1))
                )

    window = torch.cat([routes, different], dim=2)
    replayed = _attended(model, tokens, positions, boundaries, window)
    for layer in range(LAYERS):
        assert sum(a != b for a, b in zip(own[layer], expected[layer])) > length // 3
        assert replayed[layer] == expected[layer]
        torch.testing.assert_close(moe_routes[LAYERS + layer], moe_routes[layer])
    # Captured pools equal to the trainer's own leave its selection unchanged, through the uint16 record.
    baseline = torch.cat([routes, matching], dim=2).to(torch.uint16).long()
    assert _attended(model, tokens, positions, boundaries, baseline) == own
    with pytest.raises(ValueError, match="rows per token"):
        _attended(model, tokens, positions, boundaries, window[:, :, :-1])


def test_replay_left_padded_row():
    model = _model()
    _record_routes(model)
    tokens = torch.arange(11)[None]
    mask = tokens >= 2
    positions = (tokens - 2).clamp_min(0)
    boundaries = torch.tensor([0, 11], dtype=torch.int32)
    window = torch.zeros((1, 11, 2 * LAYERS, 2), dtype=torch.long)
    # Query 10 sees nine real tokens. Replay the first and fourth complete pool and its one-token tail.
    window[:, -1, LAYERS:] = torch.tensor([1, 4])
    attended = _attended(model, tokens, positions, boundaries, window, attention_mask=mask)
    for layer in attended:
        assert layer[-1] == {2, 3, 8, 9, 10}


def test_replay_on_a_rank_starting_inside_a_packed_row(monkeypatch):
    model = _model()
    _record_routes(model)
    backbone = model.model.language_model
    backbone._cp_world_size = 2
    for layer in backbone.layers:
        layer.self_attn._cp_rank = 1
        layer.self_attn._cp_world_size = 2

    # Supply both shards of each gathered tensor. Pool choice below is independent of the prefix's weights.
    def gather(local, group):
        return torch.cat([local, local], dim=1)

    monkeypatch.setattr(cp, "gather_sequence", gather)
    monkeypatch.setattr(cp, "_gather_sequence_no_grad", gather)
    boundaries = torch.tensor([0, 2, 10], dtype=torch.int32)
    positions = torch.arange(3, 8)[None]
    window = torch.zeros(1, 5, 2 * LAYERS, 2, dtype=torch.long)
    window[:, :, LAYERS:, 0] = 1
    attended = _attended(model, positions + 1, positions, boundaries, window)
    for layer in attended:
        assert layer == [{2, 3}, {2, 3, 6}, {2, 3}, {2, 3, 8}, {2, 3}]
