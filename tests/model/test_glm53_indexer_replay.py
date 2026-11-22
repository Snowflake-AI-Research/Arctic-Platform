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

from arctic_platform.common.glm5next_indexer_replay import capture_pools
from arctic_platform.common.glm5next_indexer_replay import pool_rows
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


def _attended(model, tokens, positions, boundaries, routed_experts):
    from arctic_platform.model.implementations.glm53 import context_parallel as cp

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


def test_trainer_selects_the_samplers_tokens_in_packed_rows():
    model = _model()
    moe_routes = _record_routes(model)
    text = model.config.text_config
    kpool, top_k = text.index_kpool, text.num_experts_per_tok
    select, rows = text.index_topk // kpool, pool_rows(text)
    length = sum(ROWS)
    tokens = torch.randint(2, text.vocab_size, (1, length), generator=torch.Generator().manual_seed(0))
    positions = torch.cat([torch.arange(n) for n in ROWS])[None]
    boundaries = torch.tensor([0, *torch.tensor(ROWS).cumsum(0).tolist()], dtype=torch.int32)
    routes = torch.randint(
        0, text.n_routed_experts, (1, length, LAYERS, top_k), generator=torch.Generator().manual_seed(1)
    )
    own = _attended(model, tokens, positions, boundaries, routes)

    # The sampler's choice for each sparse query: pools the trainer's own indexer did not keep, written the way vLLM's
    # indexer leaves them (row-relative token ids, pool by pool, then the tail; -1 unused), through the capture rows.
    capture = torch.zeros(length, text.num_hidden_layers + LAYERS * rows, top_k, dtype=torch.int32)
    capture[:, :LAYERS] = routes[0]
    expected = [[] for _ in range(LAYERS)]
    generator = torch.Generator().manual_seed(2)
    for start, n in zip(boundaries[:-1].tolist(), ROWS):
        for layer in range(LAYERS):
            buffer = torch.full((n, select * kpool + kpool - 1), -1, dtype=torch.int32)
            for t in range(n):
                visible = (t + 1) // kpool
                kept = {(token - start) // kpool for token in own[layer][start + t] if token < start + visible * kpool}
                pools = sorted(
                    sorted(torch.randperm(visible, generator=generator).tolist(), key=kept.__contains__)[:select]
                )
                ids = [p * kpool + o for p in pools for o in range(kpool)] + list(range(visible * kpool, t + 1))
                buffer[t, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
                expected[layer].append({start + i for i in ids})

            def write(row, ids, _start=start, _n=n):
                capture[_start : _start + _n, row] = ids

            capture_pools(write, text.num_hidden_layers + layer * rows, buffer, text.index_topk, kpool, top_k)
    # The cache narrows the capture to uint16 and the trainer widens the replay window to int64, as for routes.
    window = capture.to(torch.uint16).to(torch.int64)[None]
    for pos, fill in ((18, 0), (20, -1)):  # outside the capture, and a fresh row
        window[:, pos, LAYERS : LAYERS + rows] = fill
        expected[0][pos] = own[0][pos]  # the first layer sees the same input in both forwards
    replayed = _attended(model, tokens, positions, boundaries, window)

    for layer in range(LAYERS):  # the pool rows leave each layer's expert routes as they were
        torch.testing.assert_close(moe_routes[LAYERS + layer], moe_routes[layer])
    for layer in range(LAYERS):
        assert replayed[layer] == expected[layer]
        # The selections differ for more than a third of the queries, so the assertion above can fail.
        assert sum(a != b for a, b in zip(own[layer], expected[layer])) > length // 3
    with pytest.raises(ValueError, match="rows per token"):  # a capture of another width cannot be split
        _attended(model, tokens, positions, boundaries, capture[None, :, :-1].to(torch.int64))
