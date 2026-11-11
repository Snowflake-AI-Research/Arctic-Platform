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

from arctic_platform.inference.vllm.glm5next_indexer_replay import capture_pools
from arctic_platform.inference.vllm.glm5next_indexer_replay import pool_rows
from arctic_platform.inference.vllm.glm5next_indexer_replay import sparse_layers
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
    """The token indices each sparse-attention layer hands to the sparse-MLA kernel."""
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
        for layer, layer_idx in enumerate(sparse_layers(text)):
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
    replayed = _attended(model, tokens, positions, boundaries, capture.to(torch.uint16).to(torch.int64)[None])

    assert len(replayed) == LAYERS
    for layer in range(LAYERS):  # the pool rows leave each layer's expert routes as they were
        torch.testing.assert_close(moe_routes[LAYERS + layer], moe_routes[layer])
    for layer in range(LAYERS):
        assert replayed[layer] == expected[layer]
        # The two selections differ for most sparse queries, so the assertion above can fail.
        assert sum(a != b for a, b in zip(own[layer], expected[layer])) > length // 3
    with pytest.raises(ValueError, match="rows per token"):  # a capture of another width cannot be split
        _attended(model, tokens, positions, boundaries, capture[None, :, :-1].to(torch.int64))


def test_sampler_captures_each_sparse_layers_pools_past_the_decoder_layers(monkeypatch):
    """The vLLM glue: with the key set, the capture grows by the pool rows, and binding the capturer makes each main-model
    glm5next indexer op write the pools it selected at its sparse layer's first row (MTP indexers stay unbound)."""
    aux_worker = pytest.importorskip("vllm.distributed.aux_output_connector.worker")
    from vllm.model_executor.layers.fused_moe import routed_experts_capturer

    from arctic_platform.inference.vllm import glm5next_indexer_replay as replay

    text = types.SimpleNamespace(
        layer_types=["linear_attention", "deepseek_sparse_attention"] * 2,
        index_topk=8,
        index_kpool=2,
        num_experts_per_tok=2,
    )
    layers, rows = len(text.layer_types), pool_rows(text)
    model_config = types.SimpleNamespace(
        hf_text_config=text,
        max_model_len=1024,
        get_total_num_hidden_layers=lambda: layers,
        get_num_experts=lambda: 40,
        get_num_experts_per_tok=lambda: text.num_experts_per_tok,
    )
    vllm_config = types.SimpleNamespace(model_config=model_config, additional_config={replay.INDEXER_REPLAY_KEY: True})
    selected = torch.tensor([[4, 5, 0, 1, 10, 11, -1, -1, 12], [2, 3, 8, 9, 6, 7, 0, 1, -1]], dtype=torch.int32)

    class Op(torch.nn.Module):  # a CustomOp calls the forward method it bound at construction
        def __init__(self):
            super().__init__()
            self._forward_method = lambda hidden_states, *args, **kwargs: selected

        def forward(self, *args, **kwargs):
            return self._forward_method(*args, **kwargs)

    class Indexer(torch.nn.Module):  # the attributes of vLLM's glm5next Indexer the glue reads
        def __init__(self, layer):
            super().__init__()
            self.vllm_config, self.config, self.indexer_op = vllm_config, text, Op()
            self.prefix = f"language_model.model.layers.{layer}.self_attn.indexer"
            self.topk_tokens, self.index_kpool = text.index_topk, text.index_kpool

    for owner, name in (
        (routed_experts_capturer, "_get_routed_experts_shape"),
        (routed_experts_capturer, "bind_routed_experts_capturer"),
        (aux_worker, "bind_routed_experts_capturer"),
    ):
        monkeypatch.setattr(owner, name, getattr(owner, name))
    replay.ensure_glm5next_indexer_replay()
    assert routed_experts_capturer._get_routed_experts_shape(vllm_config) == (layers + 2 * rows, 257, 2)
    off = types.SimpleNamespace(model_config=model_config, additional_config={})
    assert routed_experts_capturer._get_routed_experts_shape(off) == (layers, 40, 2)

    model = torch.nn.ModuleList([Indexer(1), Indexer(3), Indexer(layers)])  # the last one is an MTP layer's
    writes = {}
    aux_worker.bind_routed_experts_capturer(
        model, types.SimpleNamespace(capture=lambda row, ids: writes.update({row: ids.tolist()}))
    )
    for indexer in model:
        indexer.indexer_op(torch.zeros(2, 4))

    pools = [[2, 0, 5, -1], [1, 4, 3, 0]]
    expected = {
        layers + s * rows + r: [pools[q][r * 2 : r * 2 + 2] for q in range(2)] for s in range(2) for r in range(rows)
    }
    assert writes == expected
