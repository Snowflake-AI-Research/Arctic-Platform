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

import torch

from arctic_platform.model.implementations.vllm_aligned.glm5_next import _AlignedLayer


def test_aligned_layer_returns_the_aligned_value_with_the_stock_gradient():
    torch.manual_seed(0)
    stock = torch.nn.Linear(6, 6)
    streams = torch.randn(5, 6, requires_grad=True)
    nudge = torch.full((5, 6), 1e-3)

    out = _AlignedLayer.apply(streams, lambda s: torch.tanh(stock(s)) + nudge, lambda s: torch.tanh(stock(s)))
    assert torch.equal(out, torch.tanh(stock(streams)).detach() + nudge)
    weights = torch.randn(5, 6)
    (out * weights).sum().backward()

    want_streams = streams.detach().clone().requires_grad_(True)
    reference = torch.nn.Linear(6, 6)
    reference.load_state_dict(stock.state_dict())
    (torch.tanh(reference(want_streams)) * weights).sum().backward()
    assert torch.equal(streams.grad, want_streams.grad)
    assert torch.equal(stock.weight.grad, reference.weight.grad)
    assert torch.equal(stock.bias.grad, reference.bias.grad)


def test_aligned_layers_chain_like_checkpointed_layers():
    torch.manual_seed(1)
    layers = [torch.nn.Linear(4, 4) for _ in range(3)]
    x = torch.randn(3, 4, requires_grad=True)
    h = x
    for layer in layers:
        h = _AlignedLayer.apply(h, lambda s, layer=layer: torch.relu(layer(s)), lambda s, layer=layer: torch.relu(layer(s)))
    h.sum().backward()
    y = x.detach().clone().requires_grad_(True)
    clones = [torch.nn.Linear(4, 4) for _ in layers]
    for clone, layer in zip(clones, layers):
        clone.load_state_dict(layer.state_dict())
    g = y
    for clone in clones:
        g = torch.relu(clone(g))
    g.sum().backward()
    assert torch.equal(x.grad, y.grad)
    for clone, layer in zip(clones, layers):
        assert torch.equal(clone.weight.grad, layer.weight.grad)


def test_merged_fp8_stacks_weights_and_block_scales():
    import types

    import pytest

    from arctic_platform.model.implementations.vllm_aligned.glm5_next import _merged_fp8

    def linear(rows, cols, fill):
        return types.SimpleNamespace(weight=torch.full((rows, cols), fill), weight_scale_inv=torch.full((rows // 128, 1), fill))

    weight, scale = _merged_fp8(linear(256, 128, 1.0), linear(128, 128, 2.0))
    assert weight.shape == (384, 128) and scale.shape == (3, 1)
    assert scale[:, 0].tolist() == [1.0, 1.0, 2.0]
    with pytest.raises(ValueError):
        _merged_fp8(linear(100, 128, 1.0), linear(128, 128, 2.0))


def test_causal_topk_and_buffer_width_follow_vllm():
    import types

    from arctic_platform.model.implementations.vllm_aligned.glm5_next import causal_topk
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import topk_buffer_width

    assert topk_buffer_width(types.SimpleNamespace(index_topk=2048, index_kpool=4)) == 2176
    rows = causal_topk(torch.tensor([0, 2]), 4)
    assert rows.tolist() == [[0, -1, -1, -1], [0, 1, 2, -1]]


def test_mla_kv_up_weights_split_kv_b_proj_per_head():
    import types

    from arctic_platform.model.implementations.vllm_aligned.glm5_next import mla_kv_up_weights

    config = types.SimpleNamespace(num_attention_heads=2, kv_lora_rank=3, qk_nope_head_dim=4, v_head_dim=5)
    weight = torch.arange(2 * 9 * 3, dtype=torch.float32).view(2 * 9, 3)
    w_uk_t, w_uv = mla_kv_up_weights(types.SimpleNamespace(weight=weight), config)
    assert w_uk_t.shape == (2, 4, 3) and w_uv.shape == (2, 3, 5)
    per_head = weight.view(2, 9, 3)
    assert torch.equal(w_uk_t[1], per_head[1, :4])
    assert torch.equal(w_uv[1], per_head[1, 4:].t())


def test_dequant_fp8_block_matches_vllm_formula_with_partial_blocks():
    from arctic_platform.model.implementations.vllm_aligned.glm5_next import dequant_fp8_block

    torch.manual_seed(2)
    weight = torch.randn(200, 130).to(torch.float8_e4m3fn)
    scale = torch.rand(2, 2) + 0.5
    got = dequant_fp8_block(weight, scale)
    full = scale.repeat_interleave(128, 0).repeat_interleave(128, 1)[:200, :130]
    assert got.dtype == torch.bfloat16 and torch.equal(got, (weight.float() * full).to(torch.bfloat16))
