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
