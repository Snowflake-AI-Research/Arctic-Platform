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
"""BF16 experts run once per expert-parallel token chunk; their weight gradient is the sum of the chunks' gradients
rounded once, through the production dispatch path with only the cross-rank exchange replaced."""

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from arctic_platform.model.implementations.moe.distributed import expert_parallel
from arctic_platform.model.implementations.moe.layers import moe
from tests.model.test_flash_trainer_rounding import _OneRankDispatch

CHUNK, NUM_CHUNKS, DIM = 64, 8, 32


def _run(layer, x, upstream, chunk_size, checkpointed=False):
    layer.zero_grad(set_to_none=True)
    layer.set_deepep_token_chunk_size(chunk_size)
    x = x.clone().requires_grad_()
    out = checkpoint(layer, x[None], use_reentrant=False) if checkpointed else layer(x[None])
    (out[0].float() * upstream).sum().backward()
    return out, x.grad


@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize(("use_grouped_mm", "fp32_routed_weighting"), [(True, False), (True, True), (False, False)])
def test_chunked_expert_weight_gradients_are_rounded_once(
    monkeypatch, use_grouped_mm, fp32_routed_weighting, checkpointed
):
    histc = torch.histc
    monkeypatch.setattr(torch, "histc", lambda values, *args, **kwargs: histc(values.float(), *args, **kwargs))
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    monkeypatch.setattr(expert_parallel, "get_ep_group", lambda experts: None)
    monkeypatch.setattr(moe, "get_ep_comm_module", lambda backend: _OneRankDispatch())
    torch.manual_seed(0)
    args = moe.MoEArgs(
        num_experts=4,
        num_shared_experts=0,
        score_func="softmax",
        score_before_experts=False,
        top_k=2,
        use_grouped_mm=use_grouped_mm,
        load_balance_coeff=None,
        fp32_routed_weighting=fp32_routed_weighting,
    )
    layer = moe.MoE(args, dim=DIM, hidden_dim=DIM)
    with torch.no_grad():
        for parameter in layer.parameters():
            parameter.normal_(std=0.5)
    layer = layer.to(torch.bfloat16)
    weights = {name: getattr(layer.experts, name) for name in ("w1", "w2", "w3")}
    x = torch.randn(NUM_CHUNKS * CHUNK, DIM).to(torch.bfloat16)
    upstream = torch.randn(NUM_CHUNKS * CHUNK, DIM)

    fires = dict.fromkeys(weights, 0)
    for name, weight in weights.items():  # the hook DeepSpeed reduces gradients from
        weight.register_post_accumulate_grad_hook(lambda _, name=name: fires.__setitem__(name, fires[name] + 1))
    out, input_grad = _run(layer, x, upstream, CHUNK, checkpointed)
    assert fires == dict.fromkeys(weights, 1)
    grads = {name: weight.grad for name, weight in weights.items()}

    exact = dict.fromkeys(weights, 0)
    for start in range(0, len(x), CHUNK):
        _run(layer, x[start : start + CHUNK], upstream[start : start + CHUNK], None)
        for name, weight in weights.items():
            exact[name] = exact[name] + weight.grad.double()
    for name in weights:  # FP32 summation error is far below a BF16 rounding: the exact sum, rounded once
        torch.testing.assert_close(grads[name], exact[name].to(torch.bfloat16), rtol=0, atol=0)

    # The forward and the input gradient are those of the experts read directly, as without chunks.
    for weight in weights.values():
        weight.requires_grad_(False)
    frozen_out, frozen_input_grad = _run(layer, x, upstream, CHUNK, checkpointed)
    torch.testing.assert_close(out, frozen_out, rtol=0, atol=0)
    torch.testing.assert_close(input_grad, frozen_input_grad, rtol=0, atol=0)
