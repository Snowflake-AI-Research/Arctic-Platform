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
"""The fused-projection GatedDeltaNet forward against the stock transformers forward on a packed call, with the fp32 beta
gate."""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from arctic_platform.model.implementations.gpu import gated_delta_net_fused
from arctic_platform.model.implementations.gpu.gated_delta_net_fused import apply_fused_gated_delta_net
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import set_seed


def _config(value_heads: int = 4, key_heads: int = 2, hidden: int = 32, head_dim: int = 8) -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        hidden_size=hidden,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        intermediate_size=64,
        vocab_size=64,
        linear_num_value_heads=value_heads,
        linear_num_key_heads=key_heads,
        linear_key_head_dim=head_dim,
        linear_value_head_dim=head_dim,
        linear_conv_kernel_dim=4,
        layer_types=["linear_attention"],
    )


def _module(config: Qwen3_5TextConfig, dtype: torch.dtype, device: str, seed: int = 0) -> nn.Module:
    set_seed(seed)
    module = modeling_qwen3_5.Qwen3_5GatedDeltaNet(config, layer_idx=0)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.normal_(0.0, 0.3)
    return module.to(device=device, dtype=dtype).train()


def _run(module: nn.Module, hidden: torch.Tensor, packed: dict):
    module.zero_grad(set_to_none=True)
    hidden = hidden.detach().clone().requires_grad_(True)
    # The fp32 parity patch's hook: in_proj_b returns fp32, so the stock beta is a sigmoid of fp32 b.
    handle = module.in_proj_b.register_forward_hook(lambda m, a, out: out.float())
    out = module(hidden, **packed)
    handle.remove()
    (out.float() * torch.linspace(-1, 1, out.shape[-1], device=out.device)).pow(2).sum().backward()
    grads = {name: p.grad.detach().clone() for name, p in module.named_parameters()}
    return out.detach(), hidden.grad.detach().clone(), grads


def _max_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


@require_torch_gpu
def test_fused_forward_matches_stock_gpu_bf16():
    pytest.importorskip("fla")
    pytest.importorskip("causal_conv1d")
    config = _config(48, 16, hidden=512, head_dim=128)  # Qwen3.8-27B's 48 value / 16 key heads
    stock = _module(config, torch.bfloat16, "cuda")
    reference = _module(config, torch.float32, "cuda")
    reference.load_state_dict({k: v.float() for k, v in stock.state_dict().items()})
    fused = copy.deepcopy(stock)
    apply_fused_gated_delta_net(fused, fp32_beta=True)
    assert gated_delta_net_fused._FUSED_ATTRIBUTE in vars(fused)
    hidden = torch.randn(1, 2048, config.hidden_size, device="cuda", generator=torch.Generator("cuda").manual_seed(3))
    hidden = (hidden * 0.5).bfloat16()
    # Three rows packed as DSS sends them: a row's convolution must not read the end of the row before it.
    cu = torch.tensor([0, 384, 1024, 2048], device="cuda", dtype=torch.int32)
    rows = torch.repeat_interleave(torch.arange(3, device="cuda", dtype=torch.int32), cu.diff().long())
    packed = dict(cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=1024, max_length_k=1024, seq_idx=rows[None])
    expected = _run(stock, hidden, packed)
    got = _run(fused, hidden, packed)
    exact = _run(reference, hidden.float(), packed)
    # The fused forward must be as close to the fp32 forward as the stock bf16 forward is, up to bf16 rounding noise:
    # its distance from the stock bf16 forward is bounded by the stock forward's own bf16 error.
    for index, label in ((0, "output"), (1, "input grad")):
        stock_error = _max_error(expected[index], exact[index])
        assert _max_error(got[index], exact[index]) <= 1.5 * stock_error + 1e-6, label
        assert _max_error(got[index], expected[index]) <= 2.0 * stock_error + 1e-6, label
    for name in expected[2]:
        stock_error = _max_error(expected[2][name], exact[2][name])
        assert _max_error(got[2][name], exact[2][name]) <= 1.5 * stock_error + 1e-6, name
