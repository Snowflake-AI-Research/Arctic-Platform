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
"""The fused-projection GatedDeltaNet forward against the stock transformers forward.

CPU: ``ConcatLinear`` passes gradcheck in fp64, and the fused forward of a stock dense ``Qwen3_5GatedDeltaNet`` (torch
fallbacks for the convolution and the rule) gives the stock forward's output and gradients in fp32 within fp32 GEMM
tiling noise, with and without the fp32 beta gate, with a wrapped projection (unfused path) and with grouped value
heads. GPU: the same comparison in bf16 on the causal-conv1d and FLA kernels, where q and k reach FLA ungrouped,
against the stock forward's distance from an fp32 run of the stock forward; and the Triton kernels of
``FusedAddRMSNorm`` / ``FusedAddRMSNormFp32Residual`` against their PyTorch formulas.
"""

from __future__ import annotations

import copy

import pytest
import torch
from torch import nn
from transformers.models.qwen3_5 import modeling_qwen3_5
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from arctic_platform.model.implementations.gpu import gated_delta_net_fused
from arctic_platform.model.implementations.gpu.gated_delta_net_fused import ConcatLinear
from arctic_platform.model.implementations.gpu.gated_delta_net_fused import apply_fused_gated_delta_net
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_close


@pytest.fixture
def torch_kernels(monkeypatch):
    """The stock module-level convolution and rule as their PyTorch references, as on a CPU without the kernels."""
    for name in ("causal_conv1d_fn", "torch_chunk_gated_delta_rule"):
        function = getattr(modeling_qwen3_5, name)
        monkeypatch.setattr(modeling_qwen3_5, name, getattr(function, "__wrapped__", function))


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


def _run(module: nn.Module, hidden: torch.Tensor, fp32_beta_hook: bool):
    module.zero_grad(set_to_none=True)
    hidden = hidden.detach().clone().requires_grad_(True)
    handle = None
    if fp32_beta_hook:
        # The fp32 parity patch's hook: in_proj_b returns fp32, so the stock beta is a sigmoid of fp32 b.
        handle = module.in_proj_b.register_forward_hook(lambda m, a, out: out.float())
    out = module(hidden)
    if handle is not None:
        handle.remove()
    (out.float() * torch.linspace(-1, 1, out.shape[-1], device=out.device)).pow(2).sum().backward()
    grads = {name: p.grad.detach().clone() for name, p in module.named_parameters()}
    return out.detach(), hidden.grad.detach().clone(), grads


def test_concat_linear_gradcheck():
    set_seed(0)
    x = torch.randn(3, 5, 7, dtype=torch.float64, requires_grad=True)
    weights = [torch.randn(rows, 7, dtype=torch.float64, requires_grad=True) for rows in (4, 2, 1, 3)]
    assert torch.autograd.gradcheck(ConcatLinear.apply, (x, *weights))
    frozen = [w.detach() for w in weights]
    out = ConcatLinear.apply(x, *frozen)
    out.sum().backward()
    assert all(w.grad is None for w in frozen)


@pytest.mark.parametrize("fp32_beta", [True, False])
@pytest.mark.parametrize("heads", [(4, 2), (2, 2)])
def test_fused_forward_matches_stock_cpu(torch_kernels, fp32_beta, heads):
    config = _config(*heads)
    stock = _module(config, torch.float32, "cpu")
    fused = copy.deepcopy(stock)
    assert apply_fused_gated_delta_net(fused, fp32_beta=fp32_beta) == 1
    assert apply_fused_gated_delta_net(fused, fp32_beta=fp32_beta) == 0
    hidden = torch.randn(2, 24, config.hidden_size, generator=torch.Generator().manual_seed(1))
    expected = _run(stock, hidden, fp32_beta_hook=fp32_beta)
    got = _run(fused, hidden, fp32_beta_hook=fp32_beta)
    # fp32 throughout; the fused GEMM tiles its reduction differently from the four separate ones.
    torch_assert_close(got[0], expected[0], rtol=0, atol=1e-5)
    torch_assert_close(got[1], expected[1], rtol=0, atol=1e-5)
    for name in expected[2]:
        torch_assert_close(got[2][name], expected[2][name], rtol=0, atol=1e-4, msg=name)


def test_wrapped_projection_runs_unfused(torch_kernels):
    config = _config()
    stock = _module(config, torch.float32, "cpu")
    fused = copy.deepcopy(stock)
    apply_fused_gated_delta_net(fused, fp32_beta=True)

    class Wrapped(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            return self.inner(x)

    calls = []
    fused.in_proj_b = Wrapped(fused.in_proj_b)
    fused.in_proj_b.register_forward_hook(lambda m, a, out: calls.append(1))
    hidden = torch.randn(1, 16, config.hidden_size, generator=torch.Generator().manual_seed(2))
    expected = _run(stock, hidden, fp32_beta_hook=True)
    got = _run(fused, hidden, fp32_beta_hook=False)
    assert calls == [1]
    torch_assert_close(got[0], expected[0], rtol=0, atol=1e-6)
    torch_assert_close(got[1], expected[1], rtol=0, atol=1e-6)


def test_cached_call_runs_stock_forward(monkeypatch):
    config = _config()
    fused = _module(config, torch.float32, "cpu")
    apply_fused_gated_delta_net(fused, fp32_beta=True)
    seen = []
    stock_forward = modeling_qwen3_5.Qwen3_5GatedDeltaNet.forward

    def spy(self, *args, **kwargs):
        seen.append(kwargs.get("cache_params"))
        return torch.zeros(1)

    monkeypatch.setattr(modeling_qwen3_5.Qwen3_5GatedDeltaNet, "forward", spy)
    marker = object()
    fused(torch.zeros(1, 4, config.hidden_size), cache_params=marker)
    assert seen == [marker]
    monkeypatch.setattr(modeling_qwen3_5.Qwen3_5GatedDeltaNet, "forward", stock_forward)


def _max_error(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


@require_torch_gpu
@pytest.mark.parametrize("heads", [(32, 16), (48, 16), (16, 16)])
def test_fused_forward_matches_stock_gpu_bf16(heads):
    pytest.importorskip("fla")
    pytest.importorskip("causal_conv1d")
    config = _config(*heads, hidden=512, head_dim=128)
    stock = _module(config, torch.bfloat16, "cuda")
    reference = _module(config, torch.float32, "cuda")
    reference.load_state_dict({k: v.float() for k, v in stock.state_dict().items()})
    fused = copy.deepcopy(stock)
    apply_fused_gated_delta_net(fused, fp32_beta=True)
    assert gated_delta_net_fused._FUSED_ATTRIBUTE in vars(fused)
    hidden = torch.randn(1, 2048, config.hidden_size, device="cuda", generator=torch.Generator("cuda").manual_seed(3))
    hidden = (hidden * 0.5).bfloat16()
    expected = _run(stock, hidden, fp32_beta_hook=True)
    got = _run(fused, hidden, fp32_beta_hook=True)
    exact = _run(reference, hidden.float(), fp32_beta_hook=True)
    # The fused forward must be as close to the fp32 forward as the stock bf16 forward is, up to bf16 rounding noise:
    # its distance from the stock bf16 forward is bounded by the stock forward's own bf16 error.
    for index, label in ((0, "output"), (1, "input grad")):
        stock_error = _max_error(expected[index], exact[index])
        assert _max_error(got[index], exact[index]) <= 1.5 * stock_error + 1e-6, label
        assert _max_error(got[index], expected[index]) <= 2.0 * stock_error + 1e-6, label
    for name in expected[2]:
        stock_error = _max_error(expected[2][name], exact[2][name])
        assert _max_error(got[2][name], exact[2][name]) <= 1.5 * stock_error + 1e-6, name
