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
"""The fused Qwen3.8 hyper connection against Hugging Face's ``Qwen4ExpTextGatedResidual`` and injection."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("transformers.models.qwen4_exp")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused hyper connection runs on CUDA")


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _module(hidden=2560, hc=4, lowrank=320):
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

    torch.manual_seed(0)
    config = Qwen4ExpTextConfig(hidden_size=hidden, hc_count=hc, hc_lowrank=lowrank, rms_norm_eps=1e-6)
    module = Qwen4ExpTextGatedResidual(config).cuda()
    for p in module.parameters():  # trained-looking weights instead of the zero / default init
        p.data.normal_(0, 0.02 if p.dim() > 1 else 0.3)
    return module.bfloat16().requires_grad_(False)


def _sublayer(mix, inject, module, x, out_weight):
    mixed, hyper_input, inj = mix(module, x)
    out = torch.tanh(mixed @ out_weight)  # stands in for the attention / MoE block
    return inject(hyper_input, out, inj), mixed, inj


@pytest.mark.parametrize("fp32_parity", [False, True])
@pytest.mark.parametrize("tokens", [1, 257])
def test_fused_hyper_connection_matches_the_module(tokens, fp32_parity):
    from arctic_platform.model.implementations.qwen38 import modeling_qwen4_exp as qwen38

    module = _module()
    if fp32_parity:
        module.__class__ = qwen38.Qwen4ExpTextGatedResidualPrimeRL
    torch.manual_seed(1)
    x = torch.randn(1, tokens, 4 * 2560, device="cuda").bfloat16()
    out_weight = (torch.randn(2560, 2560, device="cuda") * 0.02).bfloat16()
    grad = torch.randn(1, tokens, 4 * 2560, device="cuda").bfloat16()

    def reference_inject(hyper_input, out, inj):
        if fp32_parity:
            return module.combine(hyper_input, out, inj)
        return hyper_input + (out.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)

    results = {}
    for name, mix, inject in (
        ("hf", lambda m, h: m(h), reference_inject),
        ("fused", qwen38._mix_streams, lambda h, o, i: qwen38._combine(module, h, o, i)),
    ):
        leaf = x.clone().requires_grad_(True)
        y, mixed, inj = _sublayer(mix, inject, module, leaf, out_weight)
        y.backward(grad)
        results[name] = (y.detach(), mixed.detach(), inj.detach(), leaf.grad)
    for what, fused, reference in zip(("y", "mixed", "inj", "dx"), results["fused"], results["hf"]):
        assert fused.shape == reference.shape, what
        assert _rel(fused, reference) < 1e-2, (what, _rel(fused, reference))


@pytest.mark.parametrize("tokens", [1, 257])
def test_fused_hyper_connection_rounds_once_with_fp32_parity(tokens):
    """The fused mix is the fp64 mix of its own BF16 GEMM outputs rounded once, and it hands ``combine`` the
    injection logits, as vLLM's ``hc_gate_mix`` and ``hc_combine`` do."""
    from arctic_platform.model.implementations.qwen38 import hyper_connection
    from arctic_platform.model.implementations.qwen38 import modeling_qwen4_exp as qwen38

    module = _module()
    module.__class__ = qwen38.Qwen4ExpTextGatedResidualPrimeRL
    torch.manual_seed(1)
    x = torch.randn(tokens, 4 * 2560, device="cuda").bfloat16()
    mixed, _, logits = qwen38._mix_streams(module, x)

    normed = torch.empty_like(x)
    rstd = torch.empty(tokens * 4, device="cuda")
    hyper_connection._norm_fwd_kernel[(tokens * 4,)](
        x, module.hc_norm.weight, normed, rstd, 2560, module.hc_norm.eps, HC=4, BLOCK=4096, num_warps=8
    )
    rank = module.input_mix_weight_up.weight.shape[1]
    down_inject = normed @ hyper_connection._stacked_down_inject(module).t()
    gate = module.input_mix_weight_up(F.silu(down_inject[:, :rank].float() / 4).bfloat16())
    streams = (4, -1)
    reference = (torch.sigmoid(gate.double()).unflatten(-1, streams) * normed.double().unflatten(-1, streams)).mean(-2)
    torch.testing.assert_close(logits, down_inject[:, rank:], rtol=0, atol=0)
    # Triton's sigmoid is not correctly rounded; the Hugging Face rounding changes about half the values.
    assert int((mixed != reference.bfloat16()).sum()) <= mixed.numel() // 5000


def test_trainable_hyper_connection_keeps_the_hugging_face_path():
    from arctic_platform.model.implementations.qwen38 import hyper_connection

    module = _module().requires_grad_(True)
    x = torch.randn(1, 3, 4 * 2560, device="cuda").bfloat16()
    assert not hyper_connection.fused_hyper_connection_supported(module, x)
