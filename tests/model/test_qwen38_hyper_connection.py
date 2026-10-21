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


@pytest.mark.parametrize("tokens", [1, 257])
def test_fused_hyper_connection_matches_hugging_face(tokens):
    from arctic_platform.model.implementations.qwen38 import modeling_qwen4_exp as qwen38

    module = _module()
    torch.manual_seed(1)
    x = torch.randn(1, tokens, 4 * 2560, device="cuda").bfloat16()
    out_weight = (torch.randn(2560, 2560, device="cuda") * 0.02).bfloat16()
    grad = torch.randn(1, tokens, 4 * 2560, device="cuda").bfloat16()

    def reference_inject(hyper_input, out, inj):
        return hyper_input + (out.unsqueeze(-2) * inj.unsqueeze(-1)).flatten(-2)

    results = {}
    for name, mix, inject in (("hf", lambda m, h: m(h), reference_inject), ("fused", qwen38._mix_streams, qwen38._inject)):
        leaf = x.clone().requires_grad_(True)
        y, mixed, inj = _sublayer(mix, inject, module, leaf, out_weight)
        y.backward(grad)
        results[name] = (y.detach(), mixed.detach(), inj.detach(), leaf.grad)
    for what, fused, reference in zip(("y", "mixed", "inj", "dx"), results["fused"], results["hf"]):
        assert fused.shape == reference.shape, what
        assert _rel(fused, reference) < 1e-2, (what, _rel(fused, reference))


def test_trainable_hyper_connection_keeps_the_hugging_face_path():
    from arctic_platform.model.implementations.qwen38 import hyper_connection

    module = _module().requires_grad_(True)
    x = torch.randn(1, 3, 4 * 2560, device="cuda").bfloat16()
    assert not hyper_connection.fused_hyper_connection_supported(module, x)
