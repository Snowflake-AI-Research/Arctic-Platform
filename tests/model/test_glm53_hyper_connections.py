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
"""The fused GLM-5.3-Flash mHC site against Hugging Face's ``Glm5NextTextHyperConnection``, frozen and training."""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers.models.glm5_next")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused mHC runs on CUDA")


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30)).item()


def _module(hidden: int, trainable: bool):
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection

    torch.manual_seed(0)
    module = Glm5NextTextHyperConnection(Glm5NextTextConfig(hidden_size=hidden)).cuda()
    with torch.no_grad():
        module.fn.normal_(0, 0.02)
        module.base.normal_(0, 0.5)
        module.scale.uniform_(0.5, 1.5)
    return module.requires_grad_(trainable)


def _site(module, streams, sublayer_weight, fused: bool):
    """One decoder site: collapse the streams, run a stand-in sublayer, remix."""
    from arctic_platform.model.implementations.glm53 import hyper_connections as hc
    from arctic_platform.model.implementations.glm53.modeling_glm5_next import _remix

    post, comb, collapsed = hc.hyper_connection(module, streams) if fused else module(streams)
    out = torch.tanh(collapsed @ sublayer_weight)
    return _remix(fused, False, post, comb, out, streams)


@pytest.mark.parametrize("trainable", [False, True])
@pytest.mark.parametrize("tokens", [1, 513])
def test_fused_mhc_matches_hugging_face(tokens, trainable):
    from arctic_platform.model.implementations.glm53 import hyper_connections as hc

    hidden = 1024
    torch.manual_seed(1)
    streams = torch.randn(1, tokens, 4, hidden, device="cuda").bfloat16()
    sublayer_weight = (torch.randn(hidden, hidden, device="cuda") * 0.03).bfloat16()
    grad = torch.randn(1, tokens, 4, hidden, device="cuda").bfloat16()

    # Accuracy is judged against the reference in FP64: the fused path has to be within 1e-2 of it, or no worse
    # than 1.5x the reference's own BF16/FP32 path (which reduces scale's gradient over T x 24 terms to 1.4e-2 at
    # 4K tokens and 1.9e-2 at one token).
    names = ["y", "dstreams"] + (["dfn", "dbase", "dscale"] if trainable else [])
    results = {}
    for name, fused, dtype in (
        ("fp32", False, streams.dtype),
        ("fused", True, streams.dtype),
        ("fp64", False, torch.float64),
    ):
        module = _module(hidden, trainable)
        assert not fused or hc.fused_hyper_connection_applies(module, streams)
        module = module.to(dtype) if dtype == torch.float64 else module
        leaf = streams.clone().to(dtype).requires_grad_(True)
        y = _site(module, leaf, sublayer_weight.to(dtype), fused)
        y.backward(grad.to(dtype))
        params = [module.fn.grad, module.base.grad, module.scale.grad] if trainable else []
        results[name] = [y.detach(), leaf.grad, *params]
    for what, fused, fp32, exact in zip(names, results["fused"], results["fp32"], results["fp64"]):
        assert fused.shape == exact.shape, what
        bound = max(1e-2, 1.5 * _rel(fp32, exact))
        assert _rel(fused, exact) < bound, (what, _rel(fused, exact), bound)
