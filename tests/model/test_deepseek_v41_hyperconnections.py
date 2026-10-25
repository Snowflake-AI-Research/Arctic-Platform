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
"""DeepSeek-V4.1's shifted mHC through the fused kernels against its reference functions, frozen and training."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused mHC runs on CUDA")


def _rel(a, b):
    a, b = a.double().cpu(), b.double().cpu()
    return ((a - b).norm() / b.norm().clamp_min(1e-30)).item()


def _connections(hidden: int, trainable: bool, device: str):
    from arctic_platform.model.implementations.deepseek_v41 import DeepseekV41Config
    from arctic_platform.model.implementations.deepseek_v41.hyperconnections import DeepseekV41HyperConnection

    torch.manual_seed(0)
    config = DeepseekV41Config(hidden_size=hidden)
    connections = []
    for _ in range(2):
        hc = DeepseekV41HyperConnection(config)
        with torch.no_grad():  # on the CPU generator, so every device gets the same weights
            hc.fn.normal_(0, 0.02)
            hc.base.normal_(0, 0.5)
            hc.scale.uniform_(0.5, 1.5)
        connections.append(hc.to(device).requires_grad_(trainable))
    return connections


def _layer(connections, streams, weights):
    """A decoder layer's two sites (stand-in sublayers) and the final collapse, in the shifted order."""
    from arctic_platform.model.implementations.deepseek_v41.hyperconnections import collapse
    from arctic_platform.model.implementations.deepseek_v41.hyperconnections import identity_pre
    from arctic_platform.model.implementations.deepseek_v41.hyperconnections import write_back

    pre = identity_pre(streams)
    for hc, weight in zip(connections, weights):
        next_pre, post, comb = hc.mixes(streams)
        out = torch.tanh(collapse(streams, pre) @ weight)
        streams = write_back(streams, out, post, comb)
        pre = next_pre
    return streams, collapse(streams, pre)


@pytest.mark.parametrize("trainable", [False, True])
@pytest.mark.parametrize("tokens", [1, 513])
def test_fused_mhc_matches_the_reference(tokens, trainable, monkeypatch):
    from arctic_platform.model.implementations.deepseek_v41 import hyperconnections

    hidden = 1024
    torch.manual_seed(1)
    streams = torch.randn(1, tokens, 4, hidden, device="cuda").bfloat16()
    weights = [(torch.randn(hidden, hidden, device="cuda") * 0.03).bfloat16() for _ in range(2)]
    grads = (torch.randn(1, tokens, 4, hidden, device="cuda"), torch.randn(1, tokens, hidden, device="cuda"))

    # "exact" is the reference in FP32 throughout (on CPU); the fused path has to be within 1e-2 of it, or no worse
    # than 1.5x the reference run as training ran it (BF16 streams on CUDA).
    names = ["streams", "collapsed", "dstreams"] + (["dfn", "dbase", "dscale"] if trainable else [])
    results = {}
    for name, fused, device, dtype in (
        ("bf16", False, "cuda", torch.bfloat16),
        ("fused", True, "cuda", torch.bfloat16),
        ("exact", False, "cpu", torch.float32),
    ):
        with monkeypatch.context() as patch:
            if not fused:
                patch.setattr(hyperconnections, "_fused", lambda mhc_states: False)
            connections = _connections(hidden, trainable, device)
            leaf = streams.detach().to(device, dtype).clone().requires_grad_(True)
            out, collapsed = _layer(connections, leaf, [w.to(device, dtype) for w in weights])
            torch.autograd.backward((out, collapsed), [g.to(device, dtype) for g in grads])
        params = [p.grad for hc in connections for p in (hc.fn, hc.base, hc.scale)] if trainable else []
        results[name] = [out.detach(), collapsed.detach(), leaf.grad, *params[0::3], *params[1::3], *params[2::3]]
    expanded = []
    for name in names:
        expanded += [name] if name in ("streams", "collapsed", "dstreams") else [f"{name}{i}" for i in range(2)]
    for what, fused, bf16, exact in zip(expanded, results["fused"], results["bf16"], results["exact"]):
        assert fused.shape == exact.shape, what
        bound = max(1e-2, 1.5 * _rel(bf16, exact))
        assert _rel(fused, exact) < bound, (what, _rel(fused, exact), bound)
