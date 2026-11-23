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

"""The sampler side of fp32 hyper-connection residual streams (``ARCTIC_FP32_HC_RESIDUAL``).

vLLM's own kernels run a chain of hyper-connection boundaries (combine with the
pending block output, norm, gate mix, a bf16 stand-in block) on bf16 and on
fp32 streams, against the same chain in fp64. On CPU this needs
``TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES=``.
"""

from __future__ import annotations

import os

import pytest
import torch

hc_ops = pytest.importorskip("vllm.models.qwen4_exp.nvidia.ops.hc")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() and os.environ.get("TRITON_INTERPRET") != "1",
    reason="vLLM's hyper-connection kernels need CUDA or the Triton interpreter",
)

HC, HIDDEN, TOKENS, LAYERS, EPS = 4, 64, 32, 12, 1e-6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _weights():
    torch.manual_seed(0)
    layers = []
    for _ in range(LAYERS):
        layers.append(dict(
            norm=(0.3 * torch.randn(HC * HIDDEN)).bfloat16(),
            gate=(2 * torch.randn(TOKENS, HC * HIDDEN)).bfloat16(),
            inject=(0.1 * torch.randn(HC * HIDDEN, HC)).bfloat16(),
            block=(torch.randn(HIDDEN, HIDDEN) / HIDDEN**0.5).bfloat16(),
        ))
    return [{k: v.to(DEVICE) for k, v in layer.items()} for layer in layers]


def _chain(streams, layers, combine_norm):
    block = torch.zeros(TOKENS, HIDDEN, dtype=torch.bfloat16, device=DEVICE)
    logits = torch.zeros(TOKENS, HC, dtype=torch.bfloat16, device=DEVICE)
    for layer in layers:
        streams, normed = combine_norm(streams, block, logits, layer["norm"])
        assert normed.dtype == torch.bfloat16
        mixed = hc_ops.hc_gate_mix(normed, layer["gate"], HC)
        block = torch.tanh(mixed.float() @ layer["block"].float()).to(mixed.dtype)
        logits = (normed.float() @ layer["inject"].float()).to(normed.dtype)
    return streams


def _combine_norm_fp64(streams, block, logits, weight):
    weight_stream = 2 * torch.sigmoid(logits.double() / HC)
    streams = streams + (block.double().unsqueeze(1) * weight_stream.unsqueeze(-1)).flatten(1)
    grouped = streams.view(TOKENS, HC, HIDDEN)
    normed = grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + EPS)
    return streams, (normed.flatten(1) * (1 + weight.double()))


def test_fp32_streams_feed_bf16_blocks_and_are_closer_to_fp64():
    from arctic_platform.inference.vllm import fp32_hc_residual

    def sampler(streams, block, logits, weight):
        return fp32_hc_residual._hc_combine_norm(streams, block, logits, weight, EPS, HC)

    layers = _weights()
    x = (2 * torch.randn(TOKENS, HC * HIDDEN, device=DEVICE)).bfloat16()

    def exact_chain(streams):
        block = torch.zeros(TOKENS, HIDDEN, dtype=torch.float64, device=DEVICE)
        logits = torch.zeros(TOKENS, HC, dtype=torch.float64, device=DEVICE)
        for layer in layers:
            streams, normed = _combine_norm_fp64(streams, block, logits, layer["norm"])
            gate = torch.sigmoid(layer["gate"].double()).view(TOKENS, HC, HIDDEN)
            mixed = (gate * normed.view(TOKENS, HC, HIDDEN)).mean(1)
            block = torch.tanh(mixed @ layer["block"].double())
            logits = normed @ layer["inject"].double()
        return streams

    reference = exact_chain(x.double())
    errors = {}
    for dtype in (torch.bfloat16, torch.float32):
        streams = _chain(x.to(dtype), layers, sampler)
        assert streams.dtype == dtype
        errors[dtype] = ((streams.double() - reference).norm() / reference.norm()).item()
    assert errors[torch.float32] < 0.8 * errors[torch.bfloat16], errors



class _KVProj:
    def __init__(self, kv, output_sizes):
        self.kv, self.output_sizes = kv, output_sizes

    def __call__(self, _embeddings):
        return self.kv, None


def _ple_layer(kv, conv_weights, conv_state):
    """The attributes ``Qwen4ExpPLELayer.forward`` reads, with vLLM's real gate and short-convolution kernels; the
    n-gram lookup and projection return fixed values, and the convolution runs in decode mode, token t in slot t + 1."""
    from types import SimpleNamespace

    from vllm.models.qwen4_exp.common.ops.ple import ple_conv

    channels = HC * HIDDEN

    def short_conv(inputs, residual, outer_residual):
        slots = torch.arange(1, inputs.shape[0] + 1, device=DEVICE, dtype=torch.int32)  # slot 0 is the null slot
        ple_conv(inputs, residual, conv_state, conv_weights, slots, outer_residual, mode="decode", dilation=3)

    torch.manual_seed(3)
    norms = [SimpleNamespace(weight=(0.3 * torch.randn(channels, device=DEVICE)).bfloat16(), eps=EPS) for _ in range(3)]
    return SimpleNamespace(
        model_config=SimpleNamespace(dtype=torch.bfloat16),
        use_sequence_parallel=False,
        ple_embedding=lambda hidden_states, *_: hidden_states,
        _dequantize_embeddings=lambda embeddings, _dtype: embeddings,
        kv_proj=_KVProj(kv, [channels, HIDDEN]),
        norm_key=norms[0],
        norm_query=norms[1],
        norm_conv=norms[2],
        _short_conv=short_conv,
    )



def test_ple_adds_gated_value_and_convolution_to_the_streams():
    """The patched PLE forward adds vLLM's gated value plus its short convolution (decode mode: three taps from the
    state, one from the token) to the streams, in fp32 for fp32 streams. Tolerances cover bf16 rounding only."""
    from vllm.models.qwen4_exp.common.ops.ple import ple_gate

    from arctic_platform.inference.vllm import fp32_hc_residual

    torch.manual_seed(2)
    channels = HC * HIDDEN
    kv = torch.randn(TOKENS, channels + HIDDEN, device=DEVICE).bfloat16()
    conv_weights = (0.5 * torch.randn(channels, 4, device=DEVICE)).bfloat16()
    conv_state = torch.randn(TOKENS + 1, channels, 9, device=DEVICE).bfloat16()
    streams = 2 * torch.randn(TOKENS, channels, device=DEVICE)
    ids = torch.zeros(TOKENS, dtype=torch.long, device=DEVICE)

    layer = _ple_layer(kv, conv_weights, conv_state.clone())
    key, value = kv.split([channels, HIDDEN], dim=-1)
    gated, conv_input = ple_gate(
        key, value, streams.bfloat16(), layer.norm_key.weight, layer.norm_query.weight, layer.norm_conv.weight, EPS
    )
    state = conv_state[1:]
    taps = torch.stack([state[..., 0], state[..., 3], state[..., 6], conv_input], dim=-1).float()
    conv = (taps * conv_weights.float()).sum(-1).bfloat16().float()
    ple_output = gated.float() + (conv * torch.sigmoid(conv)).bfloat16().float()

    patched = fp32_hc_residual.Qwen4ExpPLELayerPatch.forward
    fp32 = patched(_ple_layer(kv, conv_weights, conv_state.clone()), streams, ids, None, None)
    bf16 = patched(_ple_layer(kv, conv_weights, conv_state.clone()), streams.bfloat16(), ids, None, None)
    assert fp32.dtype == torch.float32 and bf16.dtype == torch.bfloat16
    torch.testing.assert_close(fp32 - streams, ple_output, rtol=0.02, atol=0.02)
    torch.testing.assert_close(bf16.float(), streams.bfloat16().float() + ple_output, rtol=0.02, atol=0.05)
