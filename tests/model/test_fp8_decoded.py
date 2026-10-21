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
"""Block-FP8 GEMMs that decode the frozen weight to BF16 on CUDA: the input gradient, and the expert forward."""

from __future__ import annotations

import pytest
import torch

from arctic_platform.model.implementations import fp8

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the decoded FP8 paths run on CUDA")


def _quantized(*shape: int) -> tuple[torch.Tensor, torch.Tensor]:
    if len(shape) == 2:
        return fp8.quantize_to_fp8_blockwise(torch.randn(*shape, device="cuda") * 0.02, 128)
    pairs = [_quantized(*shape[1:]) for _ in range(shape[0])]
    return torch.stack([p[0] for p in pairs]), torch.stack([p[1] for p in pairs])


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("shape", [(576, 6144), (1000, 300), (3, 512, 768)])
def test_decode_matches_reference_including_partial_blocks(shape):
    from arctic_platform.model.implementations.kernels.fp8_block_dequant import dequantize_fp8_blocks

    torch.manual_seed(0)
    weight, scale = _quantized(*shape)
    expected = fp8.dequantize_from_fp8_blockwise(weight, scale, dtype=torch.float32)
    # Only the final BF16 rounding differs.
    torch.testing.assert_close(dequantize_fp8_blocks(weight, scale).float(), expected, rtol=8e-3, atol=1e-6)


def test_fp8_linear_input_gradient_decodes_to_bf16():
    torch.manual_seed(0)
    weight, scale = _quantized(1536, 4096)
    x = torch.randn(700, 4096, device="cuda").bfloat16().requires_grad_(True)
    out = fp8.fp8_linear(x, weight, scale, 128)
    grad = torch.randn_like(out)
    out.backward(grad)
    expected = fp8._fp8_linear_bwd_input(grad.float(), weight, scale, 128)
    assert _rel(x.grad, expected) < 1e-2


def test_grouped_fp8_mm_input_gradient_decodes_and_zeroes_padding_rows():
    torch.manual_seed(0)
    weight, scale = _quantized(4, 512, 768)
    counts = torch.tensor([100, 0, 37, 255], device="cuda")
    routed = int(counts.sum())
    x = torch.randn(routed + 9, 768, device="cuda").bfloat16().requires_grad_(True)
    out = fp8.grouped_fp8_mm(x, weight, scale, counts, 128)
    grad = torch.randn_like(out)
    out.backward(grad)
    grads = torch.split(grad[:routed].float(), counts.tolist())
    expected = torch.cat([fp8._fp8_linear_bwd_input(g, weight[i], scale[i], 128) for i, g in enumerate(grads)])
    assert _rel(x.grad[:routed], expected) < 1e-2
    assert (x.grad[routed:] == 0).all()


def test_grouped_fp8_decoded_mm_matches_decoded_weights_and_zeroes_padding_rows():
    torch.manual_seed(0)
    weight, scale = _quantized(4, 512, 768)
    counts = torch.tensor([100, 0, 37, 255], device="cuda")
    offsets = torch.cumsum(counts, 0, dtype=torch.int32)
    routed = int(counts.sum())
    x = torch.randn(routed + 9, 768, device="cuda").bfloat16().requires_grad_(True)
    out = fp8.grouped_fp8_decoded_mm(x, weight, scale, offsets, routed, 128)
    grad = torch.randn_like(out)
    out.backward(grad)

    decoded = fp8.dequantize_from_fp8_blockwise(weight, scale, dtype=torch.float32)
    tokens = torch.split(x[:routed].float(), counts.tolist())
    grads = torch.split(grad[:routed].float(), counts.tolist())
    expected = torch.cat([t @ decoded[i].T for i, t in enumerate(tokens)])
    expected_dx = torch.cat([g @ decoded[i] for i, g in enumerate(grads)])
    assert type(out.grad_fn).__name__ == "_GroupedDecodedFp8MmFnBackward"
    assert _rel(out[:routed], expected) < 1e-2
    assert _rel(x.grad[:routed], expected_dx) < 1e-2
    assert (out[routed:] == 0).all() and (x.grad[routed:] == 0).all()


def test_grouped_fp8_decoded_mm_stacks_side_by_side():
    torch.manual_seed(0)
    w1, s1 = _quantized(4, 256, 768)
    w3, s3 = _quantized(4, 384, 768)
    counts = torch.tensor([100, 0, 37, 255], device="cuda")
    offsets = torch.cumsum(counts, 0, dtype=torch.int32)
    routed = int(counts.sum())
    x = torch.randn(routed + 9, 768, device="cuda").bfloat16().requires_grad_(True)
    out = fp8.grouped_fp8_decoded_mm(x, (w1, w3), (s1, s3), offsets, routed, 128)
    grad = torch.randn_like(out)
    out.backward(grad)
    stacked_dx = x.grad.clone()

    x.grad = None
    gate = fp8.grouped_fp8_decoded_mm(x, w1, s1, offsets, routed, 128)
    up = fp8.grouped_fp8_decoded_mm(x, w3, s3, offsets, routed, 128)
    (gate * grad[:, :256]).sum().add((up * grad[:, 256:]).sum()).backward()

    assert out.shape == (routed + 9, 640)
    assert _rel(out[:routed], torch.cat([gate, up], 1)[:routed]) < 1e-3
    assert _rel(stacked_dx[:routed], x.grad[:routed]) < 1e-2
    assert (out[routed:] == 0).all() and (stacked_dx[routed:] == 0).all()
