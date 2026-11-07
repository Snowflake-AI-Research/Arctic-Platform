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
"""The head-512 TileLang attention matches fp32 causal attention, forward and backward, within bf16 error."""

import pytest
import torch

from arctic_platform.model.implementations.kernels import causal_attention_hd512 as hd512


def _inputs(heads, kv_heads, seq_len, device):
    generator = torch.Generator(device="cpu").manual_seed(seq_len + heads)
    query = (torch.randn(1, heads, seq_len, 512, generator=generator) * 0.5).to(device, torch.bfloat16)
    key = (torch.randn(1, kv_heads, seq_len, 512, generator=generator) * 0.5).to(device, torch.bfloat16)
    value = torch.randn(1, kv_heads, seq_len, 512, generator=generator).to(device, torch.bfloat16)
    grad = torch.randn(1, seq_len, heads, 512, generator=generator).to(device, torch.bfloat16)
    return query, key, value, grad


def test_unsupported_inputs_are_named():
    query, key, value, _ = _inputs(4, 2, 64, "cpu")
    assert "CUDA" in hd512.unsupported_reason(query, key, value)
    if not torch.cuda.is_available():
        return
    query, key, value = (tensor.cuda() for tensor in (query, key, value))
    assert hd512.unsupported_reason(query, key, value) is None
    assert "bfloat16" in hd512.unsupported_reason(query.float(), key.float(), value.float())
    assert "multiple of the 64-token tile" in hd512.unsupported_reason(
        query[:, :, :48], key[:, :, :48], value[:, :, :48]
    )
    assert "head size 512" in hd512.unsupported_reason(query[..., :256], key[..., :256], value[..., :256])
    assert "do not group" in hd512.unsupported_reason(query[:, :3], key, value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the TileLang kernels need CUDA")
@pytest.mark.parametrize(("heads", "kv_heads", "seq_len"), [(8, 2, 1024), (4, 1, 576), (2, 2, 64)])
def test_matches_fp32_attention_and_gradients(heads, kv_heads, seq_len):
    query, key, value, grad = _inputs(heads, kv_heads, seq_len, "cuda")
    leaves = [tensor.clone().requires_grad_(True) for tensor in (query, key, value)]
    output = hd512.causal_gqa_attention_hd512(*leaves, 1.0)
    output.backward(grad)

    references = [tensor.float().requires_grad_(True) for tensor in (query, key, value)]
    expected = torch.nn.functional.scaled_dot_product_attention(
        *references, is_causal=True, scale=1.0, enable_gqa=True
    ).transpose(1, 2)
    expected.backward(grad.float())

    def relative_error(mine, reference):
        return (torch.max(torch.abs(mine.float() - reference)) / torch.max(torch.abs(reference))).item()

    assert output.shape == (1, seq_len, heads, 512)
    assert relative_error(output.detach(), expected.detach()) < 1e-2
    for name, leaf, reference in zip("qkv", leaves, references):
        assert relative_error(leaf.grad, reference.grad) < 2e-2, name


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the TileLang kernels need CUDA")
def test_softmax_scale_is_applied():
    query, key, value, _ = _inputs(4, 2, 128, "cuda")
    output = hd512.causal_gqa_attention_hd512(query, key, value, 512**-0.5)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.float(), key.float(), value.float(), is_causal=True, scale=512**-0.5, enable_gqa=True
    ).transpose(1, 2)
    assert torch.max(torch.abs(output.float() - expected)) / torch.max(torch.abs(expected)) < 1e-2


def _segmented_reference(query, key, value, cu_seqlens, scale):
    """fp32 causal attention of each packed row on its own, concatenated: [B, S, H, D]."""
    pieces = []
    bounds = cu_seqlens.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        pieces.append(
            torch.nn.functional.scaled_dot_product_attention(
                query[:, :, start:end],
                key[:, :, start:end],
                value[:, :, start:end],
                is_causal=True,
                scale=scale,
                enable_gqa=True,
            )
        )
    return torch.cat(pieces, dim=2).transpose(1, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the TileLang kernels need CUDA")
@pytest.mark.parametrize("bounds", [[0, 300, 700, 1024], [0, 64, 128, 1024], [0, 1, 1023, 1024], [0, 1024]])
def test_packed_rows_attend_only_within_their_row(bounds):
    query, key, value, grad = _inputs(8, 2, 1024, "cuda")
    cu_seqlens = torch.tensor(bounds, device="cuda", dtype=torch.int32)
    leaves = [tensor.clone().requires_grad_(True) for tensor in (query, key, value)]
    output = hd512.causal_gqa_attention_hd512(*leaves, 1.0, cu_seqlens=cu_seqlens)
    output.backward(grad)

    references = [tensor.float().requires_grad_(True) for tensor in (query, key, value)]
    expected = _segmented_reference(*references, cu_seqlens, 1.0)
    expected.backward(grad.float())

    def relative_error(mine, reference):
        return (torch.max(torch.abs(mine.float() - reference)) / torch.max(torch.abs(reference))).item()

    assert torch.isfinite(output).all()
    assert relative_error(output.detach(), expected.detach()) < 1e-2
    for name, leaf, reference in zip("qkv", leaves, references):
        assert relative_error(leaf.grad, reference.grad) < 2e-2, name


def test_segment_starts():
    starts = hd512.segment_starts(torch.tensor([0, 2, 5]), 5, batch=2)
    assert starts.tolist() == [[0, 0, 2, 2, 2]] * 2
    with pytest.raises(ValueError, match="from 0 to the sequence length 6"):
        hd512.segment_starts(torch.tensor([0, 2, 5]), 6)
