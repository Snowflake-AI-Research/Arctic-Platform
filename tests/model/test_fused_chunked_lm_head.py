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

"""The fused CUDA chunked LM head against ``log_softmax`` in float64 over the head's own BF16 logits."""

from __future__ import annotations

import pytest
import torch

from arctic_platform.model.implementations.gpu.lm_head import chunked_lm_head_logprobs
from arctic_platform.model.implementations.gpu.packing import IGNORE_INDEX

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused LM head runs on CUDA")

VOCAB = 151936  # Qwen3's: a multiple of 64 that is not a multiple of the kernel's 8192-entry block
LONG_VOCAB = 248320  # Qwen3.5 / 3.6 / 3.8's: the K of the hidden gradient's GEMMs


def _inputs(tokens, hidden_size, *, seed=0, scale=1.0, vocab=VOCAB):
    torch.manual_seed(seed)
    hidden = (torch.randn(1, tokens, hidden_size, device="cuda") * scale).bfloat16()
    weight = (torch.randn(vocab, hidden_size, device="cuda") / hidden_size**0.5).bfloat16()
    labels = torch.randint(0, vocab, (1, tokens), device="cuda")
    labels[0, ::7] = IGNORE_INDEX
    temperature = torch.rand(1, tokens, device="cuda") + 0.5
    return hidden, weight, labels, temperature


def _reference(hidden, weight, labels, temperature, grad):
    """float64 log-probabilities of the BF16 logits, and float64 gradients of hidden and weight through them."""
    logits = hidden[0] @ weight.t()  # [N, V] BF16, the head's own projection
    logits64 = logits.double().requires_grad_(True)
    safe = torch.where(labels[0] == IGNORE_INDEX, 0, labels[0])
    scaled = logits64 / temperature[0].double().unsqueeze(-1)
    logprobs = torch.log_softmax(scaled, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    logprobs.backward(grad.double())
    dlogits = logits64.grad
    grad_hidden = dlogits @ weight.double()
    grad_weight = dlogits.t() @ hidden[0].double()
    return logprobs.detach(), grad_hidden, grad_weight


def _run(hidden, weight, labels, temperature, grad, *, token_chunk_size):
    hidden = hidden.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(True)
    logprobs = chunked_lm_head_logprobs(
        hidden,
        weight,
        labels,
        temperature=temperature,
        token_chunk_size=token_chunk_size,
        vocab_chunk_size=8192,
        fp32_lm_head=False,
    )
    logprobs.backward(grad.reshape(logprobs.shape))
    return logprobs.detach()[0], hidden.grad[0], weight.grad


def _assert_bf16_close(actual, expected, name):
    # A BF16 result rounds once (2**-9 relative); allow a few roundings of the largest entry.
    scale = expected.abs().max().item()
    torch.testing.assert_close(actual.double(), expected, rtol=2e-2, atol=4e-3 * scale, msg=name)


def test_fused_head_matches_float64_log_softmax():
    tokens, token_chunk_size = 1000, 256
    hidden, weight, labels, temperature = _inputs(tokens, 256)
    grad = torch.randn(tokens, device="cuda")
    grad[labels[0] == IGNORE_INDEX] = 0.0

    logprobs, grad_hidden, grad_weight = _run(
        hidden, weight, labels, temperature, grad, token_chunk_size=token_chunk_size
    )
    ref_logprobs, ref_grad_hidden, ref_grad_weight = _reference(hidden, weight, labels, temperature, grad)

    assert logprobs.dtype == torch.float32
    assert grad_hidden.dtype == torch.bfloat16 and grad_weight.dtype == torch.bfloat16
    valid = labels[0] != IGNORE_INDEX
    torch.testing.assert_close(logprobs[valid].double(), ref_logprobs[valid], rtol=1e-5, atol=1e-5)
    _assert_bf16_close(grad_hidden, ref_grad_hidden, "grad_hidden")
    _assert_bf16_close(grad_weight, ref_grad_weight, "grad_weight")


def test_fused_head_hidden_gradient_keeps_its_norm_over_a_long_vocabulary():
    """The hidden gradient's GEMMs run over the vocabulary (K = 248,320 here). As one BF16 GEMM per token chunk it
    came out 2.5e-4 smaller in norm than float64 on H200, which shrinks every gradient below the head; summed over
    8192-wide vocabulary tiles in FP32 it stays within 1e-5. A mean-NLL gradient, and 2048-token chunks (GLM-5.3-Flash's
    lm_head.token_chunk_size), so that cuBLAS does not split K."""
    tokens, hidden_size = 2048, 2048
    hidden, weight, labels, _ = _inputs(tokens, hidden_size, scale=2.0, vocab=LONG_VOCAB)
    grad = torch.full((tokens,), -1.0 / tokens, device="cuda")
    grad[labels[0] == IGNORE_INDEX] = 0.0

    actual = hidden.clone().requires_grad_(True)
    logprobs = chunked_lm_head_logprobs(
        actual, weight, labels, token_chunk_size=2048, vocab_chunk_size=8192, fp32_lm_head=False
    )
    logprobs.backward(grad.reshape(logprobs.shape))

    # float64 through the head's own BF16 logits, 256 rows at a time.
    weight64 = weight.double()
    expected = torch.empty(tokens, hidden_size, device="cuda", dtype=torch.float64)
    for start in range(0, tokens, 256):
        rows = slice(start, start + 256)
        logits = (hidden[0, rows] @ weight.t()).double().requires_grad_(True)
        safe = torch.where(labels[0, rows] == IGNORE_INDEX, 0, labels[0, rows])
        torch.log_softmax(logits, dim=-1).gather(-1, safe[:, None])[:, 0].backward(grad[rows].double())
        expected[rows] = logits.grad @ weight64

    norm_ratio = (actual.grad[0].double().norm() / expected.norm()).item()
    assert abs(norm_ratio - 1.0) < 5e-5, f"grad_hidden norm is {norm_ratio:.6f} of float64's"
    _assert_bf16_close(actual.grad[0], expected, "grad_hidden")
