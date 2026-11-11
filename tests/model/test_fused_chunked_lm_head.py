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

"""The fused CUDA chunked LM head computes the tiled composition's log-probabilities and gradients.

The reference is ``log_softmax`` in float64 over the head's own BF16 logits (one cuBLAS GEMM, as the fused head
computes them), so the comparison isolates the fused per-row math: FP32 against float64. The gradients are compared
with a BF16 tolerance against float64 autograd through those logits, and against the tiled composition.
"""

from __future__ import annotations

import pytest
import torch

from arctic_platform.model.implementations.gpu import lm_head as lm_head_mod
from arctic_platform.model.implementations.gpu.lm_head import chunked_lm_head_logprobs
from arctic_platform.model.implementations.gpu.packing import IGNORE_INDEX

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the fused LM head runs on CUDA")

VOCAB = 151936  # Qwen3's: a multiple of 64 that is not a multiple of the kernel's 8192-entry block


def _inputs(tokens, hidden_size, *, seed=0, scale=1.0):
    torch.manual_seed(seed)
    hidden = (torch.randn(1, tokens, hidden_size, device="cuda") * scale).bfloat16()
    weight = (torch.randn(VOCAB, hidden_size, device="cuda") / hidden_size**0.5).bfloat16()
    labels = torch.randint(0, VOCAB, (1, tokens), device="cuda")
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


def _run(hidden, weight, labels, temperature, grad, *, token_chunk_size, train_weight=True):
    hidden = hidden.clone().requires_grad_(True)
    weight = weight.clone().requires_grad_(train_weight)
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


@pytest.mark.parametrize("tokens,token_chunk_size", [(1000, 256), (777, 4096)], ids=["4-chunks", "1-chunk"])
def test_fused_head_matches_float64_log_softmax(tokens, token_chunk_size):
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


def test_fused_head_matches_the_tiled_composition(monkeypatch):
    tokens = 640
    hidden, weight, labels, temperature = _inputs(tokens, 128, seed=3, scale=3.0)
    grad = torch.randn(tokens, device="cuda")
    fused = _run(hidden, weight, labels, temperature, grad, token_chunk_size=200)
    monkeypatch.setattr(lm_head_mod, "_fused_head_applies", lambda *args, **kwargs: False)
    tiled = _run(hidden, weight, labels, temperature, grad, token_chunk_size=200)

    valid = labels[0] != IGNORE_INDEX
    # Both score the same BF16 logits; only the GEMM partition (one whole-vocabulary GEMM against 8192-entry tiles)
    # can round a logit differently.
    torch.testing.assert_close(fused[0][valid], tiled[0][valid], rtol=1e-3, atol=1e-3)
    for name, a, b in (("grad_hidden", fused[1], tiled[1]), ("grad_weight", fused[2], tiled[2])):
        _assert_bf16_close(a, b.double(), name)


def _near_certain_reference(hidden, weight, labels, grad):
    """float64 logprobs and hidden gradient in the others-minus-target form.

    ``log_softmax`` itself is not a usable reference here: its sum adds the target's exp(0) = 1 to ~152K terms of
    ~1e-16, so even float64 keeps only a few digits of 1 - softmax_t.
    """
    logits = (hidden[0] @ weight.t()).double()
    rows = torch.arange(logits.shape[0], device=logits.device)
    others = logits - logits[rows, labels[0]].unsqueeze(-1)
    others[rows, labels[0]] = float("-inf")
    rest = others.exp()
    s = rest.sum(-1)
    g = grad.double()
    dlogits = -g.unsqueeze(-1) * rest / (1 + s).unsqueeze(-1)
    dlogits[rows, labels[0]] = g * s / (1 + s)
    return -torch.log1p(s), dlogits @ weight.double()


def test_fused_head_near_certain_target_keeps_its_gradient():
    # Each token's hidden state is one-hot on its own column, and only its target's weight row is 1 there, so its
    # target logit is 37 and every other logit 37 * N(0, 0.01^2): the target leads the logsumexp of the other ~152K
    # logits (~12) by ~25 nats. 1 - softmax underflows in FP32 there; sigmoid(others - target) does not.
    tokens = hidden_size = 64
    torch.manual_seed(5)
    weight = (torch.randn(VOCAB, hidden_size, device="cuda") * 0.01).bfloat16()
    labels = torch.randperm(VOCAB, device="cuda")[:tokens].unsqueeze(0)
    weight[labels[0]] = 0.0
    weight[labels[0], torch.arange(tokens, device="cuda")] = 1.0
    hidden = (torch.eye(tokens, hidden_size, device="cuda") * 37.0).bfloat16().unsqueeze(0)
    grad = torch.ones(tokens, device="cuda")
    ones = torch.ones(1, tokens, device="cuda")
    logprobs, grad_hidden, _ = _run(hidden, weight, labels, ones, grad, token_chunk_size=32)
    ref_logprobs, ref_grad_hidden = _near_certain_reference(hidden, weight, labels, grad)
    assert (ref_logprobs > -1e-9).all() and (ref_logprobs < 0).all()
    torch.testing.assert_close(logprobs.double(), ref_logprobs, rtol=1e-4, atol=0.0)
    # The target column of each token's hidden gradient is 1 - softmax_t ~ exp(-25): nonzero, to BF16 rounding.
    diagonal = torch.arange(tokens, device="cuda")
    target_grad, ref_target_grad = grad_hidden[diagonal, diagonal].double(), ref_grad_hidden[diagonal, diagonal]
    assert (ref_target_grad > 0).all()
    torch.testing.assert_close(target_grad, ref_target_grad, rtol=2e-2, atol=0.0)
    _assert_bf16_close(grad_hidden, ref_grad_hidden, "grad_hidden")


def test_frozen_weight_gets_no_gradient_and_hidden_matches():
    tokens = 300
    hidden, weight, labels, temperature = _inputs(tokens, 128, seed=7)
    grad = torch.randn(tokens, device="cuda")
    logprobs, grad_hidden, grad_weight = _run(
        hidden, weight, labels, temperature, grad, token_chunk_size=128, train_weight=False
    )
    trained = _run(hidden, weight, labels, temperature, grad, token_chunk_size=128)
    assert grad_weight is None
    torch.testing.assert_close(logprobs, trained[0], rtol=0, atol=0)
    torch.testing.assert_close(grad_hidden, trained[1], rtol=0, atol=0)


@pytest.mark.parametrize("softcap", [None, 30.0], ids=["plain", "softcap"])
def test_cuda_bf16_calls_take_the_fused_head(monkeypatch, softcap):
    def tiled_not_expected():
        raise AssertionError("the tiled composition ran for a plain CUDA BF16 call")

    monkeypatch.setattr(lm_head_mod, "_get_sequence_chunked_logprob_fn", tiled_not_expected)
    hidden, weight, labels, temperature = _inputs(64, 64)
    chunked_lm_head_logprobs(
        hidden,
        weight,
        labels,
        temperature=temperature,
        token_chunk_size=16,
        vocab_chunk_size=8192,
        fp32_lm_head=False,
        softcap=softcap,
    )


def _softcap_reference(hidden, weight, labels, temperature, grad, softcap):
    """float64 log-probabilities of ``softcap * tanh(logits / softcap) / temperature`` over the head's BF16 logits."""
    logits = hidden[0] @ weight.t()
    logits64 = logits.double().requires_grad_(True)
    capped = torch.tanh(logits64 / softcap) * softcap
    safe = torch.where(labels[0] == IGNORE_INDEX, 0, labels[0])
    scaled = capped / temperature[0].double().unsqueeze(-1)
    logprobs = torch.log_softmax(scaled, dim=-1).gather(-1, safe.unsqueeze(-1)).squeeze(-1)
    logprobs.backward(grad.double())
    return logprobs.detach(), logits64.grad @ weight.double(), logits64.grad.t() @ hidden[0].double()


def _run_softcap(hidden, weight, labels, temperature, grad, *, token_chunk_size, softcap):
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
        softcap=softcap,
    )
    logprobs.backward(grad.reshape(logprobs.shape))
    return logprobs.detach()[0], hidden.grad[0], weight.grad


@pytest.mark.parametrize("scale", [1.0, 40.0], ids=["moderate", "saturated"])
def test_fused_head_softcap_matches_float64_and_the_tiled_composition(monkeypatch, scale):
    # Gemma's final_logit_softcapping = 30; at scale 40 the logits reach the hundreds and tanh saturates.
    softcap = 30.0
    tokens = 600
    hidden, weight, labels, temperature = _inputs(tokens, 256, seed=11, scale=scale)
    grad = torch.randn(tokens, device="cuda")
    grad[labels[0] == IGNORE_INDEX] = 0.0
    if scale > 1:
        assert (hidden[0] @ weight.t()).abs().max().item() > 3 * softcap

    fused = _run_softcap(hidden, weight, labels, temperature, grad, token_chunk_size=256, softcap=softcap)
    ref_logprobs, ref_grad_hidden, ref_grad_weight = _softcap_reference(
        hidden, weight, labels, temperature, grad, softcap
    )
    valid = labels[0] != IGNORE_INDEX
    torch.testing.assert_close(fused[0][valid].double(), ref_logprobs[valid], rtol=1e-5, atol=1e-5)
    _assert_bf16_close(fused[1], ref_grad_hidden, "grad_hidden")
    _assert_bf16_close(fused[2], ref_grad_weight, "grad_weight")

    monkeypatch.setattr(lm_head_mod, "_fused_head_applies", lambda *args, **kwargs: False)
    tiled = _run_softcap(hidden, weight, labels, temperature, grad, token_chunk_size=256, softcap=softcap)
    torch.testing.assert_close(fused[0][valid], tiled[0][valid], rtol=1e-3, atol=1e-3)
    for name, a, b in (("grad_hidden", fused[1], tiled[1]), ("grad_weight", fused[2], tiled[2])):
        _assert_bf16_close(a, b.double(), name)


def test_fused_head_softcap_on_gemmas_vocabulary(monkeypatch):
    # Gemma's 262,144-entry vocabulary (32 blocks of the kernel's 8192), with a token chunk that keeps the
    # whole-vocabulary logits buffer small.
    vocab, softcap, tokens = 262144, 30.0, 300
    torch.manual_seed(13)
    hidden = (torch.randn(1, tokens, 128, device="cuda") * 8).bfloat16()
    weight = (torch.randn(vocab, 128, device="cuda") / 128**0.5).bfloat16()
    labels = torch.randint(0, vocab, (1, tokens), device="cuda")
    temperature = torch.ones(1, tokens, device="cuda")
    grad = torch.randn(tokens, device="cuda")
    fused = _run_softcap(hidden, weight, labels, temperature, grad, token_chunk_size=128, softcap=softcap)
    ref_logprobs, ref_grad_hidden, ref_grad_weight = _softcap_reference(
        hidden, weight, labels, temperature, grad, softcap
    )
    torch.testing.assert_close(fused[0].double(), ref_logprobs, rtol=1e-5, atol=1e-5)
    _assert_bf16_close(fused[1], ref_grad_hidden, "grad_hidden")
    _assert_bf16_close(fused[2], ref_grad_weight, "grad_weight")
