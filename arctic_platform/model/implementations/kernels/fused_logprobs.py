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
"""Per-token target log-probabilities of a ``[rows, vocab]`` logits tile, and their logit gradient, in one pass each.

The chunked LM head's composition streams every ``token x vocab`` tile through about ten FP32 elementwise kernels
(cast, scale, max, two ``where``, ``exp``, sum, a scatter and a ``nonzero`` that synchronizes with the host) forward
and as many backward. Here the projection stays one cuBLAS GEMM over the whole vocabulary per token chunk, and each
row of its BF16 logits is read once forward (an online log-sum-exp) and read and overwritten once backward (the logit
gradient, written in place so the two gradient GEMMs read it directly).

The math is the composition's, in FP32 per element: the target's log-probability is
``-softplus(logsumexp_{j != t}(x_j) - x_t)`` with the difference formed as ``(m - x_t) + log(s)`` (``m`` the row's
max, ``s`` the sum of ``exp(x_j - m)`` over the non-target logits), so a near-certain target keeps its precision; the
backward rebuilds ``softmax`` from the cached ``logz`` and takes the target's ``1 - softmax`` as
``sigmoid(logsumexp_{j != t} - x_t)``. ``x`` is the BF16 logit times the row's inverse temperature, after Gemma's
``softcap * tanh(logit / softcap)`` when a softcap is given (the composition's order: the cap, then the temperature);
backward then multiplies the logit gradient by ``1 - tanh^2`` after the temperature's chain rule.
"""

import triton
import triton.language as tl
from triton.language.extra import libdevice

# One row per program; a 151,936-entry vocabulary is 19 blocks of 8,192.
_BLOCK_V = 8192
_NUM_WARPS = 16


@triton.jit
def _log1p(e):
    # log(1 + e) for e >= 0 to FP32 precision (Goldberg): log(u) * e / (u - 1) with u = fl(1 + e), and e itself where
    # 1 + e rounds to 1.
    u = 1.0 + e
    return tl.where(u == 1.0, e, tl.log(u) * (e / (u - 1.0)))


@triton.jit
def _softplus(x):
    # torch.nn.functional.softplus (beta 1, threshold 20).
    return tl.where(x > 20.0, x, _log1p(tl.exp(tl.minimum(x, 20.0))))


@triton.jit
def _capped(logit, softcap, HAS_SOFTCAP: tl.constexpr):
    # Gemma's cap as the composition writes it: tanh(logit / softcap) * softcap.
    if HAS_SOFTCAP:
        return libdevice.tanh(logit / softcap) * softcap
    return logit


@triton.jit
def _logprob_fwd_kernel(
    logits_ptr,
    stride_row,
    labels_ptr,
    inv_t_ptr,
    logprob_ptr,
    logz_ptr,
    omt_ptr,
    vocab,
    softcap,
    BLOCK_V: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride_row
    label = tl.load(labels_ptr + row)
    inv_t = tl.load(inv_t_ptr + row)
    x_t = _capped(tl.load(base + label).to(tl.float32), softcap, HAS_SOFTCAP) * inv_t
    m = tl.full((), float("-inf"), tl.float32)
    s = tl.zeros((), tl.float32)
    for start in range(0, vocab, BLOCK_V):
        cols = start + tl.arange(0, BLOCK_V)
        valid = cols < vocab
        x = tl.load(base + cols, mask=valid, other=0.0).to(tl.float32)
        x = tl.where(valid, _capped(x, softcap, HAS_SOFTCAP) * inv_t, float("-inf"))
        m_new = tl.maximum(m, tl.max(x, axis=0))
        # The target's exp(0) = 1 would absorb the rest of the sum in FP32; it is added back through logz.
        e = tl.where(valid & (cols != label), tl.exp(x - m_new), 0.0)
        s = s * tl.exp(m - m_new) + tl.sum(e, axis=0)
        m = m_new
    log_s = tl.log(s)
    omt = (m - x_t) + log_s
    a = m + log_s
    hi = tl.maximum(a, x_t)
    logz = hi + _log1p(tl.exp(tl.minimum(a, x_t) - hi))
    tl.store(logprob_ptr + row, -_softplus(omt))
    tl.store(logz_ptr + row, logz)
    tl.store(omt_ptr + row, omt)


@triton.jit
def _logprob_bwd_kernel(
    logits_ptr,
    stride_row,
    labels_ptr,
    inv_t_ptr,
    grad_ptr,
    logz_ptr,
    omt_ptr,
    vocab,
    softcap,
    BLOCK_V: tl.constexpr,
    HAS_SOFTCAP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    base = logits_ptr + row * stride_row
    label = tl.load(labels_ptr + row)
    inv_t = tl.load(inv_t_ptr + row)
    g = tl.load(grad_ptr + row)
    logz = tl.load(logz_ptr + row)
    omt = tl.load(omt_ptr + row)
    # d logprob / d x_t = 1 - softmax_t = sigmoid(logsumexp_{j != t} - x_t), which stays nonzero where
    # 1 - exp(x_t - logz) rounds to 0.
    target_grad = g * tl.sigmoid(omt) * inv_t
    for start in range(0, vocab, BLOCK_V):
        cols = start + tl.arange(0, BLOCK_V)
        valid = cols < vocab
        logit = tl.load(base + cols, mask=valid, other=0.0).to(tl.float32)
        if HAS_SOFTCAP:
            squashed = libdevice.tanh(logit / softcap)
            x = squashed * softcap * inv_t
        else:
            x = logit * inv_t
        d = (-g * tl.exp(x - logz)) * inv_t
        d = tl.where(cols == label, target_grad, d)
        if HAS_SOFTCAP:
            d = d * (1 - squashed * squashed)
        tl.store(base + cols, d.to(logits_ptr.dtype.element_ty), mask=valid)


def target_logprobs_forward(logits, labels, inv_temperature, logprobs, logz, others_minus_target, softcap=None):
    """Fill ``logprobs``, ``logz`` and ``others_minus_target`` (FP32 ``[rows]``) from ``logits`` ``[rows, vocab]``."""
    rows, vocab = logits.shape
    if rows == 0:
        return
    if logits.stride(1) != 1:
        raise ValueError("logits must be contiguous along the vocabulary")
    _logprob_fwd_kernel[(rows,)](
        logits,
        logits.stride(0),
        labels,
        inv_temperature,
        logprobs,
        logz,
        others_minus_target,
        vocab,
        float(softcap) if softcap else 1.0,
        BLOCK_V=_BLOCK_V,
        HAS_SOFTCAP=bool(softcap),
        num_warps=_NUM_WARPS,
    )


def target_logprobs_backward_(logits, labels, inv_temperature, grad_logprobs, logz, others_minus_target, softcap=None):
    """Overwrite ``logits`` with ``d sum(grad_logprobs * logprobs) / d logits`` in its own dtype; returns it."""
    rows, vocab = logits.shape
    if rows == 0:
        return logits
    if logits.stride(1) != 1:
        raise ValueError("logits must be contiguous along the vocabulary")
    _logprob_bwd_kernel[(rows,)](
        logits,
        logits.stride(0),
        labels,
        inv_temperature,
        grad_logprobs,
        logz,
        others_minus_target,
        vocab,
        float(softcap) if softcap else 1.0,
        BLOCK_V=_BLOCK_V,
        HAS_SOFTCAP=bool(softcap),
        num_warps=_NUM_WARPS,
    )
    return logits
