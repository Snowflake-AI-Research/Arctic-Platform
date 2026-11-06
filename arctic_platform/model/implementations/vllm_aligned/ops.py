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
"""Training operators whose forward is the kernel a batch-invariant vLLM sampler runs, with a native backward.

Each forward calls the function vLLM 0.32 calls for the same operator on CUDA with ``VLLM_BATCH_INVARIANT=1`` and
``enforce_eager=True``, on tensors of the same shape, dtype and layout, so the output has the sampler's bytes:

* linear layers: ``linear_batch_invariant``, a Triton persistent matmul whose tile shape is looked up by ``(N, K)``
  and whose ``BLOCK_K`` is fixed per shape, so a row's reduction order does not depend on how many rows a call has;
* RMSNorm on one tensor: ``rms_norm_batch_invariant``, one Triton program per row;
* a residual add followed by RMSNorm: vLLM's ``fused_add_rms_norm`` CUDA kernel, whose block size is pinned to 1024
  when ``VLLM_BATCH_INVARIANT`` is set. It rounds ``input + residual`` to the input dtype and normalizes the rounded
  sum, so it is called here with a zero input and the already-rounded sum as the residual, which gives the same
  bytes and lets the residual add stay an ordinary tensor add;
* rotary embedding: ``rotary_embedding`` on a cos/sin cache built as vLLM builds it;
* SwiGLU: ``silu_and_mul``;
* attention: ``flash_attn_varlen_func`` from ``vllm.vllm_flash_attn`` over a paged KV cache with the sampler's page
  size, ``num_splits=1`` and no scheduler metadata. vLLM asserts in ``test_decode_logprobs_match_prefill_logprobs``
  that a prefill over the full sequence reproduces each decode step bitwise in this mode, which is what lets one
  full-sequence call here stand for the sampler's token-by-token decode;
* the sampled token's logprob: Model Runner V2's ``compute_token_logprobs`` kernel, vLLM 0.32's default runner.

The backward of every operator is the ordinary math in the training dtype or fp32. It is not bitwise to anything;
attention's backward recomputes the forward with the trainer's FlashAttention, so its gradient is that of a forward
that differs from the one above in the last bits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

BATCH_INVARIANT_ENV = "VLLM_BATCH_INVARIANT"


def require_batch_invariant_env() -> None:
    """Set ``VLLM_BATCH_INVARIANT=1`` for this process, or fail if it is set to something else.

    vLLM's CUDA ops read the variable once, at their first call, and keep the answer for the life of the process, so
    it has to be set before any of them runs; the normalization kernels pick a different block size without it.
    """
    value = os.environ.get(BATCH_INVARIANT_ENV)
    if value is None:
        os.environ[BATCH_INVARIANT_ENV] = "1"
        return
    try:
        enabled = int(value) != 0
    except ValueError:
        enabled = False
    if not enabled:
        raise RuntimeError(
            f"{BATCH_INVARIANT_ENV}={value!r} in the trainer environment; the vLLM-aligned forward needs it set to 1"
        )


def _batch_invariant():
    from vllm.model_executor.determinism import batch_invariant

    return batch_invariant


def _custom_ops():
    from vllm import _custom_ops

    return _custom_ops


def _silu_and_mul_kernel(out: torch.Tensor, x: torch.Tensor) -> None:
    _custom_ops()  # loads vllm._C, which registers torch.ops._C
    torch.ops._C.silu_and_mul(out, x)


def _rms_norm_backward(x: torch.Tensor, weight: torch.Tensor, eps: float, grad: torch.Tensor):
    hidden = x.float()
    inv_rms = torch.rsqrt(hidden.pow(2).mean(-1, keepdim=True) + eps)
    normed = hidden * inv_rms
    weighted_grad = grad.float() * weight.float()
    grad_x = inv_rms * (weighted_grad - normed * (weighted_grad * normed).mean(-1, keepdim=True))
    grad_weight = (grad.float() * normed).reshape(-1, x.shape[-1]).sum(0)
    return grad_x.to(x.dtype), grad_weight.to(weight.dtype)


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias):
        ctx.save_for_backward(x, weight)
        ctx.has_bias = bias is not None
        return _batch_invariant().linear_batch_invariant(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        grad_2d = grad_out.reshape(-1, grad_out.shape[-1])
        grad_x = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_x = (grad_2d @ weight).view(x.shape)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_2d.t() @ x.reshape(-1, x.shape[-1])
        if ctx.has_bias and ctx.needs_input_grad[2]:
            grad_bias = grad_2d.sum(0)
        return grad_x, grad_weight, grad_bias


def linear(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """``x @ weight.T + bias`` through vLLM's batch-invariant matmul.

    The tile shape is chosen by ``(N, K)``, so a caller must pass the weight with the sampler's row count: the
    concatenated q/k/v and gate/up weights rather than their pieces.
    """
    if x.dtype != weight.dtype:
        raise TypeError(f"vLLM-aligned linear needs matching dtypes, got input {x.dtype} and weight {weight.dtype}")
    return _Linear.apply(x, weight, bias)


class _RMSNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, fused):
        ctx.save_for_backward(x, weight)
        ctx.eps = eps
        if not fused:
            return _batch_invariant().rms_norm_batch_invariant(x, weight, eps)
        hidden = x.reshape(-1, x.shape[-1])
        out = torch.zeros(hidden.shape, dtype=hidden.dtype, device=hidden.device)
        _custom_ops().fused_add_rms_norm(out, hidden.clone(memory_format=torch.contiguous_format), weight, eps)
        return out.view(x.shape)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        grad_x, grad_weight = _rms_norm_backward(x, weight, ctx.eps, grad_out)
        return grad_x, grad_weight, None, None


def gemma_rms_norm_kernel(
    x: torch.Tensor, weight: torch.Tensor, eps: float, residual: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """vLLM's ``GemmaRMSNorm`` (``x * (1 + w)``) without autograd, as its native IR op runs under batch invariance.

    With ``residual``, the input is ``x + residual`` summed in fp32; the returned residual is that sum rounded to
    the input dtype, while the norm reads the unrounded fp32 sum. The mean is vLLM's batch-invariant Triton mean,
    which replaces ``aten::mean.dim`` in a batch-invariant sampler.
    """
    batch_invariant = _batch_invariant()
    orig_dtype = x.dtype
    gemma_weight = weight.float() + 1.0
    hidden = x.to(torch.float32)
    new_residual = None
    if residual is not None:
        hidden = hidden + residual.to(torch.float32)
        new_residual = hidden.to(orig_dtype)
    variance = batch_invariant.mean_batch_invariant(hidden.pow(2), [-1], True)
    hidden = hidden * torch.rsqrt(variance + eps)
    hidden = hidden.to(gemma_weight.dtype) * gemma_weight
    return hidden.to(orig_dtype), new_residual


def silu_and_mul_kernel(x: torch.Tensor) -> torch.Tensor:
    """The forward of :func:`silu_and_mul` without autograd."""
    out = torch.empty(x.shape[:-1] + (x.shape[-1] // 2,), dtype=x.dtype, device=x.device)
    _silu_and_mul_kernel(out, x.contiguous())
    return out


def linear_kernel(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
    """The forward of :func:`linear` without autograd."""
    return _batch_invariant().linear_batch_invariant(x, weight, bias)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, *, fused: bool) -> torch.Tensor:
    """RMSNorm of ``x`` over its last dimension with the kernel the sampler runs at this position.

    ``fused=False`` is vLLM's norm of a single tensor: the first decoder layer's input norm and the q/k norms.
    ``fused=True`` is the norm vLLM evaluates inside ``fused_add_rms_norm``; ``x`` is then the rounded residual sum
    the sampler's kernel would form, which a caller gets with an ordinary add of the two branches.
    """
    return _RMSNorm.apply(x, weight, eps, fused)


class _Rotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, positions, query, key, cos_sin_cache, head_size, is_neox):
        query = query.contiguous().clone()
        key = key.contiguous().clone()
        _custom_ops().rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)
        ctx.save_for_backward(positions, cos_sin_cache)
        ctx.head_size = head_size
        ctx.is_neox = is_neox
        return query, key

    @staticmethod
    def backward(ctx, grad_query, grad_key):
        positions, cos_sin_cache = ctx.saved_tensors
        grad_query = grad_query.contiguous().clone()
        grad_key = grad_key.contiguous().clone()
        # The rotation is orthogonal, so its transpose is the rotation by the negated angle.
        _custom_ops().rotary_embedding(
            positions, grad_query, grad_key, ctx.head_size, cos_sin_cache, ctx.is_neox, 0, True
        )
        return None, grad_query, grad_key, None, None, None


def rotary(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    head_size: int,
    *,
    is_neox: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate ``query`` and ``key`` (``[tokens, heads * head_size]``) by vLLM's rotary kernel."""
    return _Rotary.apply(positions, query, key, cos_sin_cache, head_size, is_neox)


def rotary_cos_sin_cache(
    *, rotary_dim: int, base: float | int, max_position: int, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """The ``[max_position, rotary_dim]`` cos/sin cache of vLLM's default ``RotaryEmbedding``.

    The same expressions as ``RotaryEmbeddingBase._compute_cos_sin_cache``, evaluated in fp32 on the GPU as vLLM
    does when it builds the model under a CUDA default device, then cast to the model dtype. ``base`` is the config's
    ``rope_theta`` as given, since vLLM passes it on unconverted.
    """
    inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float, device=device) / rotary_dim))
    positions = torch.arange(max_position, dtype=torch.float, device=device)
    freqs = torch.einsum("i,j -> ij", positions, inv_freq)
    return torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dtype)


class _SiluAndMul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        half = x.shape[-1] // 2
        out = torch.empty(x.shape[:-1] + (half,), dtype=x.dtype, device=x.device)
        _silu_and_mul_kernel(out, x)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        half = x.shape[-1] // 2
        gate = x[..., :half].float()
        up = x[..., half:].float()
        sigmoid = torch.sigmoid(gate)
        grad = grad_out.float()
        grad_gate = grad * up * sigmoid * (1.0 + gate * (1.0 - sigmoid))
        grad_up = grad * gate * sigmoid
        return torch.cat((grad_gate, grad_up), dim=-1).to(x.dtype)


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    """``silu(x[..., :d]) * x[..., d:]`` by vLLM's kernel; ``x`` is the concatenated gate/up projection."""
    return _SiluAndMul.apply(x.contiguous())


@dataclass(frozen=True)
class PackedLayout:
    """Where each packed sequence lies in a ``[tokens]`` window, and the paged KV layout the sampler would use.

    The sampler stores a sequence's keys and values in pages of ``block_size`` tokens and attends through a block
    table. The layout here gives every sequence its own run of pages, so a full-sequence attention call reads the
    cache the way the sampler's prefill and decode do.
    """

    positions: torch.Tensor
    cu_seqlens: torch.Tensor
    seq_lens: torch.Tensor
    max_seqlen: int
    block_size: int
    block_table: torch.Tensor
    slots: torch.Tensor
    num_blocks: int

    @property
    def num_tokens(self) -> int:
        return int(self.positions.numel())

    @classmethod
    def build(cls, position_ids: torch.Tensor, *, block_size: int, cu_seqlens: torch.Tensor | None = None):
        """Lay out ``position_ids`` (``[batch, seq]``), one sequence per run that starts at position 0.

        A row boundary of a ``[batch, seq]`` input also starts a sequence. ``cu_seqlens`` overrides the boundaries
        when the caller has them.
        """
        positions = position_ids.reshape(-1).to(torch.long)
        device = positions.device
        num_tokens = int(positions.numel())
        if cu_seqlens is None:
            starts = positions == 0
            starts[0] = True
            if position_ids.dim() == 2 and position_ids.shape[0] > 1:
                starts[:: position_ids.shape[1]] = True
            boundaries = torch.nonzero(starts, as_tuple=False).flatten()
            cu_seqlens = torch.cat((boundaries, torch.tensor([num_tokens], device=device)))
        cu_seqlens = cu_seqlens.to(device=device, dtype=torch.int32)
        if int(cu_seqlens[0].item()) != 0 or int(cu_seqlens[-1].item()) != num_tokens:
            raise ValueError(f"cu_seqlens must run from 0 to the token count {num_tokens}, got {cu_seqlens.tolist()}")
        seq_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)
        if bool((seq_lens <= 0).any().item()):
            raise ValueError("every packed sequence must hold at least one token")
        blocks_per_seq = (seq_lens + block_size - 1) // block_size
        first_block = torch.cumsum(blocks_per_seq, 0) - blocks_per_seq
        num_blocks = int(blocks_per_seq.sum().item())
        max_blocks = int(blocks_per_seq.max().item())
        columns = torch.arange(max_blocks, device=device, dtype=torch.int32)
        block_table = torch.where(
            columns[None, :] < blocks_per_seq[:, None],
            first_block[:, None].to(torch.int32) + columns[None, :],
            torch.zeros((), device=device, dtype=torch.int32),
        ).contiguous()
        seq_of_token = torch.repeat_interleave(torch.arange(seq_lens.numel(), device=device), seq_lens.to(torch.long))
        offset_in_seq = torch.arange(num_tokens, device=device) - cu_seqlens[:-1].to(torch.long)[seq_of_token]
        slots = first_block.to(torch.long)[seq_of_token] * block_size + offset_in_seq
        return cls(
            positions=positions,
            cu_seqlens=cu_seqlens,
            seq_lens=seq_lens,
            max_seqlen=int(seq_lens.max().item()),
            block_size=block_size,
            block_table=block_table,
            slots=slots,
            num_blocks=num_blocks,
        )


def _fa_utils():
    from vllm.v1.attention.backends import fa_utils

    return fa_utils


def _trainer_flash_attn_varlen_func():
    """The trainer's differentiable FlashAttention varlen call: FA3 on Hopper, else FA2, else FA4.

    The returned function takes ``(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale=,
    causal=)`` and returns the attention output alone, whichever package provides it.
    """
    try:
        from flash_attn_interface import flash_attn_varlen_func as fa3
    except ImportError:
        fa3 = None
    if fa3 is not None:

        def call_fa3(q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale=None, causal=False):
            out = fa3(q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale=softmax_scale, causal=causal)
            return out[0] if isinstance(out, tuple) else out

        return call_fa3
    try:
        from flash_attn import flash_attn_varlen_func as fa2

        return fa2
    except ImportError:
        pass
    try:
        from flash_attn.cute import flash_attn_varlen_func as fa4
    except ImportError as error:
        raise ImportError(
            "the vLLM-aligned attention backward needs flash_attn_interface (FA3), flash_attn (FA2) or "
            "flash_attn.cute (FA4)"
        ) from error

    def call_fa4(q, k, v, cu_q, cu_k, max_q, max_k, softmax_scale=None, causal=False):
        out = fa4(q, k, v, cu_seqlens_q=cu_q, cu_seqlens_k=cu_k, softmax_scale=softmax_scale, causal=causal)
        return out[0] if isinstance(out, tuple) else out

    return call_fa4


def paged_attention_kernel(
    query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, layout: PackedLayout, scale: float, fa_version: int
) -> torch.Tensor:
    """The forward of :func:`paged_attention` without autograd."""
    fa_utils = _fa_utils()
    num_kv_heads, head_size = key.shape[1], key.shape[2]
    cache_shape = (layout.num_blocks * layout.block_size, num_kv_heads, head_size)
    key_cache = key.new_zeros(cache_shape).index_copy_(0, layout.slots, key)
    value_cache = value.new_zeros(cache_shape).index_copy_(0, layout.slots, value)
    paged_shape = (layout.num_blocks, layout.block_size, num_kv_heads, head_size)
    num_seqs = int(layout.seq_lens.numel())
    unit_scale = torch.ones((), dtype=torch.float32, device=query.device).expand(num_seqs, num_kv_heads)
    return fa_utils.flash_attn_varlen_func(
        q=query,
        k=key_cache.view(paged_shape),
        v=value_cache.view(paged_shape),
        out=torch.empty_like(query),
        cu_seqlens_q=layout.cu_seqlens,
        max_seqlen_q=layout.max_seqlen,
        seqused_k=layout.seq_lens,
        max_seqlen_k=layout.max_seqlen,
        softmax_scale=scale,
        causal=True,
        alibi_slopes=None,
        window_size=None,
        block_table=layout.block_table,
        softcap=0,
        scheduler_metadata=None,
        fa_version=fa_version,
        q_descale=unit_scale if fa_utils.flash_attn_supports_quant_query_input() else None,
        k_descale=unit_scale,
        v_descale=unit_scale,
        num_splits=1,
        s_aux=None,
    )


class _PagedAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, layout: PackedLayout, scale: float, fa_version: int):
        out = paged_attention_kernel(query, key, value, layout, scale, fa_version)
        ctx.save_for_backward(query, key, value, layout.cu_seqlens)
        ctx.max_seqlen = layout.max_seqlen
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, grad_out):
        flash_attn_varlen_func = _trainer_flash_attn_varlen_func()
        query, key, value, cu_seqlens = ctx.saved_tensors
        with torch.enable_grad():
            inputs = tuple(t.detach().requires_grad_(True) for t in (query, key, value))
            out = flash_attn_varlen_func(
                *inputs,
                cu_seqlens,
                cu_seqlens,
                ctx.max_seqlen,
                ctx.max_seqlen,
                softmax_scale=ctx.scale,
                causal=True,
            )
            grads = torch.autograd.grad(out, inputs, grad_out)
        return grads[0], grads[1], grads[2], None, None, None


def flash_attn_version(head_size: int) -> int:
    """The FlashAttention version vLLM's FLASH_ATTN backend picks on this GPU for ``head_size``."""
    version = _fa_utils().get_flash_attn_version(head_size=head_size)
    if version is None:
        raise RuntimeError("vLLM reports no FlashAttention version for this GPU")
    return int(version)


def paged_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layout: PackedLayout,
    *,
    scale: float,
    fa_version: int,
) -> torch.Tensor:
    """Causal attention of each packed sequence, ``[tokens, heads, head_size]`` in and out, as vLLM computes it."""
    return _PagedAttention.apply(query, key, value, layout, scale, fa_version)


def _padded_vocab(vocab_size: int, padding: int) -> int:
    return -(-vocab_size // padding) * padding


def _token_logprobs_kernel(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    from vllm.v1.worker.gpu.sample.logprob import compute_token_logprobs

    return compute_token_logprobs(logits, token_ids[:, None])[:, 0]


def lm_head_logprobs_kernel(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    temperature: float,
    token_chunk_size: int,
    vocab_padding: int,
) -> torch.Tensor:
    """The forward of :func:`lm_head_logprobs` without autograd; ``labels`` must be valid token ids."""
    batch_invariant = _batch_invariant()
    vocab_size = weight.shape[0]
    padded = _padded_vocab(vocab_size, vocab_padding)
    lm_weight = weight
    if padded != vocab_size:
        lm_weight = torch.cat((weight, weight.new_zeros(padded - vocab_size, weight.shape[1])))
    out = torch.empty(hidden.shape[0], dtype=torch.float32, device=hidden.device)
    for start in range(0, hidden.shape[0], token_chunk_size):
        stop = min(start + token_chunk_size, hidden.shape[0])
        logits = batch_invariant.linear_batch_invariant(hidden[start:stop], lm_weight)[:, :vocab_size]
        if temperature != 1.0:
            logits = logits.float() / torch.tensor(temperature, dtype=torch.float32, device=logits.device)
        out[start:stop] = _token_logprobs_kernel(logits, labels[start:stop])
    return out


class _LMHeadLogprobs(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, temperature, token_chunk_size, vocab_padding):
        out = lm_head_logprobs_kernel(hidden, weight, labels, temperature, token_chunk_size, vocab_padding)
        ctx.save_for_backward(hidden, weight, labels)
        ctx.temperature = temperature
        ctx.token_chunk_size = token_chunk_size
        return out

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, labels = ctx.saved_tensors
        grad_hidden = torch.empty_like(hidden)
        grad_weight = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
        for start in range(0, hidden.shape[0], ctx.token_chunk_size):
            stop = min(start + ctx.token_chunk_size, hidden.shape[0])
            hidden_chunk = hidden[start:stop]
            logits = (hidden_chunk @ weight.t()).float()
            if ctx.temperature != 1.0:
                logits = logits / ctx.temperature
            grad = grad_out[start:stop].float()
            grad_logits = torch.softmax(logits, dim=-1).mul_(-grad[:, None])
            grad_logits.scatter_add_(1, labels[start:stop, None], grad[:, None])
            if ctx.temperature != 1.0:
                grad_logits = grad_logits / ctx.temperature
            grad_logits = grad_logits.to(weight.dtype)
            grad_hidden[start:stop] = grad_logits @ weight
            grad_weight.add_((grad_logits.t() @ hidden_chunk).float())
        return grad_hidden, grad_weight.to(weight.dtype), None, None, None, None


def lm_head_logprobs(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 1.0,
    token_chunk_size: int = 1024,
    vocab_padding: int = 64,
) -> torch.Tensor:
    """Log-probability of ``labels`` (``[tokens]``) under the logits of ``hidden`` (``[tokens, hidden]``).

    The logits are vLLM's: the LM-head matmul over the vocabulary padded to a multiple of ``vocab_padding`` (vLLM's
    ``ParallelLMHead`` padding, which sets the matmul's ``N``), in the model dtype, divided in fp32 by the
    temperature when it is not 1. The logprob of each label is then the value Model Runner V2's sampler reports,
    from its ``compute_token_logprobs`` kernel: ``logit - max - log(sum(exp(logit - max)))`` in fp32, which rounds
    differently from a full log-softmax followed by a gather. Rows are independent, so chunking them changes nothing.
    A label of -100 scores 0 and receives no gradient.
    """
    ignored = labels == -100
    safe_labels = labels.masked_fill(ignored, 0).to(torch.long)
    logprobs = _LMHeadLogprobs.apply(
        hidden, weight, safe_labels, float(temperature), int(token_chunk_size), int(vocab_padding)
    )
    return logprobs.masked_fill(ignored, 0.0)
