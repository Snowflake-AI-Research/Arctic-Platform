"""Folding a token's several expert results back onto one row.

Token-choice routing sends every token to ``top_k`` experts, so one token's layer output is a sum of several rows
of expert output. The expert-parallel combines in ``distributed.deepep`` and ``distributed.ucclep``, and the
rank-local paths in ``layers.moe``, all perform that sum. They share the implementation here so that the order
it sums in, and any change to it, is one property of the model rather than four properties of whichever combine
a job happens to select. The permutation that fans a token out to its expert rows, ``gather_token_rows``, does the
same fold in backward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs keep index_add_
    triton = None

if triton is not None:

    @triton.jit
    def _sum_rows_by_token_kernel(
        rows_ptr, order_ptr, starts_ptr, out_ptr, D, stride_rows, stride_out, BLOCK_D: tl.constexpr
    ):
        token = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BLOCK_D + tl.arange(0, BLOCK_D)
        mask = cols < D
        start = tl.load(starts_ptr + token)
        end = tl.load(starts_ptr + token + 1)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for i in range(start, end):
            row = tl.load(order_ptr + i).to(tl.int64)
            acc += tl.load(rows_ptr + row * stride_rows + cols, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + token * stride_out + cols, acc.to(out_ptr.dtype.element_ty), mask=mask)


_BLOCK_D = 1024


def sum_rows_by_token(rows: torch.Tensor, token_indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """Sum each token's rows, accumulating 16-bit inputs in fp32 and rounding once, as vLLM's ``moe_sum`` does.

    ``token_indices[i]`` names the destination token of ``rows[i]``, and a token appears once per expert holding
    it, so several rows fold onto one destination. ``embedding_bag`` adds each token's rows in the order they
    appear in ``rows``, accumulating 16-bit rows in fp32 (CUDA ``EmbeddingBag_updateOutputKernel_sum_mean``),
    where a ``scatter_add`` into a 16-bit output rounds after every atomic, in an order the device chooses. Rows
    the index does not name are dropped.
    """
    return _SumRowsByToken.apply(rows, token_indices, num_tokens)


class _SumRowsByToken(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rows: torch.Tensor, token_indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
        ctx.save_for_backward(token_indices)
        sorted_tokens, bags = torch.sort(token_indices, stable=True)
        tokens = torch.arange(num_tokens, dtype=token_indices.dtype, device=token_indices.device)
        # Unlike CUDA bincount, fixed-size searchsorted does not read index extrema back to the host.
        return F.embedding_bag(bags, rows, torch.searchsorted(sorted_tokens, tokens), mode="sum")

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        # Each row reaches one token: its gradient is that token's, as scatter_add's backward gives it.
        (token_indices,) = ctx.saved_tensors
        return grad.index_select(0, token_indices), None, None


def _gather_sum_rows_by_token(rows: torch.Tensor, token_indices: torch.Tensor, num_tokens: int) -> torch.Tensor:
    """``sum_rows_by_token`` without atomics: each token gathers its rows, in row order, and sums them in FP32."""
    order = torch.argsort(token_indices, stable=True)
    starts = torch.zeros(num_tokens + 1, dtype=torch.int64, device=rows.device)
    torch.cumsum(torch.bincount(token_indices, minlength=num_tokens), dim=0, out=starts[1:])
    output = torch.empty((num_tokens, rows.shape[1]), dtype=rows.dtype, device=rows.device)
    if num_tokens:
        rows = rows if rows.stride(-1) == 1 else rows.contiguous()
        grid = (num_tokens, triton.cdiv(rows.shape[1], _BLOCK_D))
        _sum_rows_by_token_kernel[grid](
            rows, order, starts, output, rows.shape[1], rows.stride(0), output.stride(0), BLOCK_D=_BLOCK_D
        )
    return output


class _GatherTokenRows(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
        ctx.num_tokens = hidden_states.shape[0]
        ctx.save_for_backward(token_indices)
        return hidden_states.index_select(0, token_indices)

    @staticmethod
    def backward(ctx, grad_rows: torch.Tensor):
        (token_indices,) = ctx.saved_tensors
        return _gather_sum_rows_by_token(grad_rows, token_indices, ctx.num_tokens), None


def gather_token_rows(hidden_states: torch.Tensor, token_indices: torch.Tensor) -> torch.Tensor:
    """``hidden_states[token_indices]``: one row per routing entry, ``token_indices[i]`` naming its token.

    The gradient sums each token's rows back onto it. ``index_select``'s own backward does that with an atomic
    per row into BF16 (at 256K tokens over 8 GPUs, ~5 ms for each MoE layer's 262K rows on H200, and in delivery
    order); on CUDA each token instead gathers its rows in row order and sums them in FP32, one pass over the
    gradient.
    """
    if triton is None or not hidden_states.is_cuda or hidden_states.ndim != 2:
        return hidden_states.index_select(0, token_indices)
    return _GatherTokenRows.apply(hidden_states, token_indices)
