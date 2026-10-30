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
    """Sum the rows belonging to each token, with one atomic per row.

    ``token_indices[i]`` names the destination token of ``rows[i]``, and a token appears once per expert holding
    it, so several rows fold onto one destination. Atomics aimed at a repeated destination are applied in
    delivery order, which is the device's to choose: a token holding three or more of its experts on one rank
    sums them in an order two runs of one configuration need not agree on. Rows the index does not name are
    dropped, because the scatter iterates over the index rather than over the source.
    """
    hidden_dim = rows.shape[1]
    output = rows.new_zeros((num_tokens, hidden_dim))
    output.scatter_add_(0, token_indices.unsqueeze(1).expand(-1, hidden_dim), rows)
    return output


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
