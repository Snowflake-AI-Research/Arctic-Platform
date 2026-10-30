"""Folding a token's several expert results back onto one row.

Token-choice routing sends every token to ``top_k`` experts, so one token's layer output is a sum of several rows
of expert output. The expert-parallel combines in ``distributed.deepep`` and ``distributed.ucclep``, and the
rank-local paths in ``layers.moe``, all perform that sum. They share the implementation here so that the order
it sums in, and any change to it, is one property of the model rather than four properties of whichever combine
a job happens to select.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


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
        bags = torch.argsort(token_indices, stable=True)
        tokens = torch.arange(num_tokens, dtype=token_indices.dtype, device=token_indices.device)
        # Unlike CUDA bincount, fixed-size searchsorted does not read index extrema back to the host.
        offsets = torch.searchsorted(token_indices.index_select(0, bags), tokens)
        return F.embedding_bag(bags, rows, offsets, mode="sum")

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        # Each row reaches one token: its gradient is that token's, as scatter_add's backward gives it.
        (token_indices,) = ctx.saved_tensors
        return grad.index_select(0, token_indices), None, None
