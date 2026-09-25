"""Folding a token's several expert results back onto one row.

Token-choice routing sends every token to ``top_k`` experts, so one token's layer output is a sum of several rows
of expert output. The expert-parallel combines in ``distributed.deepep`` and ``distributed.ucclep``, and the
rank-local paths in ``layers.moe``, all perform that sum. They share the implementation here so that the order
it sums in, and any change to it, is one property of the model rather than four properties of whichever combine
a job happens to select.
"""

from __future__ import annotations

import torch


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
