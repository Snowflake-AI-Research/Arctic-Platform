"""Apply expert routing weights without retaining an FP32 activation copy."""

from __future__ import annotations

import torch


class _ApplyRoutingWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(hidden_states, scores)
        return (hidden_states.float() * scores.float().reshape(-1, 1)).to(hidden_states.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden_states, scores = ctx.saved_tensors
        grad_output_fp32 = grad_output.float()
        grad_hidden = grad_scores = None
        if ctx.needs_input_grad[0]:
            grad_hidden = (grad_output_fp32 * scores.float().reshape(-1, 1)).to(hidden_states.dtype)
        if ctx.needs_input_grad[1]:
            # Recompute the cast at use rather than saving it for every routed chunk.
            grad_scores = (grad_output_fp32 * hidden_states.float()).sum(dim=1).reshape(scores.shape).to(scores.dtype)
        return grad_hidden, grad_scores


def apply_routing_weights(hidden_states: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Preserve FP32 multiply/reduction while saving source tensors for backward."""
    return _ApplyRoutingWeights.apply(hidden_states, scores)
