"""Apply expert routing weights without retaining an FP32 activation copy."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only installs keep the PyTorch path
    triton = None

if triton is not None:

    @triton.jit
    def _scale_rows_kernel(x_ptr, s_ptr, out_ptr, D, stride_x, stride_out, BLOCK_D: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        scale = tl.load(s_ptr + row).to(tl.float32)
        for start in range(0, D, BLOCK_D):
            cols = start + tl.arange(0, BLOCK_D)
            mask = cols < D
            x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
            tl.store(out_ptr + row * stride_out + cols, (x * scale).to(out_ptr.dtype.element_ty), mask=mask)

    @triton.jit
    def _routing_weights_bwd_kernel(
        go_ptr,
        x_ptr,
        s_ptr,
        gx_ptr,
        gs_ptr,
        D,
        stride_go,
        stride_x,
        stride_gx,
        NEED_GX: tl.constexpr,
        NEED_GS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        row = tl.program_id(0).to(tl.int64)
        scale = tl.load(s_ptr + row).to(tl.float32)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)
        for start in range(0, D, BLOCK_D):
            cols = start + tl.arange(0, BLOCK_D)
            mask = cols < D
            go = tl.load(go_ptr + row * stride_go + cols, mask=mask, other=0.0).to(tl.float32)
            if NEED_GX:
                tl.store(gx_ptr + row * stride_gx + cols, (go * scale).to(gx_ptr.dtype.element_ty), mask=mask)
            if NEED_GS:
                x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
                acc += go * x
        if NEED_GS:
            tl.store(gs_ptr + row, tl.sum(acc, axis=0))


_BLOCK_D = 1024


def _fused(*tensors: torch.Tensor) -> bool:
    """One Triton pass per row on CUDA: FP32 math in registers instead of FP32 copies of whole activations."""
    return triton is not None and all(t.is_cuda for t in tensors) and tensors[0].ndim == 2


def _rows(t: torch.Tensor) -> torch.Tensor:
    return t if t.stride(-1) == 1 else t.contiguous()


class _ApplyRoutingWeights(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states: torch.Tensor, scores: torch.Tensor, out_dtype: torch.dtype) -> torch.Tensor:
        ctx.save_for_backward(hidden_states, scores)
        if not _fused(hidden_states, scores) or hidden_states.shape[0] == 0:
            return (hidden_states.float() * scores.float().reshape(-1, 1)).to(out_dtype)
        x = _rows(hidden_states)
        out = torch.empty(x.shape, dtype=out_dtype, device=x.device)
        _scale_rows_kernel[(x.shape[0],)](
            x, scores.reshape(-1).contiguous(), out, x.shape[1], x.stride(0), out.stride(0), BLOCK_D=_BLOCK_D
        )
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        hidden_states, scores = ctx.saved_tensors
        need_hidden, need_scores = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        # Double backward (create_graph) needs differentiable ops, so it keeps the PyTorch path.
        if _fused(hidden_states, scores, grad_output) and not torch.is_grad_enabled() and hidden_states.shape[0]:
            x, go = _rows(hidden_states), _rows(grad_output)
            grad_hidden = torch.empty(x.shape, dtype=hidden_states.dtype, device=x.device) if need_hidden else None
            grad_scores = torch.empty(x.shape[0], dtype=torch.float32, device=x.device) if need_scores else None
            hidden_stride = 0
            if need_hidden:
                assert grad_hidden is not None
                hidden_stride = grad_hidden.stride(0)
            _routing_weights_bwd_kernel[(x.shape[0],)](
                go,
                x,
                scores.reshape(-1).contiguous(),
                grad_hidden if need_hidden else go,
                grad_scores if need_scores else go,
                x.shape[1],
                go.stride(0),
                x.stride(0),
                hidden_stride,
                NEED_GX=need_hidden,
                NEED_GS=need_scores,
                BLOCK_D=_BLOCK_D,
            )
            if need_scores:
                assert grad_scores is not None
                grad_scores = grad_scores.reshape(scores.shape).to(scores.dtype)
            return grad_hidden, grad_scores, None
        grad_output_fp32 = grad_output.float()
        grad_hidden = grad_scores = None
        if need_hidden:
            grad_hidden = (grad_output_fp32 * scores.float().reshape(-1, 1)).to(hidden_states.dtype)
        if need_scores:
            # Recompute the cast at use rather than saving it for every routed chunk.
            grad_scores = (grad_output_fp32 * hidden_states.float()).sum(dim=1).reshape(scores.shape).to(scores.dtype)
        return grad_hidden, grad_scores, None


def apply_routing_weights(
    hidden_states: torch.Tensor, scores: torch.Tensor, *, out_dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Save source tensors for backward and cast the FP32 product directly to the output dtype.

    The output dtype defaults to hidden_states.dtype. On CUDA the product, and in backward the hidden
    gradient and the per-row score gradient (an FP32 dot), are each one Triton pass over the rows.
    """
    return _ApplyRoutingWeights.apply(hidden_states, scores, hidden_states.dtype if out_dtype is None else out_dtype)
