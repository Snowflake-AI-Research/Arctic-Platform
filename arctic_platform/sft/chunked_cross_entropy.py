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

"""Hugging Face causal cross-entropy with bounded fp32 intermediates."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


class _ChunkedCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, denominator, ignore_index, chunk_rows, mean_reduction):
        flat = logits.reshape(-1, logits.shape[-1])
        targets = targets.reshape(-1).to(logits.device)
        losses = torch.empty(flat.shape[0], dtype=torch.float32, device=logits.device)
        for start in range(0, flat.shape[0], chunk_rows):
            losses[start : start + chunk_rows] = F.cross_entropy(
                flat[start : start + chunk_rows].float(),
                targets[start : start + chunk_rows],
                ignore_index=ignore_index,
                reduction="none",
            )
        # Retain NLL's full-token reduction order without full-vocabulary fp32 tensors.
        reduction_targets = torch.zeros_like(targets).masked_fill(targets == ignore_index, ignore_index)
        total = F.nll_loss(-losses.unsqueeze(-1), reduction_targets, ignore_index=ignore_index, reduction="sum")
        ctx.save_for_backward(logits, targets, denominator)
        ctx.ignore_index = ignore_index
        ctx.chunk_rows = chunk_rows
        ctx.mean_reduction = mean_reduction
        return total / denominator

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        logits, targets, denominator = ctx.saved_tensors
        flat = logits.reshape(-1, logits.shape[-1])
        gradient = torch.empty_like(flat)
        # HF mean CE returns NaN but zero gradients for an all-ignored batch.
        scale = denominator.clamp_min(1) if ctx.mean_reduction else denominator
        # Replay the actual PyTorch CE backward, not a different softmax derivative.
        for start in range(0, flat.shape[0], ctx.chunk_rows):
            stop = start + ctx.chunk_rows
            with torch.enable_grad():
                chunk = flat[start:stop].detach().float().requires_grad_(True)
                loss = F.cross_entropy(chunk, targets[start:stop], ignore_index=ctx.ignore_index, reduction="sum")
                chunk_gradient = torch.autograd.grad(loss, chunk, grad_outputs=(grad_output / scale).to(loss.dtype))[0]
            gradient[start:stop] = chunk_gradient
        return gradient.reshape(logits.shape), None, None, None, None, None


def chunked_causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    num_items_in_batch: torch.Tensor | int | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
    *,
    chunk_rows: int = 512,
    **kwargs,
) -> torch.Tensor:
    """Preserve HF's shift, masking, fp32 CE and token denominator.

    Only first-order training backward is supported. Logits and their gradient
    retain the original dtype; fp32 CE tensors occupy at most one token chunk.
    """
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    if logits.shape[-1] != vocab_size:
        raise ValueError("vocab_size must match logits")
    if shift_labels is None:
        shift_labels = F.pad(labels, (0, 1), value=ignore_index)[..., 1:].contiguous()
    denominator = (shift_labels != ignore_index).sum() if num_items_in_batch is None else num_items_in_batch
    denominator = torch.as_tensor(denominator, device=logits.device)
    return _ChunkedCrossEntropy.apply(
        logits, shift_labels, denominator, ignore_index, chunk_rows, num_items_in_batch is None
    )
