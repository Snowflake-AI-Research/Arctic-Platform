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
"""Keep the attention kernel's outputs through activation checkpointing instead of recomputing the kernel.

Whole-layer checkpointing replays the layer's forward in the backward pass, attention kernel included. At long
sequence lengths that kernel is most of the layer's forward time, while its outputs (the attention output and its
log-sum-exp) are as small as one projection output. This patch re-arms the checkpoint functions the
``gradient_checkpointing`` patch installed with a selective-checkpoint policy that saves those outputs and recomputes
every other op, so the backward recompute rebuilds the kernel's inputs (the projections, norms and rotary
embedding, and any sequence-parallel all-to-all) and reuses its outputs.
"""

from __future__ import annotations

import functools

import torch.nn as nn

from arctic_platform.model.loader import LoaderContext
from arctic_platform.model.patch import register_patch

# Attention-kernel ops other modules register (register_attention_kernel_op), e.g. a model's own TileLang kernel.
_REGISTERED_KERNEL_OPS: set = set()


def register_attention_kernel_op(op) -> None:
    """Have the keep-attention policy save the outputs of ``op`` as well: an attention kernel's forward op, given as
    an ``OpOverload`` or as an ``OpOverloadPacket`` (all of its overloads), e.g. ``torch.ops.mylib.attn_fwd``.

    The op must be a dispatcher op (``torch.library.custom_op`` or a registered C++ op) whose outputs are everything
    its backward reads besides its recomputable inputs; a plain Python function is invisible to the policy. Takes
    effect for checkpoint functions installed before or after the call.
    """
    import torch

    if isinstance(op, torch._ops.OpOverloadPacket):
        _REGISTERED_KERNEL_OPS.update(getattr(op, overload) for overload in op.overloads())
    elif isinstance(op, torch._ops.OpOverload):
        _REGISTERED_KERNEL_OPS.add(op)
    else:
        raise TypeError(
            f"register_attention_kernel_op takes an OpOverload or OpOverloadPacket, got {type(op).__name__}"
        )


def attention_kernel_ops() -> frozenset:
    """The forward attention-kernel ops whose outputs the policy saves: FlashAttention 3 and 2, and SDPA's kernels.

    A kernel op that calls another (FlashAttention 3's ``_flash_attn_forward`` calls ``fwd``) is saved at the outer
    call; the inner one runs inside it and is not seen separately.
    """
    import torch

    names = [
        ("flash_attn_3", "_flash_attn_forward"),
        ("flash_attn_3", "fwd"),
        ("flash_attn", "_flash_attn_forward"),
        ("flash_attn", "_flash_attn_varlen_forward"),
        ("aten", "_scaled_dot_product_flash_attention"),
        ("aten", "_scaled_dot_product_cudnn_attention"),
        ("aten", "_scaled_dot_product_efficient_attention"),
        ("aten", "_flash_attention_forward"),
        ("aten", "_efficient_attention_forward"),
        ("aten", "_scaled_dot_product_flash_attention_for_cpu"),
    ]
    try:
        import flash_attn_interface  # noqa: F401  # registers torch.ops.flash_attn_3
    except (ImportError, OSError):
        pass
    ops = set()
    for namespace, name in names:
        packet = getattr(getattr(torch.ops, namespace), name, None)
        if packet is None:
            continue
        for overload in packet.overloads():
            ops.add(getattr(packet, overload))
    return frozenset(ops | _REGISTERED_KERNEL_OPS)


def keep_attention_checkpoint_fn():
    """``torch.utils.checkpoint.checkpoint`` (non-reentrant) with the keep-attention selective policy."""
    from torch.utils.checkpoint import CheckpointPolicy
    from torch.utils.checkpoint import checkpoint
    from torch.utils.checkpoint import create_selective_checkpoint_contexts

    kernels = attention_kernel_ops()
    if not kernels:
        raise RuntimeError("checkpoint_keep_attention found no attention-kernel op to keep")

    def policy(ctx, op, *args, **kwargs):
        keep = op in kernels or op in _REGISTERED_KERNEL_OPS
        return CheckpointPolicy.MUST_SAVE if keep else CheckpointPolicy.PREFER_RECOMPUTE

    return functools.partial(
        checkpoint,
        use_reentrant=False,
        context_fn=functools.partial(create_selective_checkpoint_contexts, policy),
    )


@register_patch("checkpoint_keep_attention")
def apply_checkpoint_keep_attention(model: nn.Module, ctx: LoaderContext) -> None:
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointWrapper

    checkpoint_fn = keep_attention_checkpoint_fn()
    rearmed = 0
    for module in model.modules():
        if isinstance(module, CheckpointWrapper):
            # The periodic path wraps every Nth layer; the wrapper calls checkpoint_fn(module, *args, **kwargs).
            module.checkpoint_fn = checkpoint_fn
            rearmed += 1
        elif getattr(module, "gradient_checkpointing", False) is True and hasattr(
            module, "_gradient_checkpointing_func"
        ):
            # transformers' GradientCheckpointingLayer calls _gradient_checkpointing_func(partial(forward), *args).
            module._gradient_checkpointing_func = checkpoint_fn
            rearmed += 1
    if rearmed == 0:
        raise ValueError(
            "checkpoint_keep_attention found no checkpointed layer to re-arm; it needs gradient_checkpointing on a "
            "model whose layers transformers' gradient checkpointing or the periodic checkpoint wrapper covers"
        )
