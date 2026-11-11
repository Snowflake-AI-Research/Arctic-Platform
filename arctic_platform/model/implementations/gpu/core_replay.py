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
"""Keep a checkpointed block's token-mixer kernels out of its recompute ("core replay").

Full activation checkpointing reruns every checkpointed block's forward in its backward, to rebuild the tensors the
block saved. In a long-sequence block most of that rerun is the token mixer's core kernel: FlashAttention's forward,
quadratic in the sequence, or a linear-attention / state-space scan. The backward only needs what that kernel saved,
and the original forward computed exactly that. Core replay keeps it from the forward and hands it to the recompute,
which then runs everything else of the block as before (projections, norms, the MoE).

* FlashAttention-3 varlen (``kept_flash_attn_3_varlen``): the forward keeps the kernel's output and log-sum-exp
  (``[tokens, heads, dim]`` and ``[heads, tokens]`` FP32), the recompute recomputes q, k and v and passes the kept
  pair to the same autograd node instead of running the kernel, and the backward is FlashAttention's own. Costs the
  output and its log-sum-exp per checkpointed attention layer, until that layer's backward.
* Any other core function (``keep_core``): the forward runs it with its saved tensors held by its own autograd nodes
  rather than by the checkpoint, and the recompute returns the forward's outputs without calling it. Costs what the
  function saves (its inputs and outputs, for a scan), until its backward.

Each kept value is queued under its forward's checkpoint frame (``combine_replay._checkpoint_frame``), so forwards
that are not yet recomputed keep theirs apart, and a recompute that finds nothing kept raises instead of guessing.
The arithmetic is unchanged: the backward reads the forward's own outputs rather than a recompute's copy of them.
"""

from __future__ import annotations

import logging
import weakref
from collections import deque
from typing import Any
from typing import Callable

import torch
from torch import nn

from arctic_platform.model.implementations.gpu.router_replay_recompute import _in_recompute
from arctic_platform.model.implementations.gpu.router_replay_recompute import _install_recompute_detector

logger = logging.getLogger(__name__)

_PLAIN, _KEEP, _REPLAY = "plain", "keep", "replay"
_frame_miss_warned = False


def _checkpoint_frame():
    # combine_replay's lookup of the running checkpoint's frame (imported here: the MoE package imports DeepEP).
    from arctic_platform.model.implementations.moe.distributed.combine_replay import _checkpoint_frame as lookup

    return lookup()


def _tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [t for item in value for t in _tensors(item)]
    return []


class CoreReplay:
    """One module's kept core outputs, one queue per checkpointed forward, oldest call first."""

    def __init__(self) -> None:
        _install_recompute_detector()
        self._by_forward: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

    def mode(self) -> str:
        """``keep`` on a graph-building forward inside a checkpoint, ``replay`` on its recompute, else ``plain``."""
        global _frame_miss_warned
        if not torch.is_grad_enabled():
            return _PLAIN
        frame = _checkpoint_frame()
        if frame is None:
            if _in_recompute() and not _frame_miss_warned:
                _frame_miss_warned = True
                logger.warning("core replay: no checkpoint frame in a recompute; the core kernel runs again")
            return _PLAIN
        if _in_recompute():
            if not self._by_forward.get(frame):
                raise RuntimeError("core replay: a recompute found no kept core output from its forward")
            return _REPLAY
        return _KEEP

    def keep(self, value: Any) -> None:
        frame = _checkpoint_frame()
        if frame is None:
            raise RuntimeError("core replay: keep() outside a checkpointed forward")
        queue = self._by_forward.get(frame)
        if queue is None:
            queue = self._by_forward[frame] = deque()
        queue.append((value, [t._version for t in _tensors(value)]))

    def take(self) -> Any:
        queue = self._by_forward.get(_checkpoint_frame())
        if not queue:
            raise RuntimeError("core replay: a recompute found no kept core output from its forward")
        value, versions = queue.popleft()
        if [t._version for t in _tensors(value)] != versions:
            raise RuntimeError("core replay: a kept core output was modified in place after its forward")
        return value


def _hold(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def keep_core(replay: CoreReplay | None, fn: Callable[..., Any], *args, **kwargs) -> Any:
    """``fn(*args, **kwargs)``, kept by ``replay`` from a checkpointed forward for its recompute.

    The forward call's saved tensors are held by its autograd nodes (the checkpoint does not drop them), so its
    backward needs no recompute of it; the recompute gets the forward's outputs back without calling ``fn``.
    """
    if replay is None:
        return fn(*args, **kwargs)
    mode = replay.mode()
    if mode == _REPLAY:
        return replay.take()
    if mode == _PLAIN:
        return fn(*args, **kwargs)
    with torch.autograd.graph.saved_tensors_hooks(_hold, _hold):
        out = fn(*args, **kwargs)
    replay.keep(out)
    return out


class _FlashAttn3VarlenKept(torch.autograd.Function):
    """FlashAttention-3 varlen whose forward kernel is skipped when the output and log-sum-exp are given.

    The same arguments and kernels as ``flash_attn_interface.FlashAttnVarlenFunc`` for the dense causal / windowed
    case Arctic's attention layers call (no paged KV, descales, softcap or attention chunks)."""

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
        deterministic,
        kept_out,
        kept_lse,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        if kept_out is None:
            from flash_attn_interface import _flash_attn_forward

            out, lse, *_ = _flash_attn_forward(
                q,
                k,
                v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                softmax_scale=softmax_scale,
                causal=causal,
                window_size_left=window_size[0],
                window_size_right=window_size[1],
            )
        else:
            out, lse = kept_out, kept_lse
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k)
        ctx.max_seqlen_q = max_seqlen_q
        ctx.max_seqlen_k = max_seqlen_k
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.deterministic = deterministic
        ctx.mark_non_differentiable(lse)
        return out, lse

    @staticmethod
    def backward(ctx, dout, _dlse):
        from flash_attn_interface import _flash_attn_backward

        q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k = ctx.saved_tensors
        dq, dk, dv = torch.empty_like(q), torch.empty_like(k), torch.empty_like(v)
        _flash_attn_backward(
            dout,
            q,
            k,
            v,
            out,
            lse,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            None,
            ctx.max_seqlen_q,
            ctx.max_seqlen_k,
            dq,
            dk,
            dv,
            ctx.softmax_scale,
            ctx.causal,
            ctx.window_size[0],
            ctx.window_size[1],
            0.0,
            ctx.deterministic,
            0,
        )
        dq, dk, dv = dq[..., : q.shape[-1]], dk[..., : k.shape[-1]], dv[..., : v.shape[-1]]
        return dq, dk, dv, None, None, None, None, None, None, None, None, None, None


def kept_flash_attn_3_varlen(replay: CoreReplay) -> Callable[..., torch.Tensor]:
    """A drop-in for ``flash_attn_interface.flash_attn_varlen_func`` (the arguments Arctic's layers pass) whose
    forward is kept by ``replay`` for the block's recompute."""

    def flash_attn_varlen(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=None,
        causal=False,
        window_size=(-1, -1),
        deterministic=False,
    ):
        mode = replay.mode()
        kept = replay.take() if mode == _REPLAY else (None, None)
        out, lse = _FlashAttn3VarlenKept.apply(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            softmax_scale,
            causal,
            tuple(window_size),
            deterministic,
            *kept,
        )
        if mode == _KEEP:
            replay.keep((out, lse))
        return out

    return flash_attn_varlen


def install_core_replay(block: nn.Module, scans: bool = True) -> int:
    """Keep the core kernels of ``block`` (about to be fully checkpointed) out of its recompute; returns the number
    of modules that keep one: FlashAttention-3 layers (``_flash_attn_call``) and, with ``scans``, modules that call
    ``keep_core`` with their ``_arl_core_replay`` (Mamba-2 mixers)."""
    installed = 0
    for module in block.modules():
        if getattr(module, "_arl_core_replay", None) is not None:
            continue
        if hasattr(module, "_flash_attn_call") and getattr(module, "_flash_attn_version", None) == 3:
            module._arl_core_replay = CoreReplay()
            module._flash_attn_call = kept_flash_attn_3_varlen(module._arl_core_replay)
            installed += 1
        elif scans and getattr(module, "_arl_keeps_core", False):
            module._arl_core_replay = CoreReplay()
            installed += 1
    return installed
