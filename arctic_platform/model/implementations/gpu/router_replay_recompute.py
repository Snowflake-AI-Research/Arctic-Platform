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
"""Self router-replay across activation-checkpoint recompute.

Whole-block activation checkpointing reruns the whole decoder block on the backward recompute, including the
MoE router's ``torch.topk`` over the gate scores. Upstream non-determinism (matmul / flash-attn / DeepEP
reductions) perturbs those scores in the last bits, so ``topk`` can pick a different expert set on recompute
-> different ``num_tokens_per_expert`` -> different tensor shapes inside the checkpoint ->
``CheckpointError: recomputed values have different metadata``. This blocks memory-efficient full-mode
AC (+ activation offload) at ``sp>=4``.

The fix mirrors production's train/inference router replay: capture each router's ``topk`` decision on the
original forward and replay it (as ``routed_experts``, a gather) on the recompute, so routing -- and every
tensor shape in the block -- is identical across both passes. Only the gathered gate weights differ and they
recompute normally with gradients; the checkpoint machinery checks metadata only, so this suffices.

Call ``install_self_router_replay(block)`` on each block before ``checkpoint_wrapper``. Overhead is one
detached int64 ``[tokens, top_k]`` index tensor per MoE layer, held only between a microbatch's paired forward
and backward.
"""

from __future__ import annotations

import collections
import inspect
import threading
from functools import wraps

import torch
import torch.utils.checkpoint as _ckpt

_thread_local = threading.local()


def _in_recompute() -> bool:
    return getattr(_thread_local, "depth", 0) > 0


def _install_recompute_detector() -> None:
    """Wrap ``torch.utils.checkpoint._recomputation_hook`` so a module forward can tell whether it is running
    the original pass or the backward recompute. The non-reentrant checkpoint resolves that name as a module
    global at call time, so patching it here is picked up by all subsequent checkpoint calls."""
    if getattr(_ckpt, "_recompute_detector_installed", False):
        return

    _BaseRecomputationHook = _ckpt._recomputation_hook

    class _DetectingRecomputationHook(_BaseRecomputationHook):  # type: ignore[misc, valid-type]
        def __enter__(self):
            _thread_local.depth = getattr(_thread_local, "depth", 0) + 1
            return super().__enter__()

        def __exit__(self, *exc):
            try:
                return super().__exit__(*exc)
            finally:
                _thread_local.depth = max(0, getattr(_thread_local, "depth", 1) - 1)

    _ckpt._recomputation_hook = _DetectingRecomputationHook
    _ckpt._recompute_detector_installed = True


def _router_supports_routed_experts(router: object) -> bool:
    # Replayable iff forward accepts routed_experts (TokenChoiceTopKRouter does; NemotronHRouter does not).
    forward_method = getattr(router, "forward", None)
    if forward_method is None:
        return False
    try:
        return "routed_experts" in inspect.signature(forward_method).parameters
    except (TypeError, ValueError):
        return False


def _wrap_router(router) -> bool:
    # ``_self_replay_wrapped`` (set below) is also read by trainers to verify that a checkpointed router is
    # protected before they enable fresh routing on it; keep the name stable.
    if getattr(router, "_self_replay_wrapped", False):
        return False
    if not _router_supports_routed_experts(router):
        return False

    original = router.forward
    router._self_replay_queue = collections.deque()
    # Contract read by forward below: a router that never opts into fresh routing (only
    # TokenChoiceTopKRouter.enable_fresh_replay_routing does) keeps gather-only replay, so production replay
    # defers to it.
    if not hasattr(router, "fresh_replay_routing"):
        router.fresh_replay_routing = False

    @wraps(original)
    def forward(x, expert_bias=None, routed_experts=None):
        replay_queue = router._self_replay_queue

        # Production replay (sampler-sourced routing) already feeds routed_experts to both passes; defer to it.
        if routed_experts is not None and not router.fresh_replay_routing:
            return original(x, expert_bias, routed_experts=routed_experts)

        # An opted-in router picks experts by topk for tokens marked ROUTER_REPLAY_FRESH, so the experts the
        # original forward actually selected are captured and replayed verbatim on recompute. The router's
        # fresh-routing path saves the same tensors whichever experts it is handed, so no extra off-graph pass
        # is needed.
        if routed_experts is not None:
            if _in_recompute():
                if len(replay_queue) == 0:
                    # Falling back to routed_experts would re-topk the fresh tokens and diverge from the forward.
                    raise RuntimeError(
                        "router-replay recompute found no captured experts for a fresh-routing replay forward"
                    )
                return original(x, expert_bias, routed_experts=replay_queue.popleft())
            out = original(x, expert_bias, routed_experts=routed_experts)
            if torch.is_grad_enabled():
                # Each graph-building forward is recomputed before this router's next one, so anything still
                # queued belongs to a forward whose backward never ran (an aborted step); replaying it would hand
                # this forward's recompute another step's experts.
                replay_queue.clear()
                replay_queue.append(out[1].detach())
            return out

        # Forward and recompute must save the same tensors, so both passes take the gather path
        # (routed_experts=expert_indices). The discrete topk decision is made once off the autograd graph
        # (no_grad, no saved tensors) on the original forward and replayed verbatim on recompute; the gathered
        # weights are numerically identical to the model's expert_bias topk path.
        if _in_recompute():
            expert_indices = replay_queue.popleft() if replay_queue else None
            if expert_indices is None:
                return original(x, expert_bias)  # should not happen: each captured forward recomputes once
            return original(x, expert_bias, routed_experts=expert_indices)

        with torch.no_grad():
            _, expert_indices, _ = original(x, expert_bias)
        expert_indices = expert_indices.detach()
        if torch.is_grad_enabled():
            # Only capture graph-building (checkpointed) forwards; eval/no_grad forwards are never recomputed.
            replay_queue.append(expert_indices)
        return original(x, expert_bias, routed_experts=expert_indices)

    router.forward = forward
    router._self_replay_wrapped = True
    return True


def install_self_router_replay(module: torch.nn.Module) -> int:
    """Install capture/replay on every replayable MoE router under ``module``; return the number wrapped.

    Call before wrapping the block with ``checkpoint_wrapper`` so only checkpointed blocks capture, keeping
    each router's FIFO drained by its matching recompute.
    """
    _install_recompute_detector()
    num_wrapped = 0
    for submodule in module.modules():
        router = getattr(submodule, "router", None)
        if router is not None and _wrap_router(router):
            num_wrapped += 1
    return num_wrapped
