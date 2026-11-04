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
detached int64 ``[tokens, top_k]`` index tensor per router call, held for the lifetime of its checkpoint graph.
"""

from __future__ import annotations

import inspect
import threading
import weakref
from functools import wraps

import torch
import torch.utils.checkpoint as _ckpt

_thread_local = threading.local()


def _in_recompute() -> bool:
    thread_local = getattr(_ckpt, "_recompute_thread_local", _thread_local)
    return getattr(thread_local, "depth", 0) > 0


def _checkpoint_context():
    contexts = _checkpoint_contexts()
    return contexts[-1] if contexts else None


def _checkpoint_contexts():
    thread_local = getattr(_ckpt, "_recompute_thread_local", _thread_local)
    return getattr(thread_local, "checkpoint_contexts", ())


def _install_recompute_detector() -> None:
    """Track the non-reentrant checkpoint frame and recomputation scope through its saved-tensor hooks.

    The hook class closes over a thread_local captured here (not the module global) and that same object is
    stashed on _ckpt, so a second import of this module (e.g. resolved under a different module path) finds
    the canonical thread_local via _ckpt instead of racing to install its own detector against a different
    thread_local that the already-installed hook never touches."""
    if getattr(_ckpt, "_recompute_detector_installed", False):
        return

    thread_local = _thread_local
    _BaseCheckpointHook = _ckpt._checkpoint_hook
    _BaseRecomputationHook = _ckpt._recomputation_hook

    class _DetectingCheckpointHook(_BaseCheckpointHook):  # type: ignore[misc, valid-type]
        def __init__(self, frame):
            super().__init__(frame)
            self._replay_frame_ref = weakref.ref(frame)

        def __enter__(self):
            result = super().__enter__()
            if not hasattr(thread_local, "checkpoint_contexts"):
                thread_local.checkpoint_contexts = []
            thread_local.checkpoint_contexts.append((self._replay_frame_ref(), False, {}))
            return result

        def __exit__(self, *exc):
            try:
                return super().__exit__(*exc)
            finally:
                thread_local.checkpoint_contexts.pop()

    class _DetectingRecomputationHook(_BaseRecomputationHook):  # type: ignore[misc, valid-type]
        def __init__(self, target_frame_ref, gid):
            super().__init__(target_frame_ref, gid)
            self._replay_frame_ref = target_frame_ref

        def __enter__(self):
            result = super().__enter__()
            if not hasattr(thread_local, "checkpoint_contexts"):
                thread_local.checkpoint_contexts = []
            thread_local.checkpoint_contexts.append((self._replay_frame_ref(), True, {}))
            thread_local.depth = getattr(thread_local, "depth", 0) + 1
            return result

        def __exit__(self, *exc):
            try:
                return super().__exit__(*exc)
            finally:
                thread_local.checkpoint_contexts.pop()
                thread_local.depth = max(0, getattr(thread_local, "depth", 1) - 1)

    _ckpt._checkpoint_hook = _DetectingCheckpointHook
    _ckpt._recomputation_hook = _DetectingRecomputationHook
    _ckpt._recompute_thread_local = thread_local
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
    router._self_replay_captures = weakref.WeakKeyDictionary()
    # Contract read by forward below: a router that never opts into fresh routing (only
    # TokenChoiceTopKRouter.enable_fresh_replay_routing does) keeps gather-only replay, so production replay
    # defers to it.
    if not hasattr(router, "fresh_replay_routing"):
        router.fresh_replay_routing = False

    @wraps(original)
    def forward(x, expert_bias=None, routed_experts=None):
        # Production replay (sampler-sourced routing) already feeds routed_experts to both passes; defer to it.
        if routed_experts is not None and not router.fresh_replay_routing:
            return original(x, expert_bias, routed_experts=routed_experts)

        contexts = _checkpoint_contexts()
        recompute_context = next((context for context in reversed(contexts) if context[1]), None)
        if recompute_context is not None:
            frame, _, positions = recompute_context
            position = positions.get(router, 0)
            captures = router._self_replay_captures.get(frame, ())
            if position >= len(captures):
                raise RuntimeError("router-replay recompute found no captured experts for its checkpoint forward")
            positions[router] = position + 1
            expert_indices = captures[position]
            out = original(x, expert_bias, routed_experts=expert_indices)
        elif routed_experts is not None:
            # Fresh tokens may pick new experts only on the original forward.
            out = original(x, expert_bias, routed_experts=routed_experts)
            expert_indices = out[1].detach()
        else:
            # Both passes gather so their saved tensors agree; the discrete topk runs off the autograd graph.
            with torch.no_grad():
                _, expert_indices, _ = original(x, expert_bias)
            expert_indices = expert_indices.detach()
            out = original(x, expert_bias, routed_experts=expert_indices)
        if torch.is_grad_enabled():
            # Outer frames capture nested calls too. A new inner forward during outer recompute inherits
            # the replayed experts and records them for its own later backward.
            for frame, recompute, _ in contexts:
                if not recompute:
                    router._self_replay_captures.setdefault(frame, []).append(expert_indices)
        return out

    router.forward = forward
    router._self_replay_wrapped = True
    return True


def install_self_router_replay(module: torch.nn.Module) -> int:
    """Install capture/replay on every replayable MoE router under ``module``; return the number wrapped.

    Call before wrapping the block with ``checkpoint_wrapper``. Captures belong to each checkpoint frame, so
    pending forwards and retained graphs cannot consume another forward's routing decisions.
    """
    _install_recompute_detector()
    num_wrapped = 0
    for submodule in module.modules():
        router = getattr(submodule, "router", None)
        if router is not None and _wrap_router(router):
            num_wrapped += 1
    return num_wrapped
