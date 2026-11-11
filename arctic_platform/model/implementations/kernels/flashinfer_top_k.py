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
"""FlashInfer's radix top-k for the sparse-attention indexers, its module built at model load and never in a step.

FlashInfer compiles its top-k module the first time a process on a node calls it: about 10 minutes on an H200 node,
whose JIT cache (``FLASHINFER_WORKSPACE_BASE``) is its own. A rank that compiles inside a training step keeps the
ranks of every warm node waiting in their next collective until their process-group timeout aborts the job (GLM-5.3-
Flash on 4 nodes: the first MoE layer's expert-parallel all-gather, 600 s). So the loaders of the models whose
sparse-attention indexers select with it call ``prepare_flashinfer_top_k()`` while the model loads: every rank builds
or loads the module (FlashInfer's file lock makes it one build per node) and then waits for the others in a CPU barrier
with a long timeout, so a cold node spends its build in job creation and never inside a collective.

``topk_indices`` only uses a module that is ready: prepared, or built ahead of time (the JIT cache or FlashInfer's AOT
kernels). It never builds one, and it raises when FlashInfer is installed but its module is not ready. A failed build
at load raises too. Without FlashInfer, or for CPU tensors, it selects with ``torch.topk``.
"""

from __future__ import annotations

import functools
import logging
import socket
import time
from datetime import timedelta

import torch

# How long the ranks of warm nodes wait for a cold node's build at load (a cold build takes ~10 minutes).
PREPARE_BARRIER_TIMEOUT = timedelta(minutes=30)
_PREBUILD_HINT = (
    "build it on every node ahead of the job: one flashinfer.top_k call on a GPU under the job's "
    "FLASHINFER_WORKSPACE_BASE (fi_prebuild.sh), or install FlashInfer's AOT kernels"
)

logger = logging.getLogger(__name__)
_ready = False


@functools.cache
def _flashinfer():
    """``(flashinfer.top_k, TopKTieBreak, gen_topk_module)``, or None without FlashInfer."""
    try:
        from flashinfer import top_k
        from flashinfer.jit.topk import gen_topk_module
        from flashinfer.topk import TopKTieBreak
    except ImportError:
        return None
    return top_k, TopKTieBreak, gen_topk_module


def prepare_flashinfer_top_k() -> None:
    """Build or load FlashInfer's top-k module on this rank, then wait for every rank to have it.

    Call it on every rank at the same point of model loading, as the indexer models' loaders do. A no-op without
    FlashInfer or CUDA. Raises, on the rank whose build failed, instead of leaving it to a step.
    """
    global _ready
    flashinfer = _flashinfer()
    if flashinfer is None or not torch.cuda.is_available():
        return
    start = time.monotonic()
    if not _ready:
        from flashinfer.topk import get_topk_module

        try:
            get_topk_module()
        except Exception as exc:
            raise RuntimeError(
                "FlashInfer's top-k module (the sparse-attention indexer's key selection) failed to build on "
                f"{socket.gethostname()}: {exc}. Fix the build, or {_PREBUILD_HINT}"
            ) from exc
        _ready = True
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        # The warm ranks wait for a cold node's build here, on the CPU, in no NCCL collective.
        group = dist.new_group(backend="gloo", timeout=PREPARE_BARRIER_TIMEOUT)
        built = time.monotonic() - start
        dist.barrier(group=group)
        dist.destroy_process_group(group)
        logger.info(
            "FlashInfer top-k module ready on %s in %.1f s; all ranks after %.1f s",
            socket.gethostname(),
            built,
            time.monotonic() - start,
        )


def _ready_top_k():
    """FlashInfer's top_k when its module is ready on this rank, None without FlashInfer; never builds it."""
    global _ready
    flashinfer = _flashinfer()
    if flashinfer is None:
        return None
    top_k, _, gen_topk_module = flashinfer
    if not _ready:
        spec = gen_topk_module()
        if not (spec.is_aot or spec.is_compiled):
            raise RuntimeError(
                f"FlashInfer's top-k module is not built on {socket.gethostname()}, and building it here, inside a "
                "step, would keep the other nodes' ranks waiting in their next collective for its ~10 minutes. The "
                "model's loader builds it (prepare_flashinfer_top_k); call that while loading, or "
                f"{_PREBUILD_HINT}"
            )
        _ready = True  # built ahead of time: FlashInfer loads it without compiling
    return top_k


def topk_indices(scores: torch.Tensor, k: int) -> torch.Tensor:
    """Column indices of each row's ``k`` largest scores, in no particular order.

    FlashInfer's radix select is about 4x faster than ``torch.topk`` on 128K-wide rows. Ties at the selection boundary
    go to the smaller column, so the selected set is deterministic.
    """
    top_k = _ready_top_k() if scores.is_cuda else None
    if top_k is None:
        return torch.topk(scores, k, dim=-1).indices
    tie_break = _flashinfer()[1]
    return top_k(scores, k, sorted=False, deterministic=True, tie_break=tie_break.SMALL)[1]
