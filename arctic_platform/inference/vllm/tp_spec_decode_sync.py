# Copyright 2026 Snowflake Inc.
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

"""Keep tensor-parallel ranks in step during speculative decoding (Model Runner V2).

vLLM 0.31.0's Model Runner V2 drafts, rejection-samples and advances each
request's device-side state on every TP rank from that rank's own results.
Nothing re-synchronizes the ranks, so they agree only if every rank's drafter
computes bit-identical draft probabilities. They need not: each rank compiles
the drafter itself, and Inductor picks some reduction kernels' configs per
process by timing (for DFlash2 on Qwen3.8-27B, an R0_BLOCK of 2048 or 4096 for
a 5120-wide norm, within about 1% of each other). A rank whose pick sums in a
different order eventually accepts a different number of draft tokens than
rank 0, falls behind it (the scheduler follows rank 0), and writes its GDN
state into blocks other requests now own. Attention reads those bytes as K/V
and the all-reduce spreads NaN logprobs to every request.

TP rank 0's result is made authoritative, as vLLM already does across PP
ranks: after each draft step its draft tokens, and after each sample its
sampled tokens and sampled/rejected counts, are broadcast to the other TP
ranks before anything reads them. It applies only when speculative decoding
runs on more than one TP rank with every rank sampling the whole batch.
``ARCTIC_TP_SPEC_SYNC=0`` disables it.

With ``ARCTIC_TP_SPEC_SYNC_STATS=1``, a rank also copies its draft and sampled
tokens before each broadcast and, every ``STATS_EVERY`` sample steps, logs the
steps where rank 0's copy (in the slots that are read) differed from its own.
Off by default, so the default path is only the broadcasts.

Remove once the pinned vLLM synchronizes TP ranks during speculative decoding.

See https://github.com/Snowflake-AI-Research/DSS-Issue-Tracker (issue 15f8b977).
"""

from __future__ import annotations

import os
from typing import Any

import torch

_APPLIED = False
STATS_EVERY = 5000
STATS_ENABLED = os.getenv("ARCTIC_TP_SPEC_SYNC_STATS", "0") == "1"


def _tp_group():
    from vllm.distributed import get_tp_group

    return get_tp_group()


def broadcast_from_rank0(tensors) -> None:
    """Overwrite each tensor in place with TP rank 0's copy.

    Every rank must call this at the same point with the same structure.
    """
    tensors = [t for t in tensors if isinstance(t, torch.Tensor)]
    tp = _tp_group()
    if not tensors or tp.world_size == 1:
        return
    for t in tensors:
        if t.is_contiguous():
            tp.broadcast(t, src=0)
        else:
            c = t.contiguous()
            tp.broadcast(c, src=0)
            t.copy_(c)


def needs_sync(runner: Any) -> bool:
    """Speculative decoding on more than one TP rank, every rank sampling the whole batch."""
    return (
        getattr(runner, "speculator", None) is not None
        and getattr(runner, "batch_sharder", None) is None
        and _tp_group().world_size > 1
    )


def drafts_differ(before: torch.Tensor, after: torch.Tensor) -> torch.Tensor:
    return (before != after).any()


def samples_differ(
    before_tokens: torch.Tensor,
    before_counts: torch.Tensor,
    after_tokens: torch.Tensor,
    after_counts: torch.Tensor,
) -> torch.Tensor:
    """Whether the sampled counts, or any token rank 0's counts make readable, differed."""
    k = torch.arange(after_tokens.shape[-1], device=after_tokens.device)
    read = k[None, :] < after_counts.reshape(-1, 1).long()
    return (before_counts != after_counts).any() | (
        (before_tokens != after_tokens) & read
    ).any()


class SyncStats:
    """Steps where rank 0's copy differed from this rank's own, counted on device."""

    def __init__(self) -> None:
        self.steps = {"draft": 0, "sample": 0}
        self.corrected: dict[str, torch.Tensor] = {}

    def add(self, kind: str, differed: torch.Tensor) -> None:
        self.steps[kind] += 1
        if kind in self.corrected:
            self.corrected[kind] += differed
        else:
            self.corrected[kind] = differed.long().clone()

    def maybe_log(self) -> None:
        if not STATS_EVERY or self.steps["sample"] % STATS_EVERY:
            return
        corrected = {k: int(v.item()) for k, v in self.corrected.items()}
        if any(corrected.values()):
            from vllm.logger import init_logger

            init_logger(__name__).warning(
                "TP rank %s differed from rank 0 and was resynced on %s of %s "
                "steps (draft, sample)",
                _tp_group().rank_in_group,
                (corrected.get("draft", 0), corrected.get("sample", 0)),
                (self.steps["draft"], self.steps["sample"]),
            )


def _stats(runner: Any) -> SyncStats:
    stats = getattr(runner, "_arctic_tp_sync_stats", None)
    if stats is None:
        stats = runner._arctic_tp_sync_stats = SyncStats()
    return stats


def wrap_propose(runner: Any) -> None:
    """Broadcast rank 0's draft tokens after each draft step of ``runner.speculator``."""
    spec = getattr(runner, "speculator", None)
    if spec is None or getattr(spec, "_arctic_tp_sync", False):
        return
    original_propose = spec.propose

    def propose(*args, **kwargs):
        draft_tokens = original_propose(*args, **kwargs)
        if isinstance(draft_tokens, torch.Tensor) and needs_sync(runner):
            before = draft_tokens.clone() if STATS_ENABLED else None
            broadcast_from_rank0([draft_tokens])
            if before is not None:
                _stats(runner).add("draft", drafts_differ(before, draft_tokens))
        return draft_tokens

    spec.propose = propose
    spec._arctic_tp_sync = True


def sync_sampler_output(runner: Any, sampler_output: Any) -> None:
    before = (
        (sampler_output.sampled_token_ids.clone(), sampler_output.num_sampled.clone())
        if STATS_ENABLED
        else None
    )
    broadcast_from_rank0(
        [
            sampler_output.sampled_token_ids,
            sampler_output.num_sampled,
            sampler_output.num_rejected,
        ]
    )
    if before is not None:
        stats = _stats(runner)
        stats.add(
            "sample",
            samples_differ(
                *before, sampler_output.sampled_token_ids, sampler_output.num_sampled
            ),
        )
        stats.maybe_log()


def patch_runner(runner_cls: Any) -> None:
    if not getattr(runner_cls.__init__, "_arctic_tp_sync", False):
        original_init = runner_cls.__init__

        def __init__(self, *args, **kwargs):
            original_init(self, *args, **kwargs)
            wrap_propose(self)

        __init__._arctic_tp_sync = True
        __init__._orig = original_init
        runner_cls.__init__ = __init__

    if not getattr(runner_cls.sample, "_arctic_tp_sync", False):
        original_sample = runner_cls.sample

        def sample(self, hidden_states, input_batch, grammar_output):
            result = original_sample(self, hidden_states, input_batch, grammar_output)
            if needs_sync(self):
                sync_sampler_output(self, result[0])
            return result

        sample._arctic_tp_sync = True
        sample._orig = original_sample
        runner_cls.sample = sample


def ensure_tp_spec_decode_sync() -> None:
    """Patch Model Runner V2 so TP rank 0's draft and sampled tokens are authoritative."""
    global _APPLIED
    if _APPLIED or os.getenv("ARCTIC_TP_SPEC_SYNC", "1") != "1":
        return

    from arctic_platform.inference.utils import require_supported_vllm_version

    require_supported_vllm_version("TP sync for speculative decoding")

    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    patch_runner(GPUModelRunner)
    _APPLIED = True
