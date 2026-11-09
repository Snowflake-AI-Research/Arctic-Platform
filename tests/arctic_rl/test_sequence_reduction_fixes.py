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

"""Sequence means and sequence-level IS agree between the packed and padded layouts, with and without SP."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from deepspeed.utils import groups

from arctic_platform.rl.processors.grpo import grpo_loss
from arctic_platform.rl.processors.packed_reduction import resolve_packed_loss_reduction
from arctic_platform.rl.processors.pipeline import run_pipeline

# Two rows of lengths 1 and 3 with different per-token advantages, so per-row and per-pack means differ.
LENGTHS = (1, 3)
OLD = torch.tensor([-1.0, -1.2, -0.8, -1.1])
LOGPROBS = torch.tensor([-0.9, -1.0, -0.9, -1.0])
ADVANTAGES = torch.tensor([1.0, -1.0, -2.0, -3.0])


def _run(layout, config, *, loss_mask=None, advantages=ADVANTAGES):
    """One grpo call on the rows above, packed ([T] + cu_seqlens) or padded ([B, S])."""
    mask = torch.ones(4, dtype=torch.bool) if loss_mask is None else loss_mask
    if layout == "packed":
        leaf = LOGPROBS.clone().requires_grad_(True)
        model = dict(logprobs=leaf)
        context = dict(
            old_log_probs_shifted=OLD,
            advantages=advantages,
            loss_mask=mask,
            cu_seqlens=torch.tensor([0, 1, 4], dtype=torch.int32),
        )
    else:
        index = torch.tensor([[0, -1, -1], [1, 2, 3]])
        valid = index >= 0

        def pad(t, fill):
            gathered = t[index.clamp(min=0)]
            return torch.where(valid, gathered, torch.full_like(gathered, fill))

        leaf = LOGPROBS.clone().requires_grad_(True)
        model = dict(logprobs=pad(leaf, 0.0))
        context = dict(
            old_log_probs_shifted=pad(OLD, 0.0), advantages=pad(advantages, 9.0), loss_mask=pad(mask, False) & valid
        )
    loss, _ = grpo_loss(model, context, {}, dict(use_cispo_loss=True, is_weight_clip_max=5.0, **config), "cpu")
    (grad,) = torch.autograd.grad(loss, leaf)
    return loss.detach(), grad


@pytest.mark.parametrize("mode", ["seq-mean-token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm"])
def test_packed_seq_mean_matches_padded(mode):
    # The padded width equals the longest sequence, so the norm mode's scale is the same in both layouts.
    padded = _run("padded", dict(loss_agg_mode=mode))
    packed = _run("packed", dict(loss_agg_mode=mode))
    torch.testing.assert_close(packed[0], padded[0])
    torch.testing.assert_close(packed[1], padded[1])


def test_padded_sequence_is_advantages_match_packed():
    # The last token is masked out; its advantage must not enter its sequence's mean.
    mask = torch.tensor([True, True, True, False])
    config = dict(importance_sampling_level="sequence", loss_agg_mode="token-mean")
    packed = _run("packed", config, loss_mask=mask)
    padded = _run("padded", config, loss_mask=mask)
    torch.testing.assert_close(padded[0], packed[0])
    torch.testing.assert_close(padded[1], packed[1])


class _Engine:
    global_rank = 0

    def __init__(self):
        self.logprobs = LOGPROBS.clone().requires_grad_(True)

    def __call__(self, input_ids, **kwargs):
        return dict(logprobs=self.logprobs[input_ids])

    def eval(self):
        pass

    def train(self):
        pass


@pytest.mark.parametrize("contract", ["grpo", "grpo_echo_v1"])
@pytest.mark.parametrize("mode", ["seq-mean-token-mean", "seq-mean-token-sum"])
def test_packing_width_does_not_change_the_loss(contract, mode):
    ids = torch.tensor([[0, 0, 0], [1, 2, 3]])
    mask = torch.tensor([[True, False, False], [True, True, True]])
    context = dict(old_log_probs_shifted=OLD[ids], advantages=ADVANTAGES[ids], loss_mask=mask)
    config = dict(
        use_cispo_loss=True,
        is_weight_clip_max=5.0,
        loss_agg_mode=mode,
        importance_sampling_level="sequence",
        global_batch_size=2,
    )
    if contract == "grpo_echo_v1":
        config.update(aux_ce_weight=0.0, echo_global_num_sequences=2)
        context.update(
            sft_mask=torch.zeros_like(mask),
            echo_observation_mask=torch.zeros_like(mask),
            echo_observation_token_counts=torch.zeros(2),
        )
    results = []
    for width in (4, 3):
        engine = _Engine()
        result = run_pipeline(
            engine,
            (),
            dict(input_ids=ids, attention_mask=mask),
            context,
            dict(loss_fn=contract, config=config),
            "cpu",
            backward="loss_only",
            max_tokens_per_mb=width,
            return_tensors=True,
        )
        (grad,) = torch.autograd.grad(result["loss_tensor"], engine.logprobs)
        results.append((result["loss_tensor"].detach(), grad))
    torch.testing.assert_close(results[0], results[1])


# --- sequence parallelism: the packed seq-mean is a whole-sequence mean across windows ------------------

SP_CU_SEQLENS = torch.tensor([0, 5, 12, 16], dtype=torch.int32)
SP_TOKENS = 16


def _sp_frame():
    positions = torch.arange(SP_TOKENS, dtype=torch.float32)
    old = -1.0 - 0.01 * positions
    logprobs = old + 0.02 * torch.sin(positions)
    advantages = torch.where(positions.long() % 3 == 0, 1.0, -0.5)
    mask = torch.ones(SP_TOKENS, dtype=torch.bool)
    mask[[0, 5, 12]] = False
    return logprobs, old, advantages, mask


def _sp_loss(logprobs, old, advantages, masks, cu_seqlens):
    contexts = [
        dict(
            input_ids=torch.zeros_like(mask, dtype=torch.long),
            old_log_probs_shifted=old,
            advantages=advantages,
            loss_mask=mask,
            cu_seqlens=cu_seqlens,
        )
        for mask in masks
    ]
    config = dict(
        use_cispo_loss=True,
        is_weight_clip_max=5.0,
        loss_agg_mode="seq-mean-token-mean",
        dp_size=1,
    )
    reduction = resolve_packed_loss_reduction(dict(loss_fn="grpo", config=config), contexts)
    return sum(
        grpo_loss(dict(logprobs=logprobs), context, {}, config, "cpu")[0] * scale
        for context, scale in zip(contexts, reduction.loss_scales)
    )


def _sp_worker(rank, world_size, init_method, case):
    torch.set_num_threads(1)
    logprobs, old, advantages, mask = _sp_frame()
    masks = [mask]
    if case != "full":
        first_window = torch.arange(SP_TOKENS) < SP_TOKENS // world_size
        masks = [mask & first_window]
        if case == "microbatches":
            masks.append(mask & ~first_window)
    whole = logprobs.clone().requires_grad_(True)
    expected = _sp_loss(whole, old, advantages, masks, SP_CU_SEQLENS)
    (expected_grad,) = torch.autograd.grad(expected, whole)
    dist.init_process_group(
        "gloo", init_method=init_method, rank=rank, world_size=world_size, timeout=timedelta(seconds=15)
    )
    try:
        width = SP_TOKENS // world_size
        start, end = rank * width, (rank + 1) * width
        window = (SP_CU_SEQLENS.clamp(min=start, max=end) - start).to(torch.int32)
        window[-1] = end - start
        with (
            patch.object(groups, "_get_sequence_parallel_world_size", return_value=world_size),
            patch.object(groups, "_get_sequence_parallel_group", return_value=dist.group.WORLD),
        ):
            local = logprobs[start:end].clone().requires_grad_(True)
            loss = _sp_loss(local, old[start:end], advantages[start:end], [mask[start:end] for mask in masks], window)
            (grad,) = torch.autograd.grad(loss, local)
        total = loss.detach().clone()
        dist.all_reduce(total)
        torch.testing.assert_close(total, expected.detach())
        torch.testing.assert_close(grad, expected_grad[start:end])
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("case", ["full", "empty", "microbatches"])
def test_packed_seq_mean_under_sequence_parallelism(tmp_path, case):
    mp.spawn(_sp_worker, args=(2, (tmp_path / "gloo").as_uri(), case), nprocs=2, join=True)
