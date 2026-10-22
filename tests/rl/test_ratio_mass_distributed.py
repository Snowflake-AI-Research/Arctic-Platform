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

"""One request split into unequal rows across DP=2 and SP=2 (four Gloo ranks), against a single-rank reference.

Row 0 (advantages +1 -3 +2.5 -2) goes to DP rank 0 and row 1 (+2 +1, then two non-policy tokens) to DP rank 1; each
row is cut into two token windows, one per SP rank, so rank 3 holds only non-policy tokens. Every loss aggregation mode
and both importance-sampling levels run on the real ``grpo_loss`` with the step-global denominators of the whole request;
one grouped prompt-mean variant puts both rows in a single prompt, cut across the DP ranks.
"""

import math
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from arctic_platform.rl.processors.grpo import grpo_loss
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import torch_assert_close

MASS_KEYS = [f"ratio_mask_{sign}_{stage}_mass_sum" for sign in ("pos", "neg") for stage in ("pre", "kept")]
K3_KEY = "ratio_mask_kept_k3_sum"
# mode -> config and context carrying the denominators of the whole two-row request.
MODES = {
    "token-mean": (dict(batch_num_tokens=6), {}),
    "seq-mean-token-sum": (dict(global_batch_size=2), {}),
    "seq-mean-token-sum-norm": (dict(global_batch_size=2), dict(packed_loss_scale_factor=5)),
    "seq-mean-token-mean": (dict(global_batch_size=2), {}),
    "prompt-mean weighted": ({}, {}),
    "prompt-mean grouped": (dict(global_batch_size=2), {}),
    # Both rows belong to one prompt, so it is cut across the two DP ranks; each row carries the prompt's global token
    # count (4 + 2 policy tokens), as DSS supplies it.
    "prompt-mean grouped, cut prompt": (dict(global_batch_size=1), {}),
}
LEVELS = ("token", "sequence")
ROW_CONTEXT = {
    "prompt-mean weighted": dict(sequence_loss_weights=torch.tensor([0.25, 0.75])),
    "prompt-mean grouped": dict(prompt_group_ids=torch.tensor([0, 1])),
    "prompt-mean grouped, cut prompt": dict(
        prompt_group_ids=torch.tensor([0, 0]), prompt_token_counts=torch.tensor([6.0, 6.0])
    ),
}


def _loss(advantages, delta, mask, boundaries, rows, mode, level, dp_size):
    values = torch.full_like(advantages, -2.0, requires_grad=True)
    config, extra_context = MODES[mode]
    context = dict(
        old_log_probs_shifted=values.detach() - delta,
        advantages=advantages,
        loss_mask=mask,
        cu_seqlens=boundaries,
        **extra_context,
    )
    context.update({key: value[rows] for key, value in ROW_CONTEXT.get(mode, {}).items()})
    loss, metrics = grpo_loss(
        dict(logprobs=values),
        context,
        {},
        dict(
            use_cispo_loss=True,
            is_weight_clip_max=5.0,
            ratio_stats=True,
            ratio_mask_bounds_pos=[0.9, 1.1],
            ratio_mask_bounds_neg=[0.9, 1.1],
            loss_agg_mode=mode.split(" ")[0],
            importance_sampling_level=level,
            dp_size=dp_size,
            **config,
        ),
        "cpu",
    )
    loss.backward()
    return values.grad, metrics


def _request():
    advantages = torch.tensor([1.0, -3.0, 2.5, -2.0, 2.0, 1.0, 0.0, 0.0])
    # The +-0.05 tokens stay inside the [0.9, 1.1] band, so they are kept with a nonzero k3, one of them on an SP non-leader.
    delta = torch.tensor([0.0, 1.0, 0.05, 0.0, 1.0, -0.05, 0.0, 0.0])
    return advantages, delta, torch.arange(8) < 6


def _sequence_parallel_group(group):
    return (
        patch("arctic_platform.rl.processors.grpo._get_sequence_parallel_group", return_value=group),
        patch("arctic_platform.rl.processors.functional._get_sequence_parallel_group", return_value=group),
    )


def _worker(rank, init_file):
    torch.set_num_threads(1)
    advantages, delta, mask = _request()
    references = {
        (mode, level): _loss(advantages, delta, mask, torch.tensor([0, 4, 8]), [0, 1], mode, level, 1)
        for mode in MODES
        for level in LEVELS
    }
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", world_size=4, rank=rank, timeout=timedelta(seconds=45)
    )
    try:
        groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
        window, row = slice(rank * 2, rank * 2 + 2), [rank // 2]
        for (mode, level), (reference_grad, reference_metrics) in references.items():
            patch_grpo, patch_functional = _sequence_parallel_group(groups[rank // 2])
            with patch_grpo, patch_functional:
                grad, metrics = _loss(
                    advantages[window], delta[window], mask[window], torch.tensor([0, 2]), row, mode, level, 2
                )
            # DeepSpeed averages DP gradients; SP shards contain distinct tokens. A gradient is a few fp32 products
            # and sums; 1e-7 is ~2 ulp at |grad| ~ 0.5 for the different reduction order of the split run, while a
            # lost or doubled token moves a gradient by >= 1e-2.
            torch_assert_close(grad / 2, reference_grad[window], rtol=0, atol=1e-7, msg=f"{mode} {level} grad")
            if rank % 2:
                # Only the SP leader reports the group's total, although this rank holds a kept token with k3 > 0.
                assert all(metrics[key] == 0 for key in [*MASS_KEYS, K3_KEY]), (mode, level, metrics)
            totals = torch.tensor([metrics[key] for key in [*MASS_KEYS, K3_KEY]], dtype=torch.float64)
            dist.all_reduce(totals)
            # The masses are float64 sums of a few terms: the split run differs from the reference only by summation
            # order (~1e-16 relative), while a misplaced token or a missed DP / SP term changes a mass by >= 1e-2.
            reference = torch.tensor([reference_metrics[key] for key in [*MASS_KEYS, K3_KEY]], dtype=torch.float64)
            torch_assert_close(totals[:-1], reference[:-1], rtol=0, atol=1e-12, msg=f"{mode} {level} mass")
            # k3 is summed in fp32 per rank (at most 6 terms, the largest k3(1) = 0.72): the split run differs from the
            # reference by the summation order, <= ~1e-6, while a missing or doubled token moves it by >= 1e-3.
            torch_assert_close(totals[-1], reference[-1], rtol=0, atol=1e-6, msg=f"{mode} {level} k3")
            # Independently: only the two +-0.05 tokens are kept with a nonzero k3 (the +1 tokens fall outside the band
            # and the others have a log-ratio of 0), whatever the aggregation mode and importance-sampling level.
            kept_k3 = (math.expm1(0.05) - 0.05) + (math.expm1(-0.05) + 0.05)
            torch_assert_close(totals[-1].item(), kept_k3, rtol=0, atol=1e-6, msg=f"{mode} {level} k3 closed form")
    finally:
        dist.destroy_process_group()


def _zero_denominator_worker(rank, init_file):
    torch.set_num_threads(1)
    advantages, delta, mask = _request()
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", world_size=4, rank=rank, timeout=timedelta(seconds=45)
    )
    try:
        groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
        window = slice(rank * 2, rank * 2 + 2)
        for mode, denominator in (("token-mean", "batch_num_tokens"), ("seq-mean-token-sum", "global_batch_size")):
            for controls in (dict(), dict(ratio_stats=True, ratio_mask_bounds_pos=[0.9, 1.1])):
                values = torch.full_like(advantages[window], -2.0, requires_grad=True)
                context = dict(
                    old_log_probs_shifted=values.detach() - delta[window],
                    advantages=advantages[window],
                    loss_mask=mask[window],
                    cu_seqlens=torch.tensor([0, 2]),
                )
                config = dict(use_cispo_loss=True, is_weight_clip_max=5.0, loss_agg_mode=mode, dp_size=2, **controls)
                config[denominator] = 0
                patch_grpo, patch_functional = _sequence_parallel_group(groups[rank // 2])
                with patch_grpo, patch_functional:
                    try:
                        grpo_loss(dict(logprobs=values), context, {}, config, "cpu")
                    except ValueError as error:
                        # Every rank fails, including rank 3, which holds no policy token itself.
                        assert f"{denominator}=0" in str(error), error
                    else:
                        raise AssertionError(f"rank {rank} accepted {denominator}=0 with policy tokens in the step")
    finally:
        dist.destroy_process_group()


def _grouped_prompt_mean_without_global_batch_size(advantages, delta, prompt_ids, dp_size):
    values = torch.full_like(advantages, -2.0, requires_grad=True)
    context = dict(
        old_log_probs_shifted=values.detach() - delta,
        advantages=advantages,
        loss_mask=torch.ones_like(values, dtype=torch.bool),
        prompt_group_ids=prompt_ids,
    )
    config = dict(
        use_cispo_loss=True,
        is_weight_clip_max=5.0,
        ratio_stats=True,
        ratio_mask_bounds_pos=[0.9, 1.1],
        ratio_mask_bounds_neg=[0.9, 1.1],
        loss_agg_mode="prompt-mean",
        dp_size=dp_size,
    )
    loss, metrics = grpo_loss(dict(logprobs=values), context, {}, config, "cpu")
    loss.backward()
    return loss.detach(), metrics


def _dp_only_worker(rank, init_file):
    torch.set_num_threads(1)
    advantages = torch.tensor(
        [[1.0, 3.0, -2.0, -1.0], [2.0, 2.0, -1.0, -4.0], [4.0, -2.0, 2.0, 0.0], [-3.0, 1.0, -2.0, -1.0]]
    )
    delta = torch.tensor([[0.0, 1.0, -1.0, 0.0], [0.0, -1.0, 0.0, 1.0], [1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])
    patch_grpo, patch_functional = _sequence_parallel_group(None)
    reference_loss, reference = _grouped_prompt_mean_without_global_batch_size(
        advantages[:2], delta[:2], torch.tensor([0, 1]), 1
    )
    _, reference4 = _grouped_prompt_mean_without_global_batch_size(advantages, delta, torch.arange(4), 1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", world_size=2, rank=rank, timeout=timedelta(seconds=45)
    )
    try:
        # There is no sequence parallelism, and DeepSpeed's own groups are not set up in this process.
        with patch_grpo, patch_functional:
            # One prompt per DP worker, one call each, and no global_batch_size: the loss all-reduces the prompt count
            # over the workers, so the masses and the loss add up across workers.
            loss, metrics = _grouped_prompt_mean_without_global_batch_size(
                advantages[rank : rank + 1], delta[rank : rank + 1], torch.tensor([rank]), 2
            )
            totals = _summed_over_workers([metrics])
            torch_assert_close(totals, _mass(reference), rtol=0, atol=1e-12)
            # The loss carries dp_size, so the workers' losses average to the unsplit loss (fp32 sums of 4 terms).
            total_loss = loss.double()
            dist.all_reduce(total_loss)
            torch_assert_close(total_loss / 2, reference_loss.double(), rtol=0, atol=1e-6)

            # Two prompts per worker in one call still add up: the all-reduced count is 2 + 2 = 4.
            rows = slice(2 * rank, 2 * rank + 2)
            metrics = _grouped_prompt_mean_without_global_batch_size(
                advantages[rows], delta[rows], torch.tensor([2 * rank, 2 * rank + 1]), 2
            )[1]
            torch_assert_close(_summed_over_workers([metrics]), _mass(reference4), rtol=0, atol=1e-12)

            # The same two prompts as two calls per worker do not: each call all-reduces only its own prompt count
            # (1 + 1 = 2 instead of the step's 4), so every prompt is normalised by half the step's count and the
            # sum over calls and workers is twice the unsplit value.
            calls = [
                _grouped_prompt_mean_without_global_batch_size(
                    advantages[2 * rank + i : 2 * rank + i + 1],
                    delta[2 * rank + i : 2 * rank + i + 1],
                    torch.tensor([2 * rank + i]),
                    2,
                )[1]
                for i in range(2)
            ]
            torch_assert_close(_summed_over_workers(calls), 2 * _mass(reference4), rtol=0, atol=1e-12)
    finally:
        dist.destroy_process_group()


def _mass(metrics):
    return torch.tensor([metrics[key] for key in MASS_KEYS], dtype=torch.float64)


def _summed_over_workers(calls):
    """The four masses summed over this worker's calls and then over all workers."""
    totals = sum(_mass(metrics) for metrics in calls)
    dist.all_reduce(totals)
    return totals


class TestRatioMassDistributed(TestCasePlus):
    def test_packed_dp_sp_matches_unsplit_request_in_every_mode(self):
        mp.spawn(_worker, args=(self.get_auto_remove_tmp_dir_str() + "/group",), nprocs=4)

    def test_zero_denominator_with_policy_tokens_fails_on_every_rank(self):
        mp.spawn(_zero_denominator_worker, args=(self.get_auto_remove_tmp_dir_str() + "/group",), nprocs=4)

    def test_grouped_prompt_mean_without_global_batch_size_adds_across_dp_workers(self):
        mp.spawn(_dp_only_worker, args=(self.get_auto_remove_tmp_dir_str() + "/group",), nprocs=2)
