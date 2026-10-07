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

"""One request split into unequal rows across DP=2 and SP=2."""

from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from arctic_platform.rl.processors.grpo import grpo_loss
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import torch_assert_close


def _loss(advantages, delta, mask, boundaries, weights, mode, dp_size):
    values = torch.full_like(advantages, -2.0, requires_grad=True)
    context = dict(
        old_log_probs_shifted=values.detach() - delta, advantages=advantages, loss_mask=mask, cu_seqlens=boundaries
    )
    if mode == "prompt-mean":
        context["sequence_loss_weights"] = weights
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
            loss_agg_mode=mode,
            batch_num_tokens=6,
            dp_size=dp_size,
        ),
        "cpu",
    )
    loss.backward()
    return values.grad, metrics


def _worker(rank, init_file):
    torch.set_num_threads(1)
    advantages = torch.tensor([1.0, 3.0, -2.0, -2.0, 2.0, -2.0, 0.0, 0.0])
    delta = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    mask = torch.arange(8) < 6
    weights = torch.tensor([0.25, 0.75])
    references = {
        mode: _loss(advantages, delta, mask, torch.tensor([0, 4, 8]), weights, mode, 1)
        for mode in ("token-mean", "prompt-mean")
    }
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", world_size=4, rank=rank, timeout=timedelta(seconds=45)
    )
    try:
        groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
        window = slice(rank * 2, rank * 2 + 2)
        for mode, (reference_grad, reference_metrics) in references.items():
            with (
                patch(
                    "arctic_platform.rl.processors.grpo._get_sequence_parallel_group", return_value=groups[rank // 2]
                ),
                patch(
                    "arctic_platform.rl.processors.functional._get_sequence_parallel_group",
                    return_value=groups[rank // 2],
                ),
            ):
                grad, metrics = _loss(
                    advantages[window],
                    delta[window],
                    mask[window],
                    torch.tensor([0, 2]),
                    weights[rank // 2 : rank // 2 + 1],
                    mode,
                    2,
                )
            # DeepSpeed averages DP gradients; SP shards contain distinct tokens.
            torch_assert_close(grad / 2, reference_grad[window], rtol=1e-6, atol=1e-7)
            assert torch.isfinite(grad).all()
            keys = [f"ratio_mask_{sign}_{stage}_mass_sum" for sign in ("pos", "neg") for stage in ("pre", "kept")]
            if rank % 2:
                assert all(metrics[key] == 0 for key in keys)
            totals = torch.tensor([metrics[key] for key in keys], dtype=torch.float64)
            dist.all_reduce(totals)
            torch_assert_close(
                totals,
                torch.tensor([reference_metrics[key] for key in keys], dtype=torch.float64),
                rtol=1e-6,
                atol=1e-7,
            )
    finally:
        dist.destroy_process_group()


class TestRatioMassDistributed(TestCasePlus):
    def test_packed_dp_sp_matches_unsplit_request(self):
        mp.spawn(_worker, args=(self.get_auto_remove_tmp_dir_str() + "/group",), nprocs=4)
