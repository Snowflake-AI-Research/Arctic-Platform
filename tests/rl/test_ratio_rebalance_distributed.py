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

"""``ratio_mask_rebalance`` on one request split across DP=2 and SP=2 (four Gloo ranks), against a single-rank run.

The request, its split and the loss modes are those of ``test_ratio_mass_distributed.py`` (row 0 on DP rank 0, row 1 on
DP rank 1, each cut into two SP token windows, the last window empty). Each rank makes exactly one model call, as the
flag requires. The scales come from the masses summed over the whole world, so every rank must end up with the same
scale as the single-rank run.
"""

from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from test_ratio_mass_distributed import LEVELS
from test_ratio_mass_distributed import MASS_KEYS
from test_ratio_mass_distributed import MODES
from test_ratio_mass_distributed import ROW_CONTEXT
from test_ratio_mass_distributed import _request
from test_ratio_mass_distributed import _sequence_parallel_group

from arctic_platform.rl.processors.grpo import grpo_loss
from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import torch_assert_close

FLAG = "ratio_mask_rebalance"
K3_KEY = "ratio_mask_kept_k3_sum"
REBALANCE_KEYS = [
    f"ratio_rebalance_{sign}_{stage}_mass_sum" for sign in ("pos", "neg") for stage in ("post", "unrestored")
]
KEYS = [*MASS_KEYS, *REBALANCE_KEYS, K3_KEY]
# case -> (controls, log-ratio offsets or None for the request's own, teacher term on?, importance-sampling levels)
BAND = dict(ratio_mask_bounds_pos=[0.9, 1.1], ratio_mask_bounds_neg=[0.9, 1.1])
CASES = {
    "drops mass of both signs": (BAND, None, False, LEVELS),
    # Every positive-side token is outside the band (every ratio of the request is <= e).
    "drops every positive-side token": (dict(ratio_mask_bounds_pos=[5.0, 6.0]), None, False, LEVELS),
    "drops nothing": (dict(ratio_mask_bounds_pos=[1e-6, 1e6], ratio_mask_bounds_neg=[1e-6, 1e6]), None, False, LEVELS),
    # Token 0 (A = +1, offset +1) and token 1 (A = -3, offset -1) exceed a 0.05 probability gap on their own side.
    "probability gap": (
        dict(prob_diff_mask_max_pos=0.05, prob_diff_mask_max_neg=0.05),
        torch.tensor([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        False,
        LEVELS,
    ),
    # Row 0 has a mean log ratio of 0.2, above the 0.1 bound, but it is cut across two SP ranks and only the first
    # window sees the offsets (0.4 there, 0 on the second), so the second window drops its tokens only if the statistic
    # is summed over the SP group.
    "sequence gate": (
        dict(seq_mask_bounds_pos=[-0.1, 0.1], seq_mask_bounds_neg=[-0.1, 0.1]),
        torch.tensor([0.4, 0.4, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        False,
        LEVELS,
    ),
    # The teacher is 1 nat above the policy: with tau 2 every advantage gains 2, so the -2 advantages become 0 and the
    # -3 one a -1. It only exists at token level.
    "teacher term with the ratio band": (dict(teacher_tau=2.0, teacher_clip=2.0, **BAND), None, True, ("token",)),
}


def _loss(advantages, delta, mask, boundaries, rows, mode, level, dp_size, controls, teacher=False):
    values = torch.full_like(advantages, -2.0, requires_grad=True)
    mode_config, extra_context = MODES[mode]
    context = dict(
        old_log_probs_shifted=values.detach() - delta,
        advantages=advantages,
        loss_mask=mask,
        cu_seqlens=boundaries,
        **extra_context,
    )
    if teacher:
        context["teacher_log_probs_shifted"] = torch.full_like(advantages, -1.0)
    context.update({key: value[rows] for key, value in ROW_CONTEXT.get(mode, {}).items()})
    config = dict(
        use_cispo_loss=True,
        is_weight_clip_max=5.0,
        loss_agg_mode=mode.split(" ")[0],
        importance_sampling_level=level,
        dp_size=dp_size,
        **mode_config,
    )
    loss, metrics = grpo_loss(dict(logprobs=values), context, {}, {**config, **controls}, "cpu")
    loss.backward()
    return values.grad, metrics


def _worker(rank, init_file):
    torch.set_num_threads(1)
    advantages, request_delta, mask = _request()
    deltas = {case: request_delta if spec[1] is None else spec[1] for case, spec in CASES.items()}
    references = {
        (case, mode, level): _loss(
            advantages,
            deltas[case],
            mask,
            torch.tensor([0, 4, 8]),
            [0, 1],
            mode,
            level,
            1,
            {**controls, FLAG: True},
            teacher,
        )
        for case, (controls, _, teacher, levels) in CASES.items()
        for mode in MODES
        for level in levels
    }
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", world_size=4, rank=rank, timeout=timedelta(seconds=60)
    )
    try:
        groups = [dist.new_group(ranks) for ranks in ([0, 1], [2, 3])]
        window, row = slice(rank * 2, rank * 2 + 2), [rank // 2]
        for (case, mode, level), (reference_grad, reference_metrics) in references.items():
            label = f"{case} / {mode} / {level}"
            case_controls, _, teacher, _ = CASES[case]
            controls, delta = {**case_controls, FLAG: True}, deltas[case]
            patch_grpo, patch_functional = _sequence_parallel_group(groups[rank // 2])

            def run(extra_controls):
                with patch_grpo, patch_functional:
                    return _loss(
                        advantages[window],
                        delta[window],
                        mask[window],
                        torch.tensor([0, 2]),
                        row,
                        mode,
                        level,
                        2,
                        extra_controls,
                        teacher,
                    )

            grad, metrics = run(controls)
            # DeepSpeed averages DP gradients; SP shards contain distinct tokens. A gradient is a few fp32 products
            # and sums; 1e-7 is ~2 ulp at |grad| ~ 0.5 for the different reduction order of the split run, while a
            # lost or doubled token, or a scale computed from one rank's masses alone, moves it by >= 1e-2.
            torch_assert_close(grad / 2, reference_grad[window], rtol=0, atol=1e-7, msg=f"{label} grad")
            if rank % 2:
                # Only the SP leader reports the group's total, including the restoration measurements.
                assert all(metrics[key] == 0 for key in KEYS), (label, metrics)
            totals = torch.tensor([metrics[key] for key in KEYS], dtype=torch.float64)
            dist.all_reduce(totals)
            reference = torch.tensor([reference_metrics[key] for key in KEYS], dtype=torch.float64)
            # Float64 sums of a few terms: the split run differs only by summation order (~1e-16 relative), while a
            # misplaced token or a missed DP / SP term changes a value by >= 1e-2. The k3 sum is summed in fp32 per
            # rank (<= ~1e-6 apart from the reference), and a wrong token moves it by >= 1e-3.
            torch_assert_close(totals[:-1], reference[:-1], rtol=0, atol=1e-12, msg=f"{label} masses")
            torch_assert_close(totals[-1], reference[-1], rtol=0, atol=1e-6, msg=f"{label} k3")

            # Independently of the single-rank run: over the whole world each sign with a survivor ends at its
            # pre-mask mass, and a sign without one reports all of it as unrestored.
            world = dict(zip(KEYS, totals.tolist()))
            for sign in ("pos", "neg"):
                pre, kept = world[f"ratio_mask_{sign}_pre_mass_sum"], world[f"ratio_mask_{sign}_kept_mass_sum"]
                post, lost = (
                    world[f"ratio_rebalance_{sign}_post_mass_sum"],
                    world[f"ratio_rebalance_{sign}_unrestored_mass_sum"],
                )
                if kept > 0:
                    assert abs(post - pre) <= 1e-12 and lost == 0.0, (label, sign, world)
                else:
                    assert post == 0.0 and abs(lost - pre) <= 1e-12, (label, sign, world)
            if case == "drops nothing":
                # Nothing dropped: the flag changes nothing, bit for bit, on this very rank.
                off_grad, _ = run({**case_controls, FLAG: False})
                assert torch.equal(grad, off_grad), (label, grad, off_grad)
    finally:
        dist.destroy_process_group()


def _m2_dp_worker(rank, init_file):
    # Worker 0 holds row A, whose first token (A = +2, log-ratio offset 2) is an M2PO outlier on its own (squared offset 4,
    # average 1 over its 4 tokens), worker 1 holds row B. Ranked together the same outlier is not removed (4 / 8 tokens).
    torch.set_num_threads(1)
    advantages = torch.tensor([[2.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]])
    delta = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

    def run(rows, flag, dp_size, scale=None):
        values = torch.full((len(rows), 4), -2.0, requires_grad=True)
        rows_advantages = advantages[rows] if scale is None else advantages[rows] * scale
        context = dict(
            old_log_probs_shifted=values.detach() - delta[rows],
            advantages=rows_advantages,
            loss_mask=torch.ones(len(rows), 4, dtype=torch.bool),
        )
        config = dict(
            use_cispo_loss=True, is_weight_clip_max=5.0, batch_num_tokens=8, dp_size=dp_size, ratio_m2_threshold=0.75
        )
        loss, metrics = grpo_loss(dict(logprobs=values), context, {}, {**config, FLAG: flag}, "cpu")
        loss.backward()
        return values.grad, metrics

    _, reference = run([0, 1], True, 1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", world_size=2, rank=rank, timeout=timedelta(seconds=60)
    )
    try:
        patch_grpo, patch_functional = _sequence_parallel_group(None)
        with patch_grpo, patch_functional:
            grad, metrics = run([rank], True, 2)
            totals = torch.tensor(
                [metrics[key] for key in [*MASS_KEYS, "ratio_mask_dropped_token_count"]], dtype=torch.float64
            )
            dist.all_reduce(totals)
            pre, kept, dropped = totals[0].item(), totals[1].item(), totals[-1].item()
            # Together: nothing dropped, so pre == kept == 9/8 and the scale is 1. Per worker: A loses its outlier.
            assert (reference["ratio_mask_dropped_token_count"], reference["ratio_mask_pos_kept_mass_sum"]) == (
                0.0,
                1.125,
            )
            assert (pre, kept, dropped) == (1.125, 0.875, 1.0), totals
            # The restoration works on what each worker's own ranking kept: the world's post mass is the pre mass.
            post = torch.tensor(metrics["ratio_rebalance_pos_post_mass_sum"], dtype=torch.float64)
            dist.all_reduce(post)
            assert abs(post.item() - 1.125) <= 1e-12, post
            # and the scale is 9 / 7 on both workers, which is what the loss then uses.
            by_hand, _ = run([rank], False, 2, scale=torch.tensor(9.0 / 7.0, dtype=torch.float64).to(torch.float32))
            assert torch.equal(grad, by_hand), (rank, grad, by_hand)
    finally:
        dist.destroy_process_group()


def _float64_nothing_dropped_worker(rank, init_file, world_size, trials):
    # Float64 advantages are the dtype in which a scale that is 1 only up to rounding would show: for float32 (or lower)
    # advantages the cast of a scale within 1e-16 of 1 gives exactly 1.
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", world_size=world_size, rank=rank, timeout=timedelta(seconds=120)
    )
    try:
        patch_grpo, patch_functional = _sequence_parallel_group(None)
        differing = 0
        with patch_grpo, patch_functional:
            for trial in range(trials):
                generator = torch.Generator().manual_seed(1000 * trial + rank)
                advantages = torch.randn(1, 8, generator=generator, dtype=torch.float64)

                def run(flag):
                    values = torch.full((1, 8), -2.0, requires_grad=True)
                    context = dict(
                        old_log_probs_shifted=values.detach(),  # log ratio 0: no token leaves any band
                        advantages=advantages,
                        loss_mask=torch.ones(1, 8, dtype=torch.bool),
                    )
                    config = dict(
                        use_cispo_loss=True,
                        is_weight_clip_max=5.0,
                        batch_num_tokens=8 * world_size,
                        dp_size=world_size,
                        ratio_stats=True,
                        ratio_mask_bounds_pos=[0.5, 2.0],
                        ratio_mask_bounds_neg=[0.5, 2.0],
                    )
                    loss, metrics = grpo_loss(dict(logprobs=values), context, {}, {**config, FLAG: flag}, "cpu")
                    loss.backward()
                    return loss.detach(), values.grad, metrics

                off_loss, off_grad, _ = run(False)
                on_loss, on_grad, on_metrics = run(True)
                assert on_metrics["ratio_mask_dropped_token_count"] == 0.0
                differs = torch.tensor(float(not (torch.equal(on_loss, off_loss) and torch.equal(on_grad, off_grad))))
                dist.all_reduce(differs)  # a trial differs if it does on any rank
                differing += int(differs.item() > 0)
        # The invariant: with nothing dropped the scale is exactly 1.0, so the flag changes nothing, bit for bit. A scale
        # derived from separately reduced pre and kept sums can differ from 1 in the last bit on some rank counts (the
        # two sums may be added in different orders), which float64 advantages expose and a cast to a narrower advantage
        # dtype hides. The seeds are fixed, so the trials are reproducible.
        assert differing == 0, f"{differing} of {trials} trials with nothing dropped differ from the flag-off run"
    finally:
        dist.destroy_process_group()


class TestRatioRebalanceDistributed(TestCasePlus):
    def test_packed_dp_sp_matches_unsplit_request_in_every_mode(self):
        mp.spawn(_worker, args=(self.get_auto_remove_tmp_dir_str() + "/group",), nprocs=4)

    def test_ratio_m2_threshold_restores_what_each_workers_own_ranking_kept(self):
        mp.spawn(_m2_dp_worker, args=(self.get_auto_remove_tmp_dir_str() + "/group",), nprocs=2)

    def test_float64_advantages_with_nothing_dropped_are_bit_identical_to_flag_off_on_many_ranks(self):
        for world_size in (4, 7):
            with self.subTest(world_size=world_size):
                mp.spawn(
                    _float64_nothing_dropped_worker,
                    args=(self.get_auto_remove_tmp_dir_str() + f"/group{world_size}", world_size, 40),
                    nprocs=world_size,
                )
