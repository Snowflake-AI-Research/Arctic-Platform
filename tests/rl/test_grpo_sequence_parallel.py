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

"""CPU Gloo parity for GRPO reductions over real sequence-parallel process groups."""

from __future__ import annotations

from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import arctic_platform.rl.processors.functional as functional
import arctic_platform.rl.processors.grpo as grpo_module
from arctic_platform.rl.processors.functional import _compute_sequence_level_ratio_and_advantages
from arctic_platform.rl.processors.functional import agg_loss
from arctic_platform.rl.processors.functional import cispo_actor_loss_fn
from arctic_platform.rl.processors.functional import echo_env_prediction_loss_fn

_TOKENS = 8
_CU_SEQLENS = torch.tensor([0, 5, 8], dtype=torch.int32)
_SEQUENCE_WEIGHTS = torch.tensor([0.4, 0.6])
# Gloo sums the two FP32 windows in a different order from the unsharded
# reference. The observed rounding bound is below one FP32 ulp at these values.
_FP32_COLLECTIVE_ATOL = 1e-6


def _window_cu_seqlens(rank: int) -> torch.Tensor:
    start = rank * (_TOKENS // 2)
    end = start + (_TOKENS // 2)
    boundaries = (_CU_SEQLENS.clamp(min=start, max=end) - start).to(torch.int32)
    boundaries[-1] = end - start
    return boundaries


def _sequence_policy_loss(
    logprobs: torch.Tensor,
    proximal: torch.Tensor,
    advantages: torch.Tensor,
    mask: torch.Tensor,
    cu_seqlens: torch.Tensor,
):
    return cispo_actor_loss_fn(
        logprobs=logprobs,
        proximal_logprobs=proximal,
        old_logprobs=proximal,
        advantages=advantages,
        eps_clip=0.2,
        loss_mask=mask,
        is_weight_clip_max=5.0,
        importance_sampling_level="sequence",
        cu_seqlens=cu_seqlens,
        loss_agg_mode="token-mean",
        dp_size=1,
        batch_num_tokens=int(mask.new_tensor([1, 1, 1, 1, 1, 1]).sum()),
    )


def _sp_worker(rank: int, init_method: str) -> None:
    torch.set_num_threads(1)
    width = _TOKENS // 2
    start = rank * width
    window = slice(start, start + width)
    cu_seqlens = _window_cu_seqlens(rank)

    prompt_values = torch.tensor([0.0, 1.0, 2.0, 4.0, 6.0, 0.0, 3.0, 5.0])
    prompt_mask = torch.tensor([0, 1, 1, 1, 1, 0, 1, 1], dtype=torch.bool)
    prompt_whole = prompt_values.clone().requires_grad_()
    expected_prompt = agg_loss(
        prompt_whole,
        prompt_mask,
        "prompt-mean",
        sequence_loss_weights=_SEQUENCE_WEIGHTS,
        cu_seqlens=_CU_SEQLENS,
    )
    expected_prompt_grad = torch.autograd.grad(expected_prompt, prompt_whole)[0]

    logprobs = -torch.linspace(0.5, 1.2, _TOKENS)
    proximal = logprobs.detach() - torch.tensor([0.01, 0.02, 0.03, 0.04, 0.01, 0.02, 0.03, 0.04])
    advantages = torch.tensor([0.0, 0.5, -0.2, 0.7, 0.1, 0.0, -0.4, 0.6])
    sequence_mask = torch.tensor([0, 1, 1, 1, 1, 0, 1, 1], dtype=torch.bool)
    expected_ratio, expected_advantages = _compute_sequence_level_ratio_and_advantages(
        logprobs - proximal,
        advantages,
        sequence_mask,
        _CU_SEQLENS,
    )
    sequence_whole = logprobs.clone().requires_grad_()
    expected_sequence_loss, _ = _sequence_policy_loss(
        sequence_whole,
        proximal,
        advantages,
        sequence_mask,
        _CU_SEQLENS,
    )
    expected_sequence_grad = torch.autograd.grad(expected_sequence_loss, sequence_whole)[0]

    echo_values = -torch.linspace(0.25, 2.0, _TOKENS)
    policy_mask = torch.tensor([0, 1, 1, 0, 0, 0, 0, 0], dtype=torch.bool)
    observation_mask = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0], dtype=torch.bool)
    target_mask = torch.tensor([0, 0, 0, 0, 1, 1, 1, 0], dtype=torch.bool)
    echo_whole = echo_values.clone().requires_grad_()
    expected_echo, expected_echo_stats = echo_env_prediction_loss_fn(
        echo_whole,
        target_mask,
        observation_mask,
        policy_mask,
        global_num_echo_sequences=2,
        cu_seqlens=_CU_SEQLENS,
    )
    expected_echo_grad = torch.autograd.grad(expected_echo, echo_whole)[0]

    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:

        def get_world_group():
            return dist.group.WORLD

        with (
            patch.object(functional, "_get_sequence_parallel_group", side_effect=get_world_group),
            patch.object(grpo_module, "_get_sequence_parallel_group", side_effect=get_world_group),
        ):
            local_prompt = prompt_values[window].clone().requires_grad_()
            prompt_loss = agg_loss(
                local_prompt,
                prompt_mask[window],
                "prompt-mean",
                sequence_loss_weights=_SEQUENCE_WEIGHTS,
                cu_seqlens=cu_seqlens,
            )
            prompt_grad = torch.autograd.grad(prompt_loss, local_prompt)[0]
            prompt_total = prompt_loss.detach().clone()
            dist.all_reduce(prompt_total)
            torch.testing.assert_close(prompt_total, expected_prompt, rtol=0, atol=_FP32_COLLECTIVE_ATOL)
            torch.testing.assert_close(
                prompt_grad,
                expected_prompt_grad[window],
                rtol=0,
                atol=_FP32_COLLECTIVE_ATOL,
            )

            local_ratio, local_advantages = _compute_sequence_level_ratio_and_advantages(
                (logprobs - proximal)[window],
                advantages[window],
                sequence_mask[window],
                cu_seqlens,
            )
            torch.testing.assert_close(
                local_ratio,
                expected_ratio[window],
                rtol=0,
                atol=_FP32_COLLECTIVE_ATOL,
            )
            torch.testing.assert_close(
                local_advantages,
                expected_advantages[window],
                rtol=0,
                atol=_FP32_COLLECTIVE_ATOL,
            )

            local_sequence = logprobs[window].clone().requires_grad_()
            sequence_loss, _ = _sequence_policy_loss(
                local_sequence,
                proximal[window],
                advantages[window],
                sequence_mask[window],
                cu_seqlens,
            )
            sequence_grad = torch.autograd.grad(sequence_loss, local_sequence)[0]
            sequence_total = sequence_loss.detach().clone()
            dist.all_reduce(sequence_total)
            torch.testing.assert_close(
                sequence_total,
                expected_sequence_loss,
                rtol=0,
                atol=_FP32_COLLECTIVE_ATOL,
            )
            torch.testing.assert_close(
                sequence_grad,
                expected_sequence_grad[window],
                rtol=0,
                atol=_FP32_COLLECTIVE_ATOL,
            )

            local_echo = echo_values[window].clone().requires_grad_()
            echo_loss, echo_stats = echo_env_prediction_loss_fn(
                local_echo,
                target_mask[window],
                observation_mask[window],
                policy_mask[window],
                global_num_echo_sequences=2,
                cu_seqlens=cu_seqlens,
            )
            echo_grad = torch.autograd.grad(echo_loss, local_echo)[0]
            echo_total = echo_loss.detach().clone()
            dist.all_reduce(echo_total)
            torch.testing.assert_close(echo_total, expected_echo, rtol=0, atol=_FP32_COLLECTIVE_ATOL)
            torch.testing.assert_close(
                echo_grad,
                expected_echo_grad[window],
                rtol=0,
                atol=_FP32_COLLECTIVE_ATOL,
            )

            sequence_count = torch.tensor(
                [
                    echo_stats["num_real_sequences"],
                    echo_stats["num_echo_bearing_sequences"],
                ],
                dtype=torch.long,
            )
            dist.all_reduce(sequence_count)
            assert sequence_count.tolist() == [
                expected_echo_stats["num_real_sequences"],
                expected_echo_stats["num_echo_bearing_sequences"],
            ]
            if rank == 1:
                assert not policy_mask[window].any()
                assert echo_loss.item() > 0
    finally:
        dist.destroy_process_group()


def test_grpo_sequence_parallel_cpu_gloo(tmp_path):
    mp.spawn(
        _sp_worker,
        args=((tmp_path / "gloo").as_uri(),),
        nprocs=2,
        join=True,
    )
