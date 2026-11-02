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
"""Stochastic rounding of ZeRO-1/2's bf16 parameters from their fp32 master weights.

After each optimizer step ZeRO-1/2 copies every rank's fp32 master partition into its bf16 partition with
round-to-nearest and all-gathers the bf16 parameters, which the forward pass and every weight export read. A master
that moved less than half a bf16 spacing (about ``|w| / 256``) since it last crossed a rounding midpoint leaves its bf16
copy unchanged, so the bf16 parameters trail the masters, on the side the updates come from. Rounding up with
probability equal to the master's fractional position between its two bf16 neighbours makes each bf16 parameter an
unbiased estimate of its master instead.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.distributed as dist

# Elements rounded per pass, so the temporaries stay a few hundred MB however large a partition is.
_CHUNK_NUMEL = 1 << 24


def stochastic_round_to_bf16(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """fp32 ``x`` rounded to one of its two bf16 neighbours, up with probability equal to its fractional position.

    Adds 16 uniform random bits below bf16's last mantissa bit, then truncates them. The bit pattern is sign-magnitude,
    so this rounds away from zero with the probability of the dropped fraction for either sign, and ``E[out] == x``.
    Non-finite values round to nearest.
    """
    finite = torch.isfinite(x)
    noise = torch.randint(0, 1 << 16, x.shape, dtype=torch.int32, device=x.device, generator=generator)
    noise.masked_fill_(~finite, 0)
    truncated = noise.add_(x.view(torch.int32)).bitwise_and_(-(1 << 16)).view(torch.float32)
    return torch.where(finite, truncated, x).to(torch.bfloat16)


def round_bf16_params_stochastically(engine, seed: int) -> None:
    """Re-derive ``engine``'s bf16 parameters from their fp32 masters by stochastic rounding after every step.

    After DeepSpeed's own step, each rank rounds its fp32 master partition into its bf16 partition and the bf16
    groups are all-gathered again, as ``update_lp_params`` does. That method is replaced too, since DeepSpeed calls
    it to re-derive the bf16 parameters after loading a universal checkpoint. A step skipped on overflow draws nothing.
    Each draw is seeded by ``seed``, the engine's global step, the group and the global rank.

    Raises ``ValueError`` unless the engine is a ZeRO-1/2 optimizer whose bf16 parameter groups have fp32 masters.
    """
    from deepspeed.runtime.zero import stage_1_and_2

    optimizer = engine.optimizer
    if not isinstance(optimizer, stage_1_and_2.DeepSpeedZeroOptimizer):
        raise ValueError(
            "stochastic_round_bf16_params needs DeepSpeed's ZeRO stage 1 or 2 optimizer (DeepSpeedZeroOptimizer); this"
            f" engine runs {type(optimizer).__name__}"
        )
    # Groups of parameters kept in fp32 hold their master exactly and are left alone.
    groups = [i for i, flat in enumerate(optimizer.bit16_groups_flat) if flat.dtype == torch.bfloat16]
    master_dtypes = {optimizer.single_partition_of_fp32_groups[i].dtype for i in groups}
    if master_dtypes != {torch.float32}:
        raise ValueError(
            "stochastic_round_bf16_params needs bf16 parameters with fp32 master weights"
            " (bf16.bf16_master_weights_and_grads false); the bf16 groups' masters are"
            f" {sorted(map(str, master_dtypes))}"
        )

    @torch.no_grad()
    def update_lp_params():
        for i in groups:
            partition_id = dist.get_rank(group=optimizer.real_dp_process_group[i])
            bf16 = optimizer.parallel_partitioned_bit16_groups[i][partition_id]
            master = optimizer.single_partition_of_fp32_groups[i]
            entropy = np.random.SeedSequence([seed, engine.global_steps, i, dist.get_rank()])
            generator = torch.Generator(device=bf16.device)
            generator.manual_seed(int(entropy.generate_state(1, np.uint64)[0]))
            for start in range(0, master.numel(), _CHUNK_NUMEL):
                n = min(_CHUNK_NUMEL, master.numel() - start)
                # With an offloaded optimizer the master is on the CPU: round each chunk on the bf16 copy's device.
                chunk = master.narrow(0, start, n).to(bf16.device)
                bf16.narrow(0, start, n).copy_(stochastic_round_to_bf16(chunk, generator))
        # Looked up on the module at call time, so a replacement installed there after this hook is used too.
        stage_1_and_2.all_gather_dp_groups(
            groups_flat=[optimizer.bit16_groups_flat[i] for i in groups],
            partitioned_param_groups=[optimizer.parallel_partitioned_bit16_groups[i] for i in groups],
            dp_process_group=[optimizer.real_dp_process_group[i] for i in groups],
            start_alignment_factor=optimizer.nccl_start_alignment_factor,
            allgather_bucket_size=optimizer.allgather_bucket_size,
        )

    step = optimizer.step

    def step_then_round(*args, **kwargs):
        step(*args, **kwargs)
        if not optimizer.overflow:
            update_lp_params()

    optimizer.step = step_then_round
    optimizer.update_lp_params = update_lp_params
