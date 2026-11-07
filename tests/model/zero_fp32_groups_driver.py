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
"""Multi-rank driver: FP32 parameters keep their gradient and update through a ZeRO-2 engine with BF16 masters.

Every rank scales the loss differently, so the expected gradient is the FP32 mean of unequal, non-BF16 values.
"""

import sys

import deepspeed
import torch
import torch.nn as nn
from deepspeed.utils import safe_get_full_grad

from arctic_platform.common.deepspeed_engine import OptimizerSettings
from arctic_platform.common.deepspeed_engine import TrainingEngineSpec
from arctic_platform.common.deepspeed_engine import build_deepspeed_config
from arctic_platform.common.deepspeed_engine import initialize_training_engine
from arctic_platform.model.implementations.fp8 import mark_keep_fp32

LR = 1e-4
ACCUMULATION_STEPS = 2


class Toy(nn.Module):
    @classmethod
    def keeps_fp32_parameter(cls, name):
        return name.startswith("protected")

    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(64, 64, bias=False)
        self.protected_a = mark_keep_fp32(nn.Parameter(torch.linspace(-8.0123, 7.98765, 37)))
        self.protected_b = mark_keep_fp32(nn.Parameter(torch.linspace(-3.3333, 5.55555, 1000)))


def main():
    deepspeed.init_distributed()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    torch.cuda.set_device(rank % torch.cuda.device_count())

    model = Toy().bfloat16()
    for parameter in (model.protected_a, model.protected_b):
        parameter.data = parameter.data.float()
    exact = {name: getattr(model, name).detach().clone().cuda() for name in ("protected_a", "protected_b")}
    spec = TrainingEngineSpec(
        dtype="bfloat16",
        world_size=world_size,
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        gradient_clipping=None,
    )
    engine, _, _ = initialize_training_engine(
        model.cuda(),
        build_deepspeed_config(spec),
        optimizer=OptimizerSettings(name="fused_adam", lr=LR, eps=1e-15),
        ep_size=1,
        mpu=None,
        model_path="unused",
        rank=rank,
    )
    optimizer = engine.optimizer
    assert [master.dtype for master in optimizer.single_partition_of_fp32_groups] == [torch.bfloat16, torch.float32]

    rank_scale = [1.001 + 0.0003 * r for r in range(world_size)]
    step_scale = [1.0 + 0.3 * 0.5**k for k in range(ACCUMULATION_STEPS)]
    expected = sum(rank_scale) / world_size * sum(step_scale) / ACCUMULATION_STEPS
    protected = [engine.module.protected_a, engine.module.protected_b]
    for k in range(ACCUMULATION_STEPS):
        loss = rank_scale[rank] * step_scale[k] * sum(parameter.sum() for parameter in protected)
        engine.backward(loss)
        if k == ACCUMULATION_STEPS - 1:
            for parameter in protected:
                gradient = safe_get_full_grad(parameter)
                assert gradient.dtype == torch.float32
                assert (gradient - expected).abs().max().item() <= 2e-7, (gradient.flatten()[:2], expected)
        engine.step()

    for name, parameter in zip(("protected_a", "protected_b"), protected):
        moved = (exact[name] - parameter.detach()).abs()
        assert (moved - LR).abs().max().item() <= 2e-6, name  # a BF16 master would not move at all

    state = optimizer.optimizer.state[optimizer.single_partition_of_fp32_groups[1]]
    assert state["exp_avg"].dtype == torch.float32
    print(f"rank {rank} ok", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
