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

"""ZeRO-2 CPU offload, fp32 masters, accumulation in the parameter dtype: model calls split into several requests
before one step() apply the same gradient as one request with the same calls (gloo, CPU)."""

import importlib
import os
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

pytest.importorskip("deepspeed")

_WORLD, _DIM = 2, 64


def _worker(rank, rendezvous):
    from arctic_platform.common.deepspeed_engine import OptimizerSettings
    from arctic_platform.common.deepspeed_engine import TrainingEngineSpec
    from arctic_platform.common.deepspeed_engine import build_deepspeed_config
    from arctic_platform.common.deepspeed_engine import initialize_training_engine

    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=_WORLD, timeout=timedelta(seconds=120)
    )
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(_WORLD))
    importlib.import_module("deepspeed.comm.torch").build_shm_op = lambda: None  # a compiled CPU collective
    ds_config = build_deepspeed_config(
        TrainingEngineSpec(
            world_size=_WORLD,
            fp32_masters=True,
            ds_overrides={
                "zero_optimization": {
                    "stage": 2,
                    "reduce_bucket_size": 2000,
                    "allgather_bucket_size": 2000,
                    "offload_optimizer": {"device": "cpu", "pin_memory": False},
                },
                "data_types": {"grad_accum_dtype": "bf16"},  # the host-buffer path under test
                "zero_force_ds_cpu_optimizer": False,
            },
        )
    )
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(_DIM, _DIM, bias=False), nn.Tanh(), nn.Linear(_DIM, 16, bias=False))
    engine, _, _ = initialize_training_engine(
        model.to(torch.bfloat16),
        ds_config,
        optimizer=OptimizerSettings(name="torch_adamw", lr=0.0, fused=False),  # every window sees the same weights
        ep_size=1,
        mpu=None,
        model_path="unused",
        rank=rank,
    )

    def window(requests):
        """Each request's last call is an accumulation boundary; returns the master gradient step() applies."""
        for calls in requests:
            for k, name in enumerate(calls):
                g = torch.Generator().manual_seed(1000 * sum(map(ord, name)) + rank)
                x = torch.randn(8, _DIM, generator=g).to(torch.bfloat16)
                w = 0.0 if name == "ping" else float(torch.randn(1, generator=g))
                engine.set_gradient_accumulation_boundary(k == len(calls) - 1)
                engine.backward(engine(x).float().square().mean() * w)
        master = engine.optimizer.single_partition_of_fp32_groups[0].grad.clone()
        engine.step()
        return master

    frame = ["c0", "c1", "c2", "c3"]
    window([frame])
    for split in ([["ping"], frame], [["c0"], ["c1"]], [["c0"], ["c1"], ["c2", "c3"]]):
        whole = window([[name for calls in split for name in calls]])
        torch.testing.assert_close(window(split), whole, rtol=0, atol=0, msg=lambda detail: f"{split}: {detail}")
    dist.destroy_process_group()


def test_split_requests_apply_the_sum_of_their_model_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("DS_ACCELERATOR", "cpu")
    mp.spawn(_worker, args=((tmp_path / "rendezvous").as_uri(),), nprocs=_WORLD, join=True)
