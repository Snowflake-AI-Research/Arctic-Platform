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

"""ZeRO-2 with fp32 masters sums one step's model calls in fp32, with and without CPU optimizer offload (gloo, CPU)."""

import importlib
import os
import tempfile
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

pytest.importorskip("deepspeed")

_WORLD, _CALLS, _DIM = 2, 16, 64


def _model():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(_DIM, _DIM, bias=False),
        nn.Tanh(),
        nn.Linear(_DIM, _DIM, bias=False),
        nn.Tanh(),
        nn.Linear(_DIM, 16, bias=False),
    ).to(torch.bfloat16)


def _call(step, rank):
    g = torch.Generator().manual_seed(1000 * step + rank)
    return torch.randn(8, _DIM, generator=g).to(torch.bfloat16), float(torch.randn(1, generator=g))


def _worker(rank, rendezvous, out_dir, offload):
    from arctic_platform.common.deepspeed_engine import OptimizerSettings
    from arctic_platform.common.deepspeed_engine import TrainingEngineSpec
    from arctic_platform.common.deepspeed_engine import build_deepspeed_config
    from arctic_platform.common.deepspeed_engine import initialize_training_engine

    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=_WORLD, timeout=timedelta(seconds=120)
    )
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(_WORLD), DS_ACCELERATOR="cpu")
    importlib.import_module("deepspeed.comm.torch").build_shm_op = lambda: (
        None
    )  # a CPU collective op; needs a compiler
    zero = {"stage": 2, "reduce_bucket_size": 2000, "allgather_bucket_size": 2000}  # several buckets per backward
    if offload:
        zero["offload_optimizer"] = {"device": "cpu", "pin_memory": False}
    # The fp32-master body our RL client sends (bf16 parameters, fp32 masters and optimizer states).
    ds_config = build_deepspeed_config(
        TrainingEngineSpec(
            world_size=_WORLD,
            ds_overrides={
                "zero_optimization": zero,
                "bf16": {"enabled": True, "bf16_master_weights_and_grads": False, "bf16_optimizer_states": False},
                "zero_force_ds_cpu_optimizer": False,
            },
        )
    )
    assert ds_config["data_types"] == {"grad_accum_dtype": "fp32"}
    model, twin = _model(), _model()
    engine, _, _ = initialize_training_engine(
        model,
        ds_config,
        optimizer=OptimizerSettings(name="torch_adamw", lr=1e-3, fused=False),
        ep_size=1,
        mpu=None,
        model_path="unused",
        rank=rank,
    )
    expected = None  # fp32 running sum of each call's all-reduced gradient, rounded to bf16 as DeepSpeed stores it
    for k in range(_CALLS):
        per_rank = []
        for r in range(_WORLD):
            x, w = _call(k, r)
            twin.zero_grad(set_to_none=True)
            (twin(x).float().square().mean() * w).backward()
            per_rank.append(torch.cat([p.grad.flatten().float() for p in twin.parameters()]))
        call = sum(g / _WORLD for g in per_rank).to(torch.bfloat16).float()
        expected = call if expected is None else expected + call
        x, w = _call(k, rank)
        engine.set_gradient_accumulation_boundary(k == _CALLS - 1)
        engine.backward(engine(x).float().square().mean() * w)
    zero_optimizer = engine.optimizer
    if offload:
        partition = zero_optimizer.single_partition_of_fp32_groups[0].grad
    else:
        partition = torch.cat([t.flatten().float() for t in zero_optimizer.averaged_gradients[0]])
    parts = [torch.empty_like(partition) for _ in range(_WORLD)]
    dist.all_gather(parts, partition)
    master = torch.cat(parts)[: expected.numel()]
    torch.save(dict(master=master, expected=expected), os.path.join(out_dir, f"rank{rank}.pt"))
    dist.destroy_process_group()


@pytest.mark.parametrize("offload", [False, True], ids=["no_offload", "cpu_offload"])
def test_fp32_masters_receive_the_fp32_sum_of_every_model_call(offload):
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(f"file://{tmp}/rendezvous", tmp, offload), nprocs=_WORLD, join=True)
        result = torch.load(os.path.join(tmp, "rank0.pt"))
    master, expected = result["master"], result["expected"]
    bf16_sum_error = float((master - expected).norm() / expected.norm())
    assert torch.equal(master, expected), (
        f"master gradient differs from the fp32 sum by {bf16_sum_error:.2e} (rel. L2)"
    )
