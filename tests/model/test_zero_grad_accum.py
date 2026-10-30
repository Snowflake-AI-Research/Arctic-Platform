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


def _worker(rank, rendezvous, offload, overlap, contiguous, managed):
    import deepspeed

    from arctic_platform.common.deepspeed_engine import OptimizerSettings
    from arctic_platform.common.deepspeed_engine import TrainingEngineSpec
    from arctic_platform.common.deepspeed_engine import build_deepspeed_config
    from arctic_platform.common.deepspeed_engine import initialize_training_engine

    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=_WORLD, timeout=timedelta(seconds=120)
    )
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(_WORLD), DS_ACCELERATOR="cpu")
    importlib.import_module("deepspeed.comm.torch").build_shm_op = (
        lambda: None
    )  # a CPU collective op; needs a compiler
    zero = {
        "stage": 2,
        "reduce_bucket_size": 2000,
        "allgather_bucket_size": 2000,
        "overlap_comm": overlap,
        "contiguous_gradients": contiguous,
    }  # several buckets per backward
    if offload:
        zero["offload_optimizer"] = {"device": "cpu", "pin_memory": False}
    # bf16 parameters with fp32 masters and optimizer states.
    ds_config = build_deepspeed_config(
        TrainingEngineSpec(
            world_size=_WORLD,
            gradient_clipping=0.01,
            ds_overrides={
                "zero_optimization": zero,
                "bf16": {"enabled": True, "bf16_master_weights_and_grads": False, "bf16_optimizer_states": False},
                "zero_force_ds_cpu_optimizer": False,
                "managed_gradient_accumulation": managed,
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
    zero_optimizer = engine.optimizer
    optimizer_step = zero_optimizer.step

    def check_partition_then_step(*args, **kwargs):
        # The engine finalized the window; inspect the sum before clipping or clearing it.
        if offload:
            partition = zero_optimizer.single_partition_of_fp32_groups[0].grad
        else:
            partition = torch.cat([t.flatten().float() for t in zero_optimizer.averaged_gradients[0]])
        parts = [torch.empty_like(partition) for _ in range(_WORLD)]
        dist.all_gather(parts, partition)
        master = torch.cat(parts)[: expected.numel()]
        assert torch.equal(master, expected), f"calls={calls}, error={(master - expected).norm() / expected.norm()}"
        return optimizer_step(*args, **kwargs)

    zero_optimizer.step = check_partition_then_step
    for calls in (1, _CALLS, _CALLS):  # first call and reset after optimizer steps
        twin.load_state_dict(engine.module.state_dict())
        expected = None
        for k in range(calls):
            per_rank = []
            for r in range(_WORLD):
                x, w = _call(k, r)
                twin.zero_grad(set_to_none=True)
                (twin(x).float().square().mean() * w).backward()
                per_rank.append(torch.cat([p.grad.flatten().float() for p in twin.parameters()]))
            call = sum(g / _WORLD for g in per_rank)
            if deepspeed.__version__ == "0.19.7":
                call = call.to(torch.bfloat16).float()  # 0.19.8 retains fp32 reduction outputs too
            expected = call if expected is None else expected + call
            x, w = _call(k, rank)
            if managed:
                engine.set_gradient_accumulation_boundary(k == calls - 1)
            engine.backward(engine(x).float().square().mean() * w)
        engine.step()
        torch.testing.assert_close(
            torch.as_tensor(engine.get_global_grad_norm()).float(), expected.norm(), rtol=1e-6, atol=1e-7
        )
    dist.destroy_process_group()


@pytest.mark.parametrize("offload", [False, True], ids=["no_offload", "cpu_offload"])
@pytest.mark.parametrize(
    "overlap,contiguous,managed", [(False, True, True), (True, True, True), (False, False, True), (False, True, False)]
)
def test_fp32_masters_receive_the_fp32_sum_of_every_model_call(offload, overlap, contiguous, managed):
    import deepspeed

    if deepspeed.__version__ == "0.19.7" and not offload and not contiguous:
        pytest.skip("Stock 0.19.7 noncontiguous reduction already fails single-backward parity; fixed in 0.19.8")
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(
            _worker,
            args=(f"file://{tmp}/rendezvous", offload, overlap, contiguous, managed),
            nprocs=_WORLD,
            join=True,
        )


def test_unknown_deepspeed_version_fails_before_patching(monkeypatch):
    import deepspeed

    from arctic_platform.common.zero_grad_accum import patch_zero2_grad_accum_dtype

    monkeypatch.setattr(deepspeed, "__version__", "0.19.9")
    with pytest.raises(RuntimeError, match="validate the new version"):
        patch_zero2_grad_accum_dtype()
