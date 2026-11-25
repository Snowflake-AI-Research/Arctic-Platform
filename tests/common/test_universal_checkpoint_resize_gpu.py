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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for ``load_checkpoint_with_dp_resize`` on real DeepSpeed ZeRO-1/2 engines.

A checkpoint saved with ``engine.save_checkpoint`` at one DP size is loaded at other DP sizes. The fp32 masters, and
the Adam moments and step when loaded, must come back bitwise, compared per parameter with the values gathered
before the save. Every rank runs on one host, so the checkpoint directory is a shared filesystem and rank 0 alone
converts. Tests needing four ranks skip on smaller machines.
"""

from __future__ import annotations

import os
from datetime import timedelta
from pathlib import Path

import torch

from arctic_platform.testing_utils import TestCasePlus
from arctic_platform.testing_utils import get_unique_port_number
from arctic_platform.testing_utils import require_deepspeed
from arctic_platform.testing_utils import require_torch_multi_gpu
from arctic_platform.testing_utils import reserve_free_port

_MOMENTS = ("exp_avg", "exp_avg_sq")


class _UnevenParams(torch.nn.Module):
    # 35 + 13 elements: partitions are uneven at DP 2, 3 and 4.
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.randn(7, 5))
        self.b = torch.nn.Parameter(torch.randn(13))

    def forward(self, x):
        return (x @ self.a.t()).sum() + (self.b * self.b).sum()


def _init(rank: int, world: int, port: int) -> None:
    import deepspeed

    os.environ.update(
        MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world)
    )
    torch.cuda.set_device(rank)
    deepspeed.init_distributed(timeout=timedelta(minutes=5))


def _engine(world: int, stage: int, sp_size: int):
    import deepspeed

    mpu = None
    if sp_size > 1:
        from arctic_platform.model.parallel.groups import setup_parallel_groups

        mpu = setup_parallel_groups(sp_size, 1, world_size=world, gpus_per_node=world).mpu
    torch.manual_seed(0)
    model = _UnevenParams()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": world // sp_size,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": stage},
        "communication_data_type": "fp32",
    }
    engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config, mpu=mpu)
    return engine


def _train_step(engine) -> float:
    loss = engine(torch.ones(1, 5, device=engine.device, dtype=torch.bfloat16))
    engine.backward(loss)
    engine.step()
    return float(loss.detach())


def _full_state(engine, with_moments: bool) -> dict:
    from deepspeed.utils import safe_get_full_fp32_param
    from deepspeed.utils import safe_get_full_optimizer_state

    if with_moments:
        # Released DeepSpeed links the optimizer-state fragments only at the first step after a legacy load.
        engine.optimizer._lazy_init_hp_params_optimizer_state()
    # Collective on ZeRO-1/2: every rank must call it.
    state = {}
    for name, param in engine.module.named_parameters():
        entry = {"fp32": safe_get_full_fp32_param(param).detach().cpu().clone()}
        if with_moments:
            for key in _MOMENTS:
                entry[key] = safe_get_full_optimizer_state(param, key).detach().cpu().clone()
        state[name] = entry
    return state


def _adam_steps(engine) -> list:
    return sorted({float(s["step"]) for s in engine.optimizer.optimizer.state.values() if "step" in s})


def _save(rank: int, world: int, port: int, root: str, stage: int, sp_size: int) -> None:
    import torch.distributed as dist

    _init(rank, world, port)
    try:
        engine = _engine(world, stage, sp_size)
        _train_step(engine)
        with torch.no_grad():
            # Masters that bf16 cannot represent, so a rebuild from the bf16 weights would show.
            for group in engine.optimizer.single_partition_of_fp32_groups:
                group.add_(2**-12)
        # Save bf16 weights rounded from those masters, as a training step would.
        engine.optimizer.update_lp_params()
        reference = _full_state(engine, with_moments=True)
        if rank == 0:
            torch.save({"state": reference, "adam_steps": _adam_steps(engine)}, Path(root) / "reference.pt")
        engine.save_checkpoint(str(Path(root) / "ckpt"))
    finally:
        dist.destroy_process_group()


def _load(rank: int, world: int, port: int, root: str, stage: int, sp_size: int, load_optimizer_states: bool) -> None:
    import torch.distributed as dist

    from arctic_platform.common.utils.universal_checkpoint import load_checkpoint_with_dp_resize

    _init(rank, world, port)
    try:
        engine = _engine(world, stage, sp_size)
        ckpt = Path(root) / "ckpt"
        before = sorted(p.name for p in ckpt.iterdir())
        load_path, _ = load_checkpoint_with_dp_resize(
            engine,
            str(ckpt),
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_optimizer_states,
        )
        assert load_path is not None
        assert engine.global_steps == 1, engine.global_steps
        saved = torch.load(Path(root) / "reference.pt")
        loaded = _full_state(engine, with_moments=load_optimizer_states)
        for name, entry in saved["state"].items():
            assert torch.equal(loaded[name]["fp32"], entry["fp32"]), name
            if load_optimizer_states:
                for key in _MOMENTS:
                    assert torch.equal(loaded[name][key], entry[key]), (name, key)
        for name, param in engine.module.named_parameters():
            assert torch.equal(param.detach().cpu(), saved["state"][name]["fp32"].to(torch.bfloat16)), name
        if load_optimizer_states:
            assert _adam_steps(engine) == saved["adam_steps"], (_adam_steps(engine), saved["adam_steps"])
        else:
            assert not any("exp_avg" in s for s in engine.optimizer.optimizer.state.values())
        dist.barrier()
        assert sorted(p.name for p in ckpt.iterdir()) == before, "the universal copy was not removed"
        assert torch.isfinite(torch.tensor(_train_step(engine)))
        assert engine.global_steps == 2
    finally:
        dist.destroy_process_group()


@require_deepspeed
@require_torch_multi_gpu
class TestUniversalCheckpointResize(TestCasePlus):
    def _spawn(self, world: int, worker, *args) -> None:
        if torch.cuda.device_count() < world:
            self.skipTest(f"needs {world} GPUs")
        port = reserve_free_port(get_unique_port_number() + 1, span=7)
        torch.multiprocessing.spawn(worker, args=(world, port, *args), nprocs=world, join=True)

    def _check(self, stage: int, save: tuple[int, int], loads: list[tuple[int, int, bool]]) -> None:
        root = self.get_auto_remove_tmp_dir()
        self._spawn(save[0], _save, root, stage, save[1])
        for world, sp_size, load_optimizer_states in loads:
            with self.subTest(world=world, sp_size=sp_size, load_optimizer_states=load_optimizer_states):
                self._spawn(world, _load, root, stage, sp_size, load_optimizer_states)

    def test_zero2_shrink_restores_masters_and_moments(self):
        self._check(2, (4, 1), [(2, 1, True), (3, 1, True), (1, 1, False)])

    def test_zero2_grow(self):
        self._check(2, (2, 1), [(4, 1, True), (3, 1, False)])

    def test_zero1_at_saved_dp_and_grow(self):
        self._check(1, (2, 1), [(2, 1, True), (4, 1, True)])

    def test_ulysses_sequence_parallel(self):
        # Saved at SP 2 x DP 2; loaded at SP 2 x DP 1 and at DP 1 without SP.
        self._check(2, (4, 2), [(2, 2, True), (1, 1, True)])
