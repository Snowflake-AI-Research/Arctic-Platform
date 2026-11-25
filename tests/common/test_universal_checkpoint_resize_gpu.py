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

import contextlib
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
        # Released DeepSpeed links the optimizer-state fragments once, after the first param group's first step (or
        # at the first step after a legacy load), so later groups such as the expert groups would read no state.
        engine.optimizer._hp_optimizer_states_linked = False
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


_EP_TAG = "global_step1"


class _Experts(torch.nn.Module):
    """Fused SwiGLU experts laid out as Arctic Platform's GroupedExperts: [local experts, ...] per weight."""

    def __init__(self, local: int, seed: int, dim: int = 4, hidden: int = 6):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        self.w1 = torch.nn.Parameter(torch.randn(local, hidden, dim, generator=generator) * 0.3)
        self.w2 = torch.nn.Parameter(torch.randn(local, dim, hidden, generator=generator) * 0.3)
        self.w3 = torch.nn.Parameter(torch.randn(local, hidden, dim, generator=generator) * 0.3)

    def forward(self, x):
        gate = torch.einsum("td,ehd->eth", x, self.w1)
        up = torch.einsum("td,ehd->eth", x, self.w3)
        return torch.einsum("eth,edh->td", torch.nn.functional.silu(gate) * up, self.w2)


class _MoEModel(torch.nn.Module):
    # Two MoE layers of four experts, two per EP rank; each EP rank's experts differ, replicas share them.
    def __init__(self, ep_rank: int, ep_size: int):
        super().__init__()
        torch.manual_seed(0)
        self.proj = torch.nn.Linear(4, 4, bias=False)
        self.layers = torch.nn.ModuleList()
        for layer in range(2):
            block = torch.nn.Module()
            block.mlp = torch.nn.Module()
            block.mlp.experts = _Experts(4 // ep_size, seed=100 * layer + ep_rank)
            self.layers.append(block)

    def forward(self, x):
        h = self.proj(x)
        for block in self.layers:
            h = h + block.mlp.experts(h)
        return h.float().sum()


@contextlib.contextmanager
def _non_moe_checkpoint(engine):
    # As DSS saves expert parallelism: dense state through DeepSpeed's non-MoE path, experts in per-rank files.
    saved = engine.has_moe_layers
    engine.has_moe_layers = False
    try:
        yield
    finally:
        engine.has_moe_layers = saved


def _ep_engine(world: int, stage: int, ep_size: int):
    import deepspeed
    import deepspeed.runtime.engine as ds_engine
    import torch.distributed as dist
    from deepspeed.moe.utils import split_params_into_different_moe_groups_for_optimizer
    from deepspeed.utils import groups

    groups._create_expert_and_data_parallel(ep_size)
    group_name = f"ep_size_{ep_size}"
    model = _MoEModel(dist.get_rank(groups._get_expert_parallel_group(group_name)), ep_size)
    for block in model.layers:
        for param in block.mlp.experts.parameters():
            param.allreduce = False
            param.group_name = group_name
    original = ds_engine.DeepSpeedEngine._configure_distributed_model

    def detect_experts(self, module):
        original(self, module)
        self.has_moe_layers = True
        self.num_experts = [4, 4]

    ds_engine.DeepSpeedEngine._configure_distributed_model = detect_experts
    param_groups = split_params_into_different_moe_groups_for_optimizer(
        [{"name": "default", "params": list(model.parameters())}]
    )
    optimizer = torch.optim.AdamW(param_groups, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "train_batch_size": world,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": stage},
        "communication_data_type": "fp32",
    }
    try:
        engine, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=config)
    finally:
        ds_engine.DeepSpeedEngine._configure_distributed_model = original
    return engine, group_name


def _ep_ranks(group_name: str) -> tuple[int, int]:
    import torch.distributed as dist
    from deepspeed.utils import groups

    return (
        dist.get_rank(groups._get_expert_parallel_group(group_name)),
        dist.get_rank(groups._get_expert_data_parallel_group(group_name)),
    )


def _ep_train_step(engine) -> float:
    loss = engine(torch.ones(3, 4, device=engine.device, dtype=torch.bfloat16))
    engine.backward(loss)
    engine.step()
    return float(loss.detach())


def _ep_save(rank: int, world: int, port: int, root: str, stage: int, ep_size: int) -> None:
    import torch.distributed as dist

    from arctic_platform.common.utils.universal_checkpoint import expert_parallel_parameter_names

    _init(rank, world, port)
    try:
        engine, group_name = _ep_engine(world, stage, ep_size)
        _ep_train_step(engine)
        with torch.no_grad():
            for group in engine.optimizer.single_partition_of_fp32_groups:
                group.add_(2**-12)
        engine.optimizer.update_lp_params()
        reference = _full_state(engine, with_moments=True)
        ckpt = Path(root) / "ckpt"
        with _non_moe_checkpoint(engine):
            engine.save_checkpoint(str(ckpt), tag=_EP_TAG)
        ep_rank, edp_rank = _ep_ranks(group_name)
        if edp_rank == 0:
            params = dict(engine.module.named_parameters())
            experts = {
                name: params[name].detach().cpu().clone() for name in expert_parallel_parameter_names(engine.module)
            }
            torch.save(experts, ckpt / _EP_TAG / f"experts_ep_rank_{ep_rank:03d}.pt")
            torch.save(
                {"state": reference, "adam_steps": _adam_steps(engine)}, Path(root) / f"reference_ep{ep_rank}.pt"
            )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _ep_load(rank: int, world: int, port: int, root: str, stage: int, ep_size: int, load_optimizer_states: bool):
    import torch.distributed as dist

    from arctic_platform.common.utils.universal_checkpoint import UNIVERSAL_TAG_SUFFIX
    from arctic_platform.common.utils.universal_checkpoint import convert_to_universal
    from arctic_platform.common.utils.universal_checkpoint import universal_load

    _init(rank, world, port)
    try:
        engine, group_name = _ep_engine(world, stage, ep_size)
        src = Path(root) / "ckpt" / _EP_TAG
        out_root = Path(root) / f"universal-{world}"
        out_tag = _EP_TAG + UNIVERSAL_TAG_SUFFIX
        if rank == 0:
            shards = [str(src / f"experts_ep_rank_{r:03d}.pt") for r in range(ep_size)]
            convert_to_universal(str(src), str(out_root / out_tag), expert_shards=shards)
        dist.barrier()
        with universal_load(engine), _non_moe_checkpoint(engine):
            load_path, _ = engine.load_checkpoint(
                str(out_root),
                out_tag,
                load_optimizer_states=load_optimizer_states,
                load_lr_scheduler_states=load_optimizer_states,
            )
        assert load_path is not None
        assert engine.global_steps == 1, engine.global_steps
        ep_rank, _ = _ep_ranks(group_name)
        saved = torch.load(Path(root) / f"reference_ep{ep_rank}.pt")
        loaded = _full_state(engine, with_moments=load_optimizer_states)
        assert sorted(loaded) == sorted(saved["state"]), (sorted(loaded), sorted(saved["state"]))
        for name, entry in saved["state"].items():
            assert torch.equal(loaded[name]["fp32"], entry["fp32"]), name
            if load_optimizer_states:
                for key in _MOMENTS:
                    assert torch.equal(loaded[name][key], entry[key]), (name, key)
        for name, param in engine.module.named_parameters():
            assert torch.equal(param.detach().cpu(), saved["state"][name]["fp32"].to(torch.bfloat16)), name
        if load_optimizer_states:
            assert _adam_steps(engine) == saved["adam_steps"], (_adam_steps(engine), saved["adam_steps"])
        assert torch.isfinite(torch.tensor(_ep_train_step(engine)))
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
        import deepspeed
        from packaging.version import Version

        from arctic_platform.common.utils.universal_checkpoint import SEQUENCE_PARALLEL_MIN_DEEPSPEED

        if Version(deepspeed.__version__) < Version(SEQUENCE_PARALLEL_MIN_DEEPSPEED):
            self.skipTest(f"Ulysses SP universal resize needs deepspeed>={SEQUENCE_PARALLEL_MIN_DEEPSPEED}")
        # Saved at SP 2 x DP 2; loaded at SP 2 x DP 1 and at DP 1 without SP.
        self._check(2, (4, 2), [(2, 2, True), (1, 1, True)])

    def _check_ep(self, stage: int, save_world: int, loads: list[tuple[int, bool]]) -> None:
        root = self.get_auto_remove_tmp_dir()
        self._spawn(save_world, _ep_save, root, stage, 2)
        for world, load_optimizer_states in loads:
            with self.subTest(world=world, load_optimizer_states=load_optimizer_states):
                self._spawn(world, _ep_load, root, stage, 2, load_optimizer_states)

    def test_expert_parallel_zero2_shrink(self):
        # EP 2 throughout: expert data parallelism 2 -> 1.
        self._check_ep(2, 4, [(2, True), (2, False)])

    def test_expert_parallel_zero1_grow(self):
        self._check_ep(1, 2, [(4, True)])
