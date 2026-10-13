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

"""Rank-local expert weights must survive a replicated DeepSpeed checkpoint."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.common.deepspeed_worker import DeepSpeedWorker
from arctic_platform.testing_utils import torch_assert_equal


class _ReplicatedCheckpointEngine:
    """Model DeepSpeed's one dense model file and separately restored optimizer."""

    def __init__(self, ep_rank):
        self.module = torch.nn.Module()
        self.module.register_parameter("adapter", torch.nn.Parameter(torch.tensor([0.25])))
        self.module.register_parameter(
            "expert", torch.nn.Parameter(torch.full((2, 3), float(ep_rank + 1)), requires_grad=False)
        )
        self.module.expert.allreduce = False
        self.module.expert.group_name = "ep_size_2"
        self.global_steps = 10
        self.has_moe_layers = False
        self.optimizer = SimpleNamespace(master=torch.tensor([0.25]), moment=torch.tensor([0.125]))
        self.ep_rank = ep_rank

    def save_checkpoint(self, path, tag=None):
        tag = tag or f"global_step{self.global_steps}"
        directory = Path(path) / tag
        directory.mkdir(parents=True, exist_ok=True)
        if self.ep_rank == 0:
            torch.save(self.module.state_dict(), directory / "mp_rank_00_model_states.pt")
        (Path(path) / "latest").write_text(tag)

    def load_checkpoint(self, path):
        tag = (Path(path) / "latest").read_text().strip()
        filename = Path(path) / tag / "mp_rank_00_model_states.pt"
        self.module.load_state_dict(torch.load(filename, weights_only=True))
        self.global_steps = 10
        self.optimizer.master.fill_(0.25)
        self.optimizer.moment.fill_(0.125)
        return str(filename), {}


def _worker(engine):
    worker = object.__new__(DeepSpeedWorker.__ray_metadata__.modified_class)
    worker.engine = engine
    return worker


@pytest.fixture
def ep_ranks(monkeypatch):
    import deepspeed.utils.groups as groups

    ep_group, edp_group = object(), object()
    ranks = {"ep": 0, "edp": 0}
    monkeypatch.setattr(groups, "_get_expert_parallel_group", lambda name: ep_group)
    monkeypatch.setattr(groups, "_get_expert_data_parallel_group", lambda name: edp_group)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group: ranks["ep" if group is ep_group else "edp"])
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    return ranks


@pytest.mark.parametrize("trainable_experts", [False, True])
def test_frozen_expert_roundtrip_keeps_rank_local_weights_and_optimizer(tmp_path, ep_ranks, trainable_experts):
    workers = [_worker(_ReplicatedCheckpointEngine(rank)) for rank in range(2)]
    for worker in workers:
        worker.engine.module.expert.requires_grad_(trainable_experts)
    expected = [worker.engine.module.expert.clone() for worker in workers]
    for rank, worker in enumerate(workers):
        ep_ranks["ep"] = rank
        assert worker.save_checkpoint(str(tmp_path))["global_step"] == 10

    for rank, worker in enumerate(workers):
        ep_ranks["ep"] = rank
        worker.engine.module.expert.data.zero_()
        worker.engine.module.adapter.data.zero_()
        worker.engine.optimizer.master.zero_()
        worker.engine.optimizer.moment.zero_()
        assert worker.load_checkpoint(str(tmp_path)) == 10
        torch_assert_equal(worker.engine.module.expert, expected[rank])
        torch_assert_equal(worker.engine.module.adapter, torch.tensor([0.25]))
        torch_assert_equal(worker.engine.optimizer.master, torch.tensor([0.25]))
        torch_assert_equal(worker.engine.optimizer.moment, torch.tensor([0.125]))
        assert worker.engine.has_moe_layers is False


def test_expert_data_parallel_replica_does_not_overwrite_shard(tmp_path, ep_ranks):
    from arctic_platform.common.utils.expert_checkpoint import save_expert_parallel_weights

    model = _ReplicatedCheckpointEngine(0).module
    save_expert_parallel_weights(model, tmp_path)
    ep_ranks["edp"] = 1
    model.expert.data.fill_(99)
    save_expert_parallel_weights(model, tmp_path)

    saved = torch.load(tmp_path / "experts_ep_rank_000.pt", weights_only=True)
    assert set(saved) == {"expert"}
    torch_assert_equal(saved["expert"], torch.ones(2, 3))


def test_dense_checkpoint_needs_no_expert_file_or_collectives(tmp_path, monkeypatch):
    from arctic_platform.common.utils.expert_checkpoint import load_expert_parallel_weights
    from arctic_platform.common.utils.expert_checkpoint import save_expert_parallel_weights

    model = torch.nn.Linear(2, 3)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: pytest.fail("dense checkpoint is not collective"))
    save_expert_parallel_weights(model, tmp_path)
    load_expert_parallel_weights(model, tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_missing_expert_file_rejects_corrupt_dense_only_restore(tmp_path, ep_ranks):
    worker = _worker(_ReplicatedCheckpointEngine(0))
    worker.engine.save_checkpoint(str(tmp_path))
    with pytest.raises(FileNotFoundError, match="experts_ep_rank_000"):
        worker.load_checkpoint(str(tmp_path))


@pytest.mark.parametrize("corruption", ["missing", "unexpected", "shape", "dtype"])
def test_invalid_shard_is_rejected_before_parameter_mutation(tmp_path, ep_ranks, corruption):
    from arctic_platform.common.utils.expert_checkpoint import load_expert_parallel_weights

    model = _ReplicatedCheckpointEngine(0).module
    model.register_parameter("second_expert", torch.nn.Parameter(torch.ones(2), requires_grad=False))
    model.second_expert.allreduce = False
    model.second_expert.group_name = "ep_size_2"
    shard = {"expert": torch.full((2, 3), 9.0), "second_expert": torch.ones(2)}
    if corruption == "missing":
        del shard["second_expert"]
    elif corruption == "unexpected":
        shard["extra"] = torch.ones(1)
    elif corruption == "shape":
        shard["second_expert"] = torch.ones(1)
    else:
        shard["second_expert"] = torch.ones(2, dtype=torch.float64)
    torch.save(shard, tmp_path / "experts_ep_rank_000.pt")
    before = {name: parameter.clone() for name, parameter in model.named_parameters()}

    with pytest.raises(ValueError, match="expert parameter"):
        load_expert_parallel_weights(model, tmp_path)
    for name, parameter in model.named_parameters():
        torch_assert_equal(parameter, before[name])


def test_missing_dense_checkpoint_preserves_zero_return_without_side_load(tmp_path):
    worker = _worker(SimpleNamespace(load_checkpoint=lambda path: (None, None)))
    assert worker.load_checkpoint(str(tmp_path)) == 0


def test_multiple_expert_groups_are_rejected(tmp_path, ep_ranks):
    from arctic_platform.common.utils.expert_checkpoint import save_expert_parallel_weights

    model = _ReplicatedCheckpointEngine(0).module
    model.adapter.allreduce = False
    model.adapter.group_name = "ep_size_4"
    with pytest.raises(ValueError, match="one EP group"):
        save_expert_parallel_weights(model, tmp_path)
