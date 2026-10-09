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

from __future__ import annotations

import torch
from torch import nn

from arctic_platform.common import deepspeed_worker
from arctic_platform.common.deepspeed_worker import _canonical_hf_export_state_dict
from arctic_platform.common.deepspeed_worker import _gather_live_hf_export_state_dict


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(3, 2, bias=False)


class _CheckpointWrappedBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self._checkpoint_wrapped_module = _Block()


class _Model(nn.Module):
    def __init__(self, *, checkpoint_wrapped: bool) -> None:
        super().__init__()
        block = _CheckpointWrappedBlock() if checkpoint_wrapped else _Block()
        self.layers = nn.ModuleList([block])

    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> bool:
        return False

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raise AssertionError("an HF-format state dict must not be converted")


def test_hf_export_state_dict_matches_unwrapped_model_names() -> None:
    wrapped = _Model(checkpoint_wrapped=True)
    canonical = _Model(checkpoint_wrapped=False)

    exported = _canonical_hf_export_state_dict(wrapped, wrapped.state_dict())

    assert set(exported) == set(canonical.state_dict())
    assert "layers.0.projection.weight" in exported
    assert all("_checkpoint_wrapped_module" not in name for name in exported)
    torch.testing.assert_close(
        exported["layers.0.projection.weight"], wrapped.layers[0]._checkpoint_wrapped_module.projection.weight
    )


class _PrimeModel:
    @classmethod
    def is_prime_state_dict(cls, state_dict: dict[str, torch.Tensor]) -> bool:
        return any(".mlp.router.gate.weight" in name for name in state_dict)

    @classmethod
    def convert_to_hf(cls, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        old = "model.layers.0.mlp.router.gate.weight"
        state_dict["model.layers.0.mlp.gate.weight"] = state_dict.pop(old)
        return state_dict


def test_hf_export_converts_prime_names_after_unwrapping() -> None:
    tensor = torch.ones(1)
    exported = _canonical_hf_export_state_dict(
        _PrimeModel(),
        {"model.layers.0._checkpoint_wrapped_module.mlp.router.gate.weight": tensor},
    )

    assert exported == {"model.layers.0.mlp.gate.weight": tensor}


class _ExpertModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = nn.Parameter(torch.tensor([[1.0], [2.0]]))
        self.experts.group_name = "ep"
        self.experts.allreduce = False
        self.shared = nn.Parameter(torch.tensor([3.0]))


def test_live_export_gathers_expert_parallel_shards(monkeypatch) -> None:
    import deepspeed.utils.groups as ds_groups

    group = object()
    monkeypatch.setattr(ds_groups, "_get_expert_parallel_group", lambda name: group)
    monkeypatch.setattr(deepspeed_worker.dist, "get_world_size", lambda *, group: 2)

    def all_gather(shards, local, *, group) -> None:
        shards[0].copy_(local)
        shards[1].copy_(local + 10)

    monkeypatch.setattr(deepspeed_worker.dist, "all_gather", all_gather)

    state_dict = _gather_live_hf_export_state_dict(_ExpertModel(), rank=0)

    torch.testing.assert_close(state_dict["experts"], torch.tensor([[1.0], [2.0], [11.0], [12.0]]))
    torch.testing.assert_close(state_dict["shared"], torch.tensor([3.0]))
