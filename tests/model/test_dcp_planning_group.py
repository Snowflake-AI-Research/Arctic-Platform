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
"""The HF DCP loads of the carved MoE builders (qwen35) plan over one gloo group
(moe/weights.py ``dcp_planning_group``), not the NCCL world group. CPU only: two ranks on a gloo world load a small
safetensors checkpoint through each builder's ``load_dcp_from_hf``."""

from __future__ import annotations

import importlib
import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from safetensors.torch import save_file
from torch import nn

from arctic_platform.model.implementations.moe import weights

BUILDERS = ("arctic_platform.model.implementations.qwen35.model_builder",)


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(tie_word_embeddings=False)
        self.proj = nn.Linear(4, 3, bias=False)
        self.norm = nn.LayerNorm(3)


def _reference() -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {name: torch.randn_like(t) for name, t in _Tiny().state_dict().items()}


def test_the_planning_group_is_one_gloo_group(monkeypatch):
    calls = []
    monkeypatch.setattr(weights, "_DCP_PLANNING_GROUP", None)
    monkeypatch.setattr(weights.torch.distributed, "new_group", lambda **kwargs: calls.append(kwargs) or object())
    group = weights.dcp_planning_group()
    assert weights.dcp_planning_group() is group
    assert calls == [{"backend": "gloo"}]


def _load_worker(rank: int, world: int, port: int, checkpoint: str, results) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    seen = {}
    for name in BUILDERS:
        builder = importlib.import_module(name)
        real_load = builder.dcp_load

        def recording_load(state_dict, *, process_group=None, _real=real_load, _name=name, **kwargs):
            planning = process_group is not None and process_group is weights.dcp_planning_group()
            seen[_name] = (planning, dist.get_backend(process_group))
            return _real(state_dict, process_group=process_group, **kwargs)

        builder.dcp_load = recording_load
        model = _Tiny().to("meta")
        config = SimpleNamespace(fsdp_cpu_offload=True, debug=SimpleNamespace(random_init=False), name=checkpoint)
        builder.load_dcp_from_hf(model, config, parallel_dims=None)
        loaded = {key: value.clone() for key, value in model.state_dict().items()}
        reference = _reference()
        seen[name] += (all(torch.equal(loaded[key], reference[key]) for key in reference),)
    results[rank] = seen
    dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.timeout(300)
def test_the_qwen_dcp_load_plans_over_the_gloo_group(tmp_path):
    save_file(_reference(), str(tmp_path / "model.safetensors"), metadata={"format": "pt"})
    world = 2
    results = mp.Manager().dict()
    mp.spawn(_load_worker, args=(world, _free_port(), str(tmp_path), results), nprocs=world, join=True)
    for rank in range(world):
        assert results[rank] == {name: (True, "gloo", True) for name in BUILDERS}, (rank, results[rank])
