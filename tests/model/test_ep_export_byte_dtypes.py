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

"""EP export gathers dtypes the collective backend has no type for (e8m0 scales, packed fp4) as exact bytes (gloo)."""

import os
import tempfile
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

pytest.importorskip("deepspeed")

_WORLD = 2
_ROWS, _COLS = 6, 8
_DTYPES = [name for name in ("float8_e8m0fnu", "float4_e2m1fn_x2", "bfloat16") if hasattr(torch, name)]


class _Experts(nn.Module):
    def __init__(self, rank: int):
        super().__init__()
        for index, name in enumerate(_DTYPES):
            dtype = getattr(torch, name)
            width = _COLS // torch.empty((), dtype=dtype).element_size()
            raw = torch.arange(_ROWS * _COLS, dtype=torch.uint8).add_(64 * rank + index).view(_ROWS, _COLS)
            self.register_parameter(f"{name}_scale", nn.Parameter(raw.view(dtype).view(_ROWS, width), False))


class _Model(nn.Module):
    def __init__(self, rank: int):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.experts = _Experts(rank)

    def convert_layer_to_hf(self, layer_state, index):
        pass


def _worker(rank: int, rendezvous: str, gather_bytes: int, out_dir: str) -> None:
    import deepspeed.utils.groups as ds_groups

    import arctic_platform.model.implementations.moe.deepspeed_integration as integration

    dist.init_process_group(
        "gloo", init_method=rendezvous, rank=rank, world_size=_WORLD, timeout=timedelta(seconds=120)
    )
    integration._EXPORT_GATHER_BYTES = gather_bytes
    ds_groups._EXPERT_PARALLEL_GROUP[f"ep_size_{_WORLD}"] = dist.group.WORLD
    model = _Model(rank)
    for parameter in model.parameters():
        parameter.group_name = f"ep_size_{_WORLD}"
        parameter.allreduce = False

    exported = dict(integration.build_iter_full_hf_weights(model)(include_skipped=True))
    if rank == 0:
        shards = [_Model(r).model.layers[0].mlp.experts for r in range(_WORLD)]
        assert sorted(exported) == sorted(f"model.layers.0.mlp.experts.{name}_scale" for name in _DTYPES)
        for name in _DTYPES:
            expected = torch.cat([getattr(shard, f"{name}_scale").data.view(torch.uint8) for shard in shards])
            actual = exported[f"model.layers.0.mlp.experts.{name}_scale"]
            assert actual.dtype == getattr(torch, name), name
            assert torch.equal(actual.view(torch.uint8), expected), name
        open(os.path.join(out_dir, "ok"), "w").close()
    else:
        assert exported == {}
    dist.destroy_process_group()


@pytest.mark.parametrize("gather_bytes", [1 << 30, 2 * _COLS * _WORLD], ids=["one_gather", "row_slices"])
def test_ep_export_gathers_backend_unsupported_dtypes_as_bytes(gather_bytes: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(f"file://{tmp}/rendezvous", gather_bytes, tmp), nprocs=_WORLD, join=True)
        assert os.path.exists(os.path.join(tmp, "ok"))
