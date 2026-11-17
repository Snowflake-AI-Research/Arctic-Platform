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

"""EP export gathers every dtype as exact bytes, including ones the backend has no type for (e8m0, packed fp4)."""

import gc
import os
import tempfile
import weakref
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

pytest.importorskip("deepspeed")

_WORLD = 2
_ROWS, _COLS = 6, 8
_DTYPES = [name for name in ("float8_e8m0fnu", "float4_e2m1fn_x2", "bfloat16", "float32") if hasattr(torch, name)]


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
    group_ref = weakref.ref(dist.group.WORLD)
    group_name = f"ep_size_{_WORLD}"
    previous_gather_bytes = integration._EXPORT_GATHER_BYTES
    try:
        integration._EXPORT_GATHER_BYTES = gather_bytes
        ds_groups._EXPERT_PARALLEL_GROUP[group_name] = dist.group.WORLD
        model = _Model(rank)
        for parameter in model.parameters():
            parameter.group_name = group_name
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
    finally:
        # Otherwise DeepSpeed retains Gloo's threads until interpreter shutdown.
        ds_groups._EXPERT_PARALLEL_GROUP.pop(group_name, None)
        dist.destroy_process_group()
        integration._EXPORT_GATHER_BYTES = previous_gather_bytes
    # An active exception traceback can still retain the iterator's local group.
    gc.collect()
    assert group_ref() is None, "the test-owned Gloo group must be released before interpreter shutdown"


@pytest.mark.parametrize("gather_bytes", [1 << 30, 2 * _COLS * _WORLD], ids=["one_gather", "row_slices"])
def test_ep_export_gathers_backend_unsupported_dtypes_as_bytes(gather_bytes: int) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_worker, args=(f"file://{tmp}/rendezvous", gather_bytes, tmp), nprocs=_WORLD, join=True)
        assert os.path.exists(os.path.join(tmp, "ok"))


def _failed_export_worker(rank: int, rendezvous: str, failure_stage: str) -> None:
    import deepspeed.utils.groups as ds_groups

    import arctic_platform.model.implementations.moe.deepspeed_integration as integration

    group_refs = []
    previous_gather_bytes = integration._EXPORT_GATHER_BYTES

    def fail_export(_model):
        group_refs.append(weakref.ref(dist.group.WORLD))
        raise RuntimeError("injected export failure")

    def fail_gather(_tensor, ep_group, **_kwargs):
        group_refs.append(weakref.ref(ep_group))
        raise RuntimeError("injected export failure")

    with pytest.MonkeyPatch.context() as monkeypatch:
        if failure_stage == "build":
            monkeypatch.setattr(integration, "build_iter_full_hf_weights", fail_export)
        else:
            monkeypatch.setattr(integration, "_gather_ep_weight_to_cpu", fail_gather)
        with pytest.raises(RuntimeError, match="injected export failure"):
            _worker(rank, rendezvous, 32, "")

    assert not dist.is_initialized()
    assert f"ep_size_{_WORLD}" not in ds_groups._EXPERT_PARALLEL_GROUP
    assert integration._EXPORT_GATHER_BYTES == previous_gather_bytes
    gc.collect()
    assert len(group_refs) == 1
    assert group_refs[0]() is None, "release the group after the export exception traceback is released"


@pytest.mark.parametrize("failure_stage", ["build", "gather"])
def test_ep_export_releases_the_process_group_when_export_fails(failure_stage: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mp.spawn(_failed_export_worker, args=(f"file://{tmp}/rendezvous", failure_stage), nprocs=_WORLD, join=True)
