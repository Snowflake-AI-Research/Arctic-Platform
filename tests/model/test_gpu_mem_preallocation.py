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

"""CPU tests of capacity-guard configuration and allocation, with CUDA calls mocked."""

import logging

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.model.implementations.debug.gpu_memory import allocate_gpu_mem_preallocation
from arctic_platform.model.implementations.debug.gpu_memory import gpu_mem_preallocate_bytes
from arctic_platform.model.implementations.debug.gpu_memory import gpu_memory_debug_config


@pytest.mark.parametrize(
    "config,expected",
    [
        ({}, 0),
        ({"debug": None}, 0),
        ({"debug": {"gpu_mem_preallocate_gib": 0}}, 0),
        ({"debug": {"gpu_mem_preallocate_gib": 20}}, 20 << 30),
        ({"debug": {"gpu_mem_preallocate_gib": 0.5}}, 1 << 29),
        ({"debug": {"gpu_mem_preallocate_gib": 0.1 / (1 << 30)}}, 1),
        ({"prime_rl": {"debug": {"gpu_mem_preallocate_gib": 2}}}, 2 << 30),
        ({"debug": {}, "prime_rl": {"debug": {"gpu_mem_preallocate_gib": 2}}}, 0),
    ],
)
def test_size_and_legacy_precedence(config, expected):
    assert gpu_mem_preallocate_bytes(config) == expected


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        -1,
        -0.1,
        "20",
        None,
        [],
        float("nan"),
        float("inf"),
        10**400,
        1e308,
    ],
)
def test_invalid_sizes_are_rejected(value):
    with pytest.raises(ValueError, match="gpu_mem_preallocate_gib"):
        gpu_mem_preallocate_bytes({"debug": {"gpu_mem_preallocate_gib": value}})


def test_only_capacity_guard_reaches_model_spec():
    assert gpu_memory_debug_config({"debug": {"random_init": True, "gpu_mem_preallocate_gib": 20}}) == {
        "gpu_mem_preallocate_gib": 20,
    }


def test_disabled_buffer_never_touches_cuda(monkeypatch):
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: pytest.fail("disabled allocation"))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: pytest.fail("disabled CUDA access"))
    assert allocate_gpu_mem_preallocation({}) is None


def test_byte_tensor_allocates_on_current_rank_device(monkeypatch, caplog, capsys):
    allocation = object()
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    def allocate(size, *, dtype, device):
        assert size == 20 << 30
        assert dtype == torch.uint8
        assert device == torch.device("cuda", 3)
        return allocation

    monkeypatch.setattr(torch, "empty", allocate)
    with caplog.at_level(logging.INFO, logger="arctic_platform.model.implementations.debug.gpu_memory"):
        assert allocate_gpu_mem_preallocation({"gpu_mem_preallocate_gib": 20}) is allocation
    assert "rank=7 [gpu-mem-preallocate] bytes=21474836480 size=20.000GiB" in caplog.text
    assert capsys.readouterr().out == ""


def test_oom_is_not_hidden(monkeypatch):
    def oom(*args, **kwargs):
        raise torch.OutOfMemoryError("reserve cannot fit")

    monkeypatch.setattr(torch, "empty", oom)
    with pytest.raises(torch.OutOfMemoryError, match="cannot fit"):
        allocate_gpu_mem_preallocation({"gpu_mem_preallocate_gib": 20}, device=torch.device("cuda", 0))


@pytest.fixture
def stub_loader(monkeypatch):
    from arctic_platform.model import loader
    from arctic_platform.model.loader import LoadedModel
    from arctic_platform.model.loader import register_loader

    monkeypatch.setattr(loader, "_LOADERS", dict(loader._LOADERS))
    model = torch.nn.Linear(1, 1)

    @register_loader("capacity_guard_test")
    def load(ctx):
        return LoadedModel(model=model)

    return model


def test_model_spec_lifts_only_legacy_capacity_guard(stub_loader):
    from arctic_platform.model import ModelSpec

    options = {"debug": {"gpu_mem_preallocate_gib": 20, "random_init": True}}
    original = deepcopy(options)
    spec = ModelSpec(model_path_or_name="x", loader="capacity_guard_test", loader_options=options)
    assert options == original
    assert spec.debug == {"gpu_mem_preallocate_gib": 20}
    assert spec.loader_options == {"debug": {"random_init": True}}
    assert ModelSpec.model_validate_json(spec.model_dump_json()).debug == spec.debug


@pytest.mark.parametrize(
    "top",
    [{}, {"gradient_norms_per_param": True}, {"gpu_mem_preallocate_gib": 2}, {"gpu_mem_preallocate_gib": 0}],
)
def test_model_spec_only_explicit_reserve_overrides_legacy_guard(stub_loader, top):
    from arctic_platform.model import ModelSpec

    options = {"debug": {"gpu_mem_preallocate_gib": 20, "random_init": True}}
    original = deepcopy(options)
    original_top = deepcopy(top)
    spec = ModelSpec(
        model_path_or_name="x",
        loader="capacity_guard_test",
        debug=top,
        loader_options=options,
    )
    assert spec.debug == {"gpu_mem_preallocate_gib": 20, **top}
    assert spec.loader_options["debug"] == {"random_init": True}
    assert options == original
    assert top == original_top
    assert ModelSpec.model_validate_json(spec.model_dump_json()).debug == spec.debug


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("top", [None, {}])
def test_model_spec_rejects_invalid_guard(stub_loader, legacy, top):
    from arctic_platform.model import ModelSpec

    controls = (
        {"loader_options": {"debug": {"gpu_mem_preallocate_gib": -1}}}
        if legacy
        else {
            "debug": {"gpu_mem_preallocate_gib": -1},
        }
    )
    if legacy and top is not None:
        controls["debug"] = top
    with pytest.raises(ValueError, match="gpu_mem_preallocate_gib"):
        ModelSpec(model_path_or_name="x", loader="capacity_guard_test", **controls)


def test_guard_is_removed_before_strict_moe_debug_validation():
    pytest.importorskip("transformers.models.qwen3_5_moe")
    from arctic_platform.model import ModelSpec

    spec = ModelSpec(
        model_path_or_name="x",
        loader="qwen4_exp",
        parallelism={"expert_parallel": 8, "sequence_parallel": 16},
        loader_options={"debug": {"gpu_mem_preallocate_gib": 20, "random_init": True}},
    )
    assert spec.debug == {"gpu_mem_preallocate_gib": 20}
    assert spec.loader_options["debug"]["random_init"] is True
    assert "gpu_mem_preallocate_gib" not in spec.loader_options["debug"]


def test_ds_worker_config_forwards_only_capacity_guard():
    from arctic_platform.model import ModelSpec

    spec = ModelSpec.from_ds_worker_config(
        "x",
        {
            "attn_implementation": "sdpa",
            "debug": {"gpu_mem_preallocate_gib": 20, "gradient_norms_per_param": True},
        },
    )
    assert spec.debug == {"gpu_mem_preallocate_gib": 20}


def test_factory_reserves_before_loading_and_retains_on_final_model(stub_loader, monkeypatch):
    import weakref

    from arctic_platform.model import ModelSpec
    from arctic_platform.model import build_model
    from arctic_platform.model import factory

    events = []
    references = []
    final_model = torch.nn.Sequential(stub_loader)

    def allocate(debug):
        assert debug == {"gpu_mem_preallocate_gib": 20}
        events.append("reserve")
        tensor = torch.empty(1)
        references.append(weakref.ref(tensor))
        return tensor

    def select(ctx):
        assert references[0]() is not None
        events.append("load")
        return lambda ctx: SimpleNamespace(model=stub_loader)

    def patch(loaded, ctx):
        assert references[0]() is not None
        events.append("patch")
        loaded.model = final_model

    def freeze(model):
        assert model is final_model
        assert references[0]() is not None
        events.append("freeze")

    monkeypatch.setattr(factory, "allocate_gpu_mem_preallocation", allocate)
    monkeypatch.setattr(factory, "select_loader", select)
    monkeypatch.setattr(factory, "apply_patches", patch)
    monkeypatch.setattr(factory, "freeze_fp8_weights_and_scales", freeze)
    loaded = build_model(
        ModelSpec(
            model_path_or_name="x",
            loader="capacity_guard_test",
            debug={"gpu_mem_preallocate_gib": 20},
        )
    )
    assert events == ["reserve", "load", "patch", "freeze"]
    assert loaded.model._arctic_gpu_mem_preallocation is references[0]()
    assert "_arctic_gpu_mem_preallocation" not in loaded.model.state_dict()
    assert "_arctic_gpu_mem_preallocation" not in dict(loaded.model.named_buffers())
    loaded.model.to(dtype=torch.float64)
    assert references[0]().dtype == torch.float32
    del loaded
    final_model = None
    assert references[0]() is None


def test_factory_oom_prevents_model_load(stub_loader, monkeypatch):
    from arctic_platform.model import ModelSpec
    from arctic_platform.model import build_model
    from arctic_platform.model import factory

    def fail(debug):
        raise torch.OutOfMemoryError("reserve cannot fit")

    monkeypatch.setattr(factory, "allocate_gpu_mem_preallocation", fail)
    monkeypatch.setattr(factory, "select_loader", lambda ctx: pytest.fail("loader ran after reserve OOM"))
    with pytest.raises(torch.OutOfMemoryError, match="reserve cannot fit"):
        build_model(
            ModelSpec(
                model_path_or_name="x",
                loader="capacity_guard_test",
                debug={"gpu_mem_preallocate_gib": 20},
            )
        )
