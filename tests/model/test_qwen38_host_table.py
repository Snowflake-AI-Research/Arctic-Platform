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

import pytest
import torch
import torch.nn.functional as F

from arctic_platform.model.implementations.qwen38 import host_table

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@requires_cuda
@pytest.mark.parametrize("dim", [160, 7, 256])
def test_gather_from_pinned_host_matches_embedding(dim):
    torch.manual_seed(0)
    weight = torch.randn(10_000, dim, dtype=torch.bfloat16, device="cuda")
    table = host_table.move_to_host(weight)
    ids = torch.randint(0, weight.shape[0], (3, 4096, 16), device="cuda")
    got = host_table.gather_rows(table, ids)
    assert got.device == ids.device and got.shape == (3, 4096, 16, dim)
    assert torch.equal(got, F.embedding(ids, weight))


@requires_cuda
def test_out_of_range_ids_are_refused():
    table = host_table.move_to_host(torch.zeros(8, 4, dtype=torch.bfloat16))
    with pytest.raises(IndexError):
        host_table.gather_rows(table, torch.tensor([0, 8], device="cuda"))


@pytest.fixture
def _no_host_table_setting(monkeypatch):
    monkeypatch.delenv(host_table.HOST_TABLE_ENV, raising=False)
    yield
    host_table.configure(None)


def test_the_loader_option_wins_and_the_environment_is_the_fallback(monkeypatch, _no_host_table_setting):
    assert host_table.host_table_enabled() is False
    monkeypatch.setenv(host_table.HOST_TABLE_ENV, "1")
    assert host_table.host_table_enabled() is True
    host_table.configure(False)
    assert host_table.host_table_enabled() is False
    host_table.configure(None)
    monkeypatch.delenv(host_table.HOST_TABLE_ENV)
    host_table.configure(True)
    assert host_table.host_table_enabled() is True


def test_only_the_qwen38_loader_takes_ngram_table_on_host():
    from pydantic import ValidationError

    from arctic_platform.model import loader as loader_mod

    assert loader_mod._LOADERS["qwen4_exp"].options(ngram_table_on_host=True).ngram_table_on_host is True
    assert loader_mod._LOADERS["qwen4_exp"].options().ngram_table_on_host is None
    for name in ("glm5_next", "generic_moe", "qwen3_5_moe"):
        with pytest.raises(ValidationError):
            loader_mod._LOADERS[name].options(ngram_table_on_host=True)


def test_the_qwen38_loader_configures_the_host_table_from_the_spec(monkeypatch, tmp_path, _no_host_table_setting):
    import json

    from torch import nn

    from arctic_platform.model import ModelSpec
    from arctic_platform.model import ParallelismConfig
    from arctic_platform.model import build_model
    from arctic_platform.model.implementations.qwen38 import deepspeed_integration as integration

    seen = {}

    def load(*args, **kwargs):
        seen.update(kwargs)
        seen["enabled"] = host_table.host_table_enabled()
        return nn.Identity()

    monkeypatch.setattr(integration.qwen_ds, "_load_moe_model", load)
    monkeypatch.setattr(integration, "_validate_parallelism", lambda sp_size, sp_group: None)
    monkeypatch.setattr(integration.qwen_ds, "maybe_apply_row_invariant_projections", lambda model, options: None)
    path = tmp_path / "qwen38"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp"}))
    spec = ModelSpec(
        model_path_or_name=str(path),
        parallelism=ParallelismConfig(expert_parallel=2),
        loader_options={"ngram_table_on_host": True},
    )
    assert spec.loader == "qwen4_exp"
    build_model(spec, parallel_groups={"ep_group": object()})
    assert seen["options"].ngram_table_on_host is True and seen["enabled"] is True
