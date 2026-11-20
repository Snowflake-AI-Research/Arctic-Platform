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
"""CPU tests of the ``tabular`` loader on a Kumo Tabular snapshot built from a randomly initialized network."""

from __future__ import annotations

import json

import pytest
import torch

pytest.importorskip("sdm")

from arctic_platform.model import ModelSpec  # noqa: E402
from arctic_platform.model import ParallelismConfig  # noqa: E402
from arctic_platform.model import Patches  # noqa: E402
from arctic_platform.model import build_model  # noqa: E402
from arctic_platform.model.loaders.tabular import TabularOptions  # noqa: E402
from arctic_platform.model.loaders.tabular import build_network  # noqa: E402
from arctic_platform.model.loaders.tabular import convert_to_safetensors  # noqa: E402
from arctic_platform.testing_utils import set_seed  # noqa: E402
from arctic_platform.testing_utils import torch_assert_equal  # noqa: E402


def write_snapshot(root, task="classification"):
    """An ``nvidia/Kumo-Tabular``-shaped snapshot holding one small checkpoint, as the vendor ships it (.pt)."""
    set_seed(0)
    network = build_network(TabularOptions(task=task, size="small"))
    filename = "small/classifier.pt" if task == "classification" else "small/regressor.pt"
    (root / "small").mkdir(parents=True)
    torch.save(network.state_dict(), root / filename)
    (root / "config.json").write_text(
        json.dumps({"model_type": "kumo_tabular", "checkpoint_files": {"small": {task: filename}}})
    )
    return network


def tabular_spec(path, task="classification", **overrides):
    return ModelSpec(
        model_path_or_name=str(path),
        dtype="float32",
        loader="tabular",
        loader_options={"task": task, "size": "small"},
        **overrides,
    )


def table(rows=12, context=8, columns=5):
    x = torch.randn(rows, columns)
    y = torch.randint(0, 3, (context,))
    return x, y, torch.zeros(columns, dtype=torch.bool)


@pytest.mark.parametrize("task", ["classification", "regression"])
def test_loads_the_vendor_checkpoint(tmp_path, task):
    reference = write_snapshot(tmp_path, task)
    model = build_model(tabular_spec(tmp_path, task)).model
    for name, value in reference.state_dict().items():
        torch_assert_equal(model.state_dict()[name], value, msg=name)

    x, y, mask = table()
    if task == "regression":
        y = y.float()
    outputs = model(x=x, y=y, categorical_mask=mask)
    assert outputs.shape == (4, 10 if task == "classification" else 999)


def test_converted_snapshot_loads_without_pickles(tmp_path):
    write_snapshot(tmp_path / "vendor")
    convert_to_safetensors(tmp_path / "vendor", tmp_path / "converted")

    config = json.loads((tmp_path / "converted" / "config.json").read_text())
    assert config["checkpoint_files"]["small"]["classification"] == "small/classifier.safetensors"
    assert not list((tmp_path / "converted").rglob("*.pt"))
    vendor = build_model(tabular_spec(tmp_path / "vendor")).model
    converted = build_model(tabular_spec(tmp_path / "converted")).model
    for name, value in vendor.state_dict().items():
        torch_assert_equal(converted.state_dict()[name], value, msg=name)


def test_conversion_of_selected_sizes_lists_only_those(tmp_path):
    write_snapshot(tmp_path / "vendor")
    convert_to_safetensors(tmp_path / "vendor", tmp_path / "converted", sizes=("small",))
    config = json.loads((tmp_path / "converted" / "config.json").read_text())
    assert list(config["checkpoint_files"]) == ["small"]
    with pytest.raises(ValueError, match="large"):
        convert_to_safetensors(tmp_path / "vendor", tmp_path / "again", sizes=("large",))


def test_rejects_a_config_of_another_model_type(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    with pytest.raises(ValueError, match="kumo_tabular"):
        build_model(tabular_spec(tmp_path))


def test_rejects_a_missing_size_or_task(tmp_path):
    write_snapshot(tmp_path, "classification")
    with pytest.raises(ValueError, match="small/regression"):
        build_model(tabular_spec(tmp_path, "regression"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"patches": Patches(liger=True)},
        {"parallelism": ParallelismConfig(sequence_parallel=2)},
        {"dtype": "float16"},
    ],
)
def test_rejects_what_the_network_cannot_take(tmp_path, overrides):
    from arctic_platform.model.loader import validate_loader_spec

    write_snapshot(tmp_path)
    spec = tabular_spec(tmp_path).model_copy(update=overrides)
    with pytest.raises(ValueError, match="tabular loader"):
        validate_loader_spec("tabular", spec)
