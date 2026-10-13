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

"""DeepSpeed precision settings must preserve explicit overrides and worker opt-ins."""

from types import SimpleNamespace

import pytest
import torch
from deepspeed.runtime import torch_autocast
from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer

from arctic_platform.common.deepspeed_worker import DeepSpeedWorker


def training_config(ds_config=None, worker_config=None):
    worker = object.__new__(DeepSpeedWorker.__ray_metadata__.modified_class)
    return worker.ds_training_config({}, ds_config or {}, worker_config or {})


@pytest.mark.parametrize("worker_config", [{}, {"use_autocast": False}])
def test_training_autocast_is_opt_in(worker_config):
    config = training_config(worker_config=worker_config)

    assert not (config.get("torch_autocast") or {}).get("enabled", False)
    assert config["bf16"] == {"enabled": True}
    assert config["communication_data_type"] == "fp32"
    assert config["data_types"] == {"grad_accum_dtype": "fp32"}


def test_training_autocast_worker_opt_in():
    assert training_config(worker_config={"use_autocast": True})["torch_autocast"] == {
        "enabled": True,
        "dtype": "bfloat16",
    }


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("worker_opt_in", [False, True])
def test_explicit_training_autocast_overrides_worker_default(enabled, worker_opt_in):
    autocast = {
        "enabled": enabled,
        "dtype": "float16",
        "lower_precision_safe_modules": ["torch.nn.Linear"],
    }
    config = training_config(
        ds_config={
            "torch_autocast": autocast.copy(),
            "communication_data_type": "fp16",
            "data_types": {"grad_accum_dtype": "fp16"},
        },
        worker_config={"use_autocast": worker_opt_in},
    )

    assert config["torch_autocast"] == autocast
    assert config["communication_data_type"] == "fp16"
    assert config["data_types"] == {"grad_accum_dtype": "fp16"}


def test_default_training_preserves_fp32_zero_communication(monkeypatch):
    config = training_config()
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(8, dtype=torch.bfloat16))
    optimizer = object.__new__(DeepSpeedZeroOptimizer)
    optimizer.communication_data_type = torch.float32
    monkeypatch.setattr(torch_autocast, "TORCH_AUTOCAST_INITIALIZED", False)
    monkeypatch.setattr(torch_autocast, "TORCH_AUTOCAST_DTYPE", None)
    autocast = config.get("torch_autocast") or {}
    if autocast.get("enabled", False):
        engine = SimpleNamespace(module=model, zero_quantized_weights=lambda: False)
        torch_autocast.init_autocast_params(
            engine,
            getattr(torch, autocast["dtype"]),
            autocast.get("lower_precision_safe_modules"),
        )

    assert optimizer.get_param_comm_dtype(model.weight) == torch.float32
