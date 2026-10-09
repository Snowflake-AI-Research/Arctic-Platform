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

"""Debug diagnostics exposed by the native DeepSpeed worker."""

from types import SimpleNamespace

import pytest
import torch

from arctic_platform.common.deepspeed_worker import DeepSpeedWorker
from arctic_platform.common.deepspeed_worker import _deepspeed_init_kwargs
from arctic_platform.common.deepspeed_worker import _worker_debug_config


def _worker(model: torch.nn.Module, *, enabled: bool, rank: int = 0, sp_size: int = 1):
    worker = object.__new__(DeepSpeedWorker.__ray_metadata__.modified_class)
    worker.rank = rank
    worker.sp_size = sp_size
    worker.engine = SimpleNamespace(module=model)
    worker._gradient_norms_per_param = enabled
    return worker


def test_worker_debug_config_preserves_native_debug_settings():
    debug = _worker_debug_config(
        {
            "full_determinism": True,
            "ds_worker_config": {
                "debug": {
                    "full_determinism_must_comply": False,
                    "gradient_norms_per_param": True,
                }
            },
        }
    )

    assert debug == {
        "full_determinism": True,
        "full_determinism_must_comply": False,
        "gradient_norms_per_param": True,
    }


def test_per_parameter_gradient_norms_are_off_by_default():
    model = torch.nn.Linear(2, 2, bias=False)
    model.weight.grad = torch.ones_like(model.weight)

    assert _worker(model, enabled=False)._per_parameter_gradient_norm_metrics() == {}


def test_per_parameter_gradient_norms_report_trainable_parameters(monkeypatch):
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias=False),
        torch.nn.Linear(2, 2, bias=False),
    )
    model[0].weight.grad = torch.ones_like(model[0].weight)
    model[1].weight.grad = torch.full_like(model[1].weight, 3.0)
    monkeypatch.setattr("deepspeed.utils.safe_get_full_grad", lambda param: param.grad)

    metrics = _worker(model, enabled=True)._per_parameter_gradient_norm_metrics()

    assert metrics["gradient_norms_per_param"] == {
        "0.weight": pytest.approx(2.0),
        "1.weight": pytest.approx(6.0),
    }


def test_per_parameter_gradient_norms_do_not_apply_extra_sp_normalization(monkeypatch):
    model = torch.nn.Linear(2, 2, bias=False)
    model.weight.grad = torch.full_like(model.weight, 4.0)
    monkeypatch.setattr("deepspeed.utils.safe_get_full_grad", lambda param: param.grad)

    metrics = _worker(model, enabled=True, sp_size=8)._per_parameter_gradient_norm_metrics()

    assert metrics["gradient_norms_per_param"] == {"weight": pytest.approx(8.0)}


def test_per_parameter_gradient_norms_are_returned_by_rank_zero_only(monkeypatch):
    model = torch.nn.Linear(2, 2, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    monkeypatch.setattr("deepspeed.utils.safe_get_full_grad", lambda param: param.grad)

    assert _worker(model, enabled=True, rank=1)._per_parameter_gradient_norm_metrics() == {}


def test_manual_learning_rate_updates_every_optimizer_group():
    worker = _worker(torch.nn.Linear(2, 2), enabled=False)
    worker.engine.optimizer = SimpleNamespace(param_groups=[{"lr": 1e-4}, {"lr": 2e-4}])

    worker._set_optimizer_learning_rate(0.01)

    assert [group["lr"] for group in worker.engine.optimizer.param_groups] == [0.01, 0.01]


def test_deepspeed_init_receives_sequence_parallel_mpu():
    model = torch.nn.Linear(2, 2)
    mpu = object()

    kwargs = _deepspeed_init_kwargs(model, {"train_batch_size": 1}, {"mpu": mpu}, has_optimizer=True)

    assert kwargs["mpu"] is mpu
    assert kwargs["model_parameters"] is not None


def test_deepspeed_forward_only_init_omits_optimizer_parameters():
    model = torch.nn.Linear(2, 2)

    kwargs = _deepspeed_init_kwargs(model, {}, {"mpu": None}, has_optimizer=False)

    assert "mpu" not in kwargs
    assert "model_parameters" not in kwargs
