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

"""Correctness-only gradient telemetry exposed by the native DeepSpeed worker."""

from types import SimpleNamespace

import pytest
import torch

from arctic_platform.common import deepspeed_worker as deepspeed_worker_module
from arctic_platform.common.deepspeed_worker import DeepSpeedWorker
from arctic_platform.common.deepspeed_worker import _deepspeed_init_kwargs
from arctic_platform.common.deepspeed_worker import _sync_initial_peft_adapter
from arctic_platform.common.deepspeed_worker import _synchronize_initial_peft_adapter
from arctic_platform.common.deepspeed_worker import _worker_debug_config


def _worker(model: torch.nn.Module, *, enabled: bool, rank: int = 0, sp_size: int = 1):
    worker = object.__new__(DeepSpeedWorker.__ray_metadata__.modified_class)
    worker.rank = rank
    worker.sp_size = sp_size
    worker.engine = SimpleNamespace(module=model)
    worker._gradient_norms_per_param = enabled
    return worker


def test_live_peft_model_exports_adapter_weights_and_metadata(tmp_path):
    import json

    from peft import LoraConfig
    from peft import get_peft_model
    from safetensors.torch import load_file

    model = get_peft_model(
        torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False)),
        LoraConfig(target_modules=["0"], r=1, lora_alpha=1),
    )
    output_dir = tmp_path / "adapter"

    _sync_initial_peft_adapter(model, str(output_dir))

    config = json.loads((output_dir / "adapter_config.json").read_text())
    weights = load_file(output_dir / "adapter_model.safetensors")
    assert config["peft_type"] == "LORA"
    assert weights
    assert all("lora_" in name for name in weights)
    assert not (output_dir / "config.json").exists()
    assert not (output_dir / "model.safetensors").exists()

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_" in name:
                parameter.add_(1)
    _sync_initial_peft_adapter(model, str(output_dir))

    from peft import get_peft_model_state_dict

    restored = get_peft_model_state_dict(model)
    assert set(restored) == set(weights)
    for name, tensor in weights.items():
        torch.testing.assert_close(restored[name], tensor)


def test_live_peft_export_passes_only_trainable_parameters_to_save_pretrained(tmp_path):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.peft_config = {"default": object()}
            self.adapter = torch.nn.Parameter(torch.ones(1))
            self.frozen = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self.saved_state = None

        def save_pretrained(self, adapter_dir, *, safe_serialization, state_dict):
            self.saved_state = state_dict
            output_dir = tmp_path / "adapter"
            output_dir.mkdir()
            (output_dir / "adapter_model.safetensors").touch()
            (output_dir / "adapter_config.json").touch()

    model = Model()

    _sync_initial_peft_adapter(model, str(tmp_path / "adapter"))

    assert model.saved_state is not None
    assert set(model.saved_state) == {"adapter"}
    assert model.saved_state["adapter"].device.type == "cpu"


@pytest.mark.parametrize(
    ("rank", "expected"),
    [
        (0, ["snapshot", "sync", "barrier", "barrier"]),
        (1, ["snapshot", "barrier", "sync", "barrier"]),
    ],
)
def test_initial_peft_adapter_is_created_before_nonzero_ranks_restore(monkeypatch, rank, expected):
    events = []
    monkeypatch.setattr(
        deepspeed_worker_module,
        "_trainable_peft_state_dict",
        lambda model: events.append("snapshot") or {},
    )
    monkeypatch.setattr(deepspeed_worker_module, "_sync_initial_peft_adapter", lambda *args: events.append("sync"))
    monkeypatch.setattr(deepspeed_worker_module.dist, "barrier", lambda: events.append("barrier"))

    _synchronize_initial_peft_adapter(torch.nn.Module(), "/adapter", rank)

    assert events == expected


def test_worker_debug_config_preserves_native_correctness_settings():
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


def test_per_parameter_gradient_norms_do_not_apply_extra_sp_scale(monkeypatch):
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
