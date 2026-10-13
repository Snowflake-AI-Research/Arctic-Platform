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


@pytest.mark.parametrize("rank", [0, 1])
def test_bf16_gradient_metrics_use_reduced_accumulation_without_mutation(monkeypatch, rank):
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16),
        torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16),
    )
    model[0].weight.grad = torch.ones_like(model[0].weight)
    model[1].weight.grad = torch.full_like(model[1].weight, 4.0)
    gradients = [torch.full((4,), 3.0), torch.full((4,), 5.0)]
    worker = _worker(model, enabled=True, rank=rank)
    worker.engine.optimizer = SimpleNamespace(
        bf16_groups=[[model[1].weight], [model[0].weight]],
        fp32_groups_gradients=[[gradients[1]], [gradients[0]]],
        fp32_groups_has_gradients=[[True], [True]],
    )
    local_before = [parameter.grad.clone() for parameter in model.parameters()]
    buffers_before = [gradient.clone() for gradient in gradients]

    def reject_local_gradient(parameter):
        pytest.fail("BF16 optimizer metrics must not use the local-gradient-first getter")

    monkeypatch.setattr("deepspeed.utils.safe_get_full_grad", reject_local_gradient)
    metrics = worker._per_parameter_gradient_norm_metrics()

    assert metrics == ({"gradient_norms_per_param": {"0.weight": 6.0, "1.weight": 10.0}} if rank == 0 else {})
    for parameter, before in zip(model.parameters(), local_before, strict=True):
        assert torch.equal(parameter.grad, before)
    for gradient, before in zip(gradients, buffers_before, strict=True):
        assert torch.equal(gradient, before)


def test_bf16_gradient_metrics_skip_parameters_without_accumulated_gradients(monkeypatch):
    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    model.weight.grad = torch.ones_like(model.weight)
    worker = _worker(model, enabled=True)
    worker.engine.optimizer = SimpleNamespace(
        bf16_groups=[[model.weight]],
        fp32_groups_gradients=[[torch.ones(4)]],
        fp32_groups_has_gradients=[[False]],
    )
    monkeypatch.setattr("deepspeed.utils.safe_get_full_grad", lambda parameter: parameter.grad)

    assert worker._per_parameter_gradient_norm_metrics() == {"gradient_norms_per_param": {}}


@pytest.mark.parametrize("has_gradient", [False, True])
@pytest.mark.parametrize("local_gradient", [False, True])
def test_optimizer_capture_uses_native_bf16_accumulation_map(monkeypatch, tmp_path, has_gradient, local_gradient):
    import ray

    from arctic_platform.correctness.harness import optimizer_capture

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16),
        torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16),
    )
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 2.0) if local_gradient else None
    gradients = [torch.tensor([3.0, 4.0, 5.0, 6.0]), torch.tensor([7.0, 8.0, 9.0, 10.0])]
    buffers_before = [gradient.clone() for gradient in gradients]
    local_before = [None if parameter.grad is None else parameter.grad.clone() for parameter in model.parameters()]
    monkeypatch.setattr(ray, "remote", lambda worker: worker)
    monkeypatch.setattr(optimizer_capture, "_OPTIMIZER_CAPTURE_WORKER", None)
    actor_class = optimizer_capture._optimizer_capture_worker_class()
    actor = object.__new__(actor_class)
    actor.rank = 0
    actor._optimizer_state_output_dir = str(tmp_path / "capture")
    actor.engine = SimpleNamespace(
        module=model,
        optimizer=SimpleNamespace(
            bf16_groups=[[model[1].weight], [model[0].weight]],
            fp32_groups_gradients=[[gradients[1]], [gradients[0]]],
            fp32_groups_has_gradients=[[has_gradient], [has_gradient]],
            state={},
        ),
        global_steps=0,
        get_lr=lambda: [0.0001],
    )
    actor._full_adam_moments = lambda parameter: (None, None)
    reads = []
    native_map = actor._bf16_accumulated_gradients

    def accumulated():
        reads.append("native_map")
        return native_map()

    def reject_local_gradient(parameter):
        pytest.fail("BF16 optimizer capture must not use the local-gradient-first getter")

    actor._bf16_accumulated_gradients = accumulated
    monkeypatch.setattr("deepspeed.utils.safe_get_full_grad", reject_local_gradient)
    monkeypatch.setattr("deepspeed.utils.safe_get_full_fp32_param", lambda parameter: parameter.detach().float())
    monkeypatch.setattr(optimizer_capture.dist, "barrier", lambda: None)
    if has_gradient:
        actor._snapshot_optimizer_state_parameters()
        for index, expected in enumerate(gradients):
            artifact = torch.load(tmp_path / f"capture/before/{index:06d}.pt", weights_only=True)
            assert artifact["gradient"].dtype == torch.float32
            torch.testing.assert_close(artifact["gradient"], expected.view(2, 2), rtol=0, atol=0)
    elif local_gradient:
        with pytest.raises(RuntimeError, match="no reduced gradient"):
            actor._snapshot_optimizer_state_parameters()
        assert not list((tmp_path / "capture/before").glob("*.pt"))
    else:
        actor._snapshot_optimizer_state_parameters()
        for index in range(2):
            artifact = torch.load(tmp_path / f"capture/before/{index:06d}.pt", weights_only=True)
            assert artifact["gradient"] is None
    assert reads == ["native_map"]
    for parameter, before in zip(model.parameters(), local_before, strict=True):
        if before is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, before, rtol=0, atol=0)
    for gradient, before in zip(gradients, buffers_before, strict=True):
        torch.testing.assert_close(gradient, before, rtol=0, atol=0)


def test_bf16_expert_gradient_metrics_use_reduced_accumulation(monkeypatch):
    model = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
    model.weight.grad = torch.ones_like(model.weight)
    model.weight.group_name = "ep_size_2"
    model.weight.allreduce = False
    worker = _worker(model, enabled=True)
    worker.engine.optimizer = SimpleNamespace(
        bf16_groups=[[model.weight]],
        fp32_groups_gradients=[[torch.tensor([3.0, 4.0, 0.0, 6.0])]],
        fp32_groups_has_gradients=[[True]],
    )
    group = object()
    monkeypatch.setattr("deepspeed.utils.safe_get_full_grad", lambda parameter: parameter.grad)
    monkeypatch.setattr("deepspeed.utils.groups._get_expert_parallel_group", lambda name: group)
    monkeypatch.setattr("arctic_platform.common.deepspeed_worker.dist.get_world_size", lambda **kwargs: 2)

    def all_gather(shards, local, **kwargs):
        assert kwargs["group"] is group
        shards[0].copy_(local)
        shards[1].copy_(torch.tensor([7.0, 8.0]))

    monkeypatch.setattr("arctic_platform.common.deepspeed_worker.dist.all_gather", all_gather)
    assert worker._per_global_expert_gradient_norm_metrics() == {
        "gradient_norms_per_expert": {"weight": [5.0, 6.0, 7.0, 8.0]}
    }


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
