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

"""Objective-owned response reduction in the native DeepSpeed worker."""

from __future__ import annotations

import torch

from arctic_platform.registry import RegistryMeta
from arctic_platform.rl.processors import BaseLoss
from arctic_platform.rl.processors import prepare_request_loss
from arctic_platform.rl.processors.packed_reduction import local_mean_packed_loss_reduction


class _StatefulWeightedLoss(BaseLoss):
    name = "_w08_stateful_weighted"
    instances = 0
    events: list[str] = []

    def __init__(self):
        type(self).instances += 1
        self.ready = False

    def packed_reduction_callback(self, microbatches, config, loss_fn_name):
        assert loss_fn_name == self.name
        assert config == {}
        self.ready = True
        self.events.append("reduction")
        return local_mean_packed_loss_reduction(
            [float(microbatch["reduction_weight"].item()) for microbatch in microbatches]
        )

    def validation_callback(self, context, config):
        assert self.ready
        self.events.append("validation")

    def model_forward_callback(self, model_kwargs, context, config, output_keys):
        assert self.ready
        self.events.append("model")
        output_keys.append("objective_score")

    def loss(self, model_outputs, batch, meta, config, device):
        assert self.ready
        self.events.append("loss")
        return -model_outputs["objective_score"].mean(), {}

    def output_callback(self, model_outputs):
        assert self.ready
        self.events.append("output")
        model_outputs.pop("objective_score")


class _DefaultStatefulWeightedLoss(_StatefulWeightedLoss):
    name = "_w09_default_stateful_weighted"

    def packed_reduction_callback(self, microbatches, config, loss_fn_name):
        assert loss_fn_name == "ap_grpo"
        assert config == {}
        self.ready = True
        self.events.append("reduction")
        return local_mean_packed_loss_reduction(
            [float(microbatch["reduction_weight"].item()) for microbatch in microbatches]
        )


class _ShadowSFTLoss(BaseLoss):
    name = "_w09_shadow_sft"
    calls = 0

    def model_forward_callback(self, model_kwargs, context, config, output_keys):
        output_keys.append("objective_score")

    def loss(self, model_outputs, batch, meta, config, device):
        type(self).calls += 1
        return -model_outputs["objective_score"].mean(), {"shadow_sft": 1.0}

    def output_callback(self, model_outputs):
        model_outputs.pop("objective_score")


class _Engine:
    global_rank = 0

    def __init__(self):
        self.parameter = torch.tensor(-2.0, requires_grad=True)
        self.steps = 0

    def gradient_accumulation_steps(self):
        return 2

    def train(self):
        pass

    def __call__(self, input_ids, **_kwargs):
        return {"logprobs": self.parameter.expand_as(input_ids)}

    def backward(self, loss, scale_wrt_gas=False):
        assert scale_wrt_gas is False
        loss.backward()

    def step(self):
        self.steps += 1


class _ScoreEngine(_Engine):
    def __call__(self, input_ids, **_kwargs):
        return {"objective_score": self.parameter.expand_as(input_ids)}


class _SingleGASScoreEngine(_ScoreEngine):
    def gradient_accumulation_steps(self):
        return 1


def _worker(engine):
    from arctic_platform.common.deepspeed_worker import DeepSpeedWorker

    worker_class = DeepSpeedWorker.__ray_metadata__.modified_class
    worker = object.__new__(worker_class)
    worker.rank = 0
    worker.world_size = 1
    worker.sp_size = 1
    worker.engine = engine
    worker._device = torch.device("cpu")
    worker.cpu_device = torch.device("cpu")
    return worker


def test_native_worker_applies_local_mean_scales_with_one_stateful_loss_object():
    _StatefulWeightedLoss.instances = 0
    _StatefulWeightedLoss.events = []
    request = {
        "batch": [
            {
                "input_ids": torch.ones(1, 1, dtype=torch.long),
                "attention_mask": torch.ones(1, 1, dtype=torch.long),
                "reduction_weight": torch.tensor(3.0),
            },
            {
                "input_ids": torch.ones(1, 1, dtype=torch.long),
                "attention_mask": torch.ones(1, 1, dtype=torch.long),
                "reduction_weight": torch.tensor(1.0),
            },
        ],
        "meta": {"pad_token_id": 0},
        "processing": {
            "loss_fn": _StatefulWeightedLoss.name,
            "config": {},
        },
    }
    worker = _worker(_ScoreEngine())

    response = worker.forward_backward(request)

    assert response["avg_loss"] == 2.0
    assert worker.engine.parameter.grad.item() == -1.0
    assert worker.engine.steps == 1
    assert _StatefulWeightedLoss.instances == 1
    assert _StatefulWeightedLoss.events == [
        "reduction",
        "validation",
        "model",
        "loss",
        "output",
        "validation",
        "model",
        "loss",
        "output",
    ]


def test_native_worker_reuses_implicit_default_loss_across_gas(monkeypatch):
    _DefaultStatefulWeightedLoss.instances = 0
    _DefaultStatefulWeightedLoss.events = []
    monkeypatch.setitem(
        RegistryMeta._registry["BaseLoss"],
        "ap_grpo",
        _DefaultStatefulWeightedLoss,
    )
    request = {
        "batch": [
            {
                "input_ids": torch.ones(1, 1, dtype=torch.long),
                "attention_mask": torch.ones(1, 1, dtype=torch.long),
                "reduction_weight": torch.tensor(3.0),
            },
            {
                "input_ids": torch.ones(1, 1, dtype=torch.long),
                "attention_mask": torch.ones(1, 1, dtype=torch.long),
                "reduction_weight": torch.tensor(1.0),
            },
        ],
        "meta": {"pad_token_id": 0},
        "processing": {"config": {}},
    }
    worker = _worker(_ScoreEngine())

    response = worker.forward_backward(request)

    assert response["avg_loss"] == 2.0
    assert worker.engine.parameter.grad.item() == -1.0
    assert _DefaultStatefulWeightedLoss.instances == 1
    assert _DefaultStatefulWeightedLoss.events == [
        "reduction",
        "validation",
        "model",
        "loss",
        "output",
        "validation",
        "model",
        "loss",
        "output",
    ]


def test_native_worker_honors_class_precedence_for_sft_name(monkeypatch):
    _ShadowSFTLoss.calls = 0
    monkeypatch.setitem(RegistryMeta._registry["BaseLoss"], "sft", _ShadowSFTLoss)
    request = {
        "batch": {
            "input_ids": torch.ones(1, 1, dtype=torch.long),
            "attention_mask": torch.ones(1, 1, dtype=torch.long),
        },
        "meta": {"pad_token_id": 0},
        "processing": {"loss_fn": "sft", "config": {}},
    }
    worker = _worker(_SingleGASScoreEngine())

    response = worker.forward_backward(request)

    assert response["avg_loss"] == 2.0
    assert response["metrics"] == {"shadow_sft": 1.0}
    assert worker.engine.parameter.grad.item() == -1.0
    assert _ShadowSFTLoss.calls == 1


def test_native_worker_sums_globally_normalized_gas_losses_without_rescaling_gradients():
    request = {
        "batch": [
            {
                "input_ids": torch.ones(1, 2, dtype=torch.long),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
                "labels": torch.tensor([[1, -100]], dtype=torch.long),
                "kd_mask": torch.tensor([[1.0, 0.0]]),
            },
            {
                "input_ids": torch.ones(1, 2, dtype=torch.long),
                "attention_mask": torch.ones(1, 2, dtype=torch.long),
                "labels": torch.tensor([[1, -100]], dtype=torch.long),
                "kd_mask": torch.tensor([[1.0, 0.0]]),
            },
        ],
        "meta": {"pad_token_id": 0},
        "processing": {
            "loss_fn": "grouped_distillation",
            "config": {"kd_coef": 0.0},
        },
    }
    prepare_request_loss(request)
    request["meta"]["dp_size"] = 1

    worker = _worker(_Engine())

    response = worker.forward_backward(request)

    assert request["processing"]["config"] == {
        "kd_coef": 0.0,
        "kd_batch_num_tokens": 2.0,
        "dp_size": None,
    }
    assert response["avg_loss"] == 2.0
    assert worker.engine.parameter.grad.item() == -1.0
    assert worker.engine.steps == 1
