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

import pytest
import torch

from arctic_platform.common import deepspeed_worker
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


class _NeverForwardEngine(_Engine):
    def __call__(self, input_ids, **_kwargs):
        raise AssertionError("M2PO with GAS > 1 must fail before model forward")


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


class _SPAccumulationEngine(_Engine):
    def __init__(self):
        super().__init__()
        self.events = []

    def set_gradient_accumulation_boundary(self, value):
        self.events.append(("boundary", value))

    def step(self):
        self.events.append(("step",))
        super().step()


def _sp_execution_worker():
    worker = _worker(_SPAccumulationEngine())
    worker.world_size = 4
    worker.sp_size = 2
    worker.sp_group = object()
    info = {"rank": 0, "world_size": 4, "sp_size": 2, "sp_rank": 0, "sp_ranks": [0, 2]}
    worker.get_sequence_parallel_info = lambda: dict(info)
    request = {
        "batch": [
            {
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "loss_mask": torch.tensor([[1, 0, 0, 0]]),
                "advantages": torch.ones(1, 4),
            },
            {
                "input_ids": torch.tensor([[5, 6, 7, 8]]),
                "attention_mask": torch.ones(1, 4, dtype=torch.long),
                "loss_mask": torch.tensor([[1, 1, 1, 0]]),
                "advantages": torch.ones(1, 4),
            },
        ],
        "meta": {"dp_size": 2},
        "processing": {"loss_fn": "grpo", "post": ["compute_logprobs"], "config": {"loss_agg_mode": "token-mean"}},
    }
    return worker, info, request


@pytest.mark.parametrize("backward", [True, False])
def test_sp_execution_worker_preserves_objective_gas_scales_boundaries_and_steps(monkeypatch, backward):
    worker, info, request = _sp_execution_worker()
    guards, calls = [], []
    monkeypatch.setattr(
        deepspeed_worker, "verify_sequence_parallel_reduction", lambda engine, **kwargs: guards.append(kwargs)
    )

    def selected(engine, batch, meta, processing, device, **kwargs):
        assert engine is worker.engine
        assert meta["dp_size"] == 2
        assert kwargs["sp_info"] == info
        assert kwargs["sp_group"] is worker.sp_group
        assert kwargs["backward"] is backward
        assert kwargs["loss_object"] is not None
        calls.append(kwargs)
        engine.events.append(("model",))
        return {
            "avg_loss": float(2 * len(calls)),
            "metrics": {"token_count": float(batch["loss_mask"].sum())},
            "batch": {"logprobs": batch["input_ids"].float()},
        }

    monkeypatch.setattr(deepspeed_worker, "run_sequence_parallel_grpo", selected)
    result = worker._forward_maybe_backward(request, backward=backward)
    assert [call["loss_scale"] for call in calls] == [0.25, 0.75]
    assert calls[0]["loss_object"] is calls[1]["loss_object"]
    assert result["avg_loss"] == 3.5
    assert result["metrics"]["token_count"] == 4.0
    assert result["batch"]["logprobs"].tolist() == [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]
    assert result["_sequence_parallel_info"] == info
    if backward:
        assert guards == [{"sp_info": info, "sp_group": worker.sp_group}] * 2
        assert worker.engine.events == [
            ("boundary", False),
            ("model",),
            ("step",),
            ("boundary", True),
            ("model",),
        ]
        assert worker.engine.steps == 1
    else:
        assert not guards
        assert worker.engine.events == [("model",), ("model",)]
        assert worker.engine.steps == 0


def test_sp_execution_worker_rejects_unselected_reducer_before_model(monkeypatch):
    worker, _, request = _sp_execution_worker()

    def reject(*args, **kwargs):
        raise ValueError("unselected reducer")

    def unexpected(*args, **kwargs):
        pytest.fail("SP adapter must not run after failed reduction admission")

    monkeypatch.setattr(deepspeed_worker, "verify_sequence_parallel_reduction", reject)
    monkeypatch.setattr(deepspeed_worker, "run_sequence_parallel_grpo", unexpected)
    with pytest.raises(ValueError, match="unselected reducer"):
        worker.forward_backward(request)
    assert not worker.engine.events


def test_sp_execution_worker_rejects_membership_changed_during_execution(monkeypatch):
    worker, info, request = _sp_execution_worker()
    monkeypatch.setattr(deepspeed_worker, "verify_sequence_parallel_reduction", lambda *args, **kwargs: None)

    def selected(*args, **kwargs):
        info["world_size"] = 8
        return {"avg_loss": 0.0, "batch": {}, "metrics": {}}

    monkeypatch.setattr(deepspeed_worker, "run_sequence_parallel_grpo", selected)
    with pytest.raises(RuntimeError, match="membership changed"):
        worker.forward_backward(request)


def test_sp_execution_worker_rechecks_reducer_before_publishing_result(monkeypatch):
    worker, _, request = _sp_execution_worker()
    calls = []

    def verify(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise ValueError("selected reducer changed")

    def selected(*args, **kwargs):
        return {"avg_loss": 0.0, "batch": {}, "metrics": {}}

    monkeypatch.setattr(deepspeed_worker, "verify_sequence_parallel_reduction", verify)
    monkeypatch.setattr(deepspeed_worker, "run_sequence_parallel_grpo", selected)
    with pytest.raises(ValueError, match="selected reducer changed"):
        worker.forward_backward(request)
    assert len(calls) == 2


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


def test_native_worker_requires_explicit_loss_fn_on_backward(monkeypatch):
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

    with pytest.raises(ValueError, match="processing requires 'loss_fn' when backward is not False"):
        worker.forward_backward(request)


def test_native_worker_rejects_split_m2po_before_forward():
    request = {
        "batch": [
            {
                "input_ids": torch.ones(1, 1, dtype=torch.long),
                "attention_mask": torch.ones(1, 1, dtype=torch.long),
            },
            {
                "input_ids": torch.ones(1, 1, dtype=torch.long),
                "attention_mask": torch.ones(1, 1, dtype=torch.long),
            },
        ],
        "meta": {"pad_token_id": 0},
        "processing": {
            "loss_fn": "ap_grpo",
            "config": {"ratio_m2_threshold": 0.1},
        },
    }
    worker = _worker(_NeverForwardEngine())

    with pytest.raises(ValueError, match="requires exactly one synchronized model call"):
        worker.forward_backward(request)


def test_native_worker_rejects_split_ratio_mask_rebalance_before_forward():
    microbatch = {
        "input_ids": torch.ones(1, 1, dtype=torch.long),
        "attention_mask": torch.ones(1, 1, dtype=torch.long),
    }

    def request(flag):
        return {
            "batch": [dict(microbatch), dict(microbatch)],
            "meta": {"pad_token_id": 0},
            "processing": {
                "loss_fn": "ap_grpo",
                "config": {"use_cispo_loss": True, "is_weight_clip_max": 5.0, "ratio_mask_rebalance": flag},
            },
        }

    # Two microbatches are two model calls on this worker: rejected before the engine is ever called.
    worker = _worker(_NeverForwardEngine())
    with pytest.raises(
        ValueError, match=r"ratio_mask_rebalance requires exactly one synchronized model call per worker, got \(2,\)"
    ):
        worker.forward_backward(request(True))

    # With the flag false the same split request is not stopped by that guard (it fails later, on the missing loss
    # context, which these bare microbatches do not carry).
    with pytest.raises(ValueError) as error:
        _worker(_NeverForwardEngine()).forward_backward(request(False))
    assert "requires exactly one synchronized model call" not in str(error.value)


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
