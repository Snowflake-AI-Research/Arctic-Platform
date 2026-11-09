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

"""CPU routing controls for essential initialization and weight-sender observation."""

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.common import deepspeed_engine as engine_module
from arctic_platform.common import deepspeed_worker as worker_module
from arctic_platform.common.optimizer import OptimizerSettings
from arctic_platform.debug import weight_sync

Worker = worker_module.DeepSpeedWorker.__ray_metadata__.modified_class


@pytest.fixture(autouse=True)
def debug_disabled(monkeypatch):
    monkeypatch.delenv(weight_sync.VERIFY_ENV, raising=False)


def worker():
    selected = object.__new__(Worker)
    selected.engine = SimpleNamespace(module=torch.nn.Linear(2, 1, bias=False))
    selected._weight_sender = None
    selected._device = "cpu"
    return selected


@pytest.mark.parametrize("prebuilt", [False, True])
def test_disabled_engine_hooks_preserve_native_arguments_and_returns(monkeypatch, prebuilt):
    model = torch.nn.Linear(2, 1)
    config = {"train_batch_size": 1, "optimizer": {"type": "AdamW", "params": {"lr": 0.01}}}
    built, native_engine, mpu = object(), object(), object()
    calls = []
    monkeypatch.setattr(engine_module, "preserve_fp8_params_through_ds_cast", lambda model: (0, 0))
    monkeypatch.setattr(engine_module, "build_training_optimizer", lambda *args, **kwargs: built)
    monkeypatch.setattr(engine_module, "optimizer_choice", lambda *args, **kwargs: {"implementation": "test"})

    def initialize(**kwargs):
        calls.append(kwargs)
        return native_engine, built, None, None

    monkeypatch.setattr(worker_module.deepspeed, "initialize", initialize)
    result = engine_module.initialize_training_engine(
        model,
        config,
        optimizer=OptimizerSettings(lr=0.01) if prebuilt else None,
        ep_size=1,
        mpu=mpu,
        model_path="dense",
        rank=0,
    )
    assert result[0] is native_engine and result[1] is built
    assert result[2] == ({"implementation": "test"} if prebuilt else None)
    assert len(calls) == 1 and calls[0]["config"] is config
    assert calls[0]["model"] is model and calls[0]["mpu"] is mpu
    if prebuilt:
        assert calls[0]["optimizer"] is built and "model_parameters" not in calls[0]
    else:
        assert list(calls[0]["model_parameters"]) == list(model.parameters())
        assert "optimizer" not in calls[0]


def test_trace_context_encloses_native_initialize(monkeypatch):
    events = []
    config, native = {"train_batch_size": 1}, {"train_batch_size": 1}
    monkeypatch.setattr(engine_module, "preserve_fp8_params_through_ds_cast", lambda model: (0, 0))

    @contextmanager
    def trace(value, rank):
        assert value is config and rank == 3
        events.append("trace-enter")
        yield native
        events.append("trace-exit")

    def initialize(**kwargs):
        assert kwargs["config"] is native
        events.append("initialize")
        return object(), None, None, None

    monkeypatch.setattr(engine_module, "maybe_trace_zero_initialization", trace)
    monkeypatch.setattr(worker_module.deepspeed, "initialize", initialize)
    engine_module.initialize_training_engine(
        torch.nn.Linear(2, 1), config, optimizer=None, ep_size=1, mpu=None, model_path="dense", rank=3
    )
    assert events == ["trace-enter", "initialize", "trace-exit"]


def test_uninitialized_sender_still_gets_weights(monkeypatch):
    selected, calls = worker(), []
    monkeypatch.setattr(selected, "get_weights", lambda: calls.append("gather") or [])
    assert selected.send_weights() == {"status": "not_initialized"}
    assert calls == ["gather"]


class Sender:
    def __init__(self):
        self.calls = []
        self.result = {"status": "ok"}

    def send(self, weights):
        self.calls.append(weights)
        return self.result


def test_disabled_send_keeps_transport_result_and_exact_tensor_list(monkeypatch):
    selected, sender = worker(), Sender()
    selected._weight_sender = sender
    weights = selected.get_weights()
    monkeypatch.setattr(selected, "get_weights", lambda: weights)
    result = selected.send_weights()
    assert result is sender.result
    assert sender.calls == [weights]
    assert result["sent_l2_sq"] == selected._weights_l2_sq(weights)
    assert set(result) == {"status", "sent_l2_sq"}


def test_enabled_send_observation_hashes_actual_transmitted_tensors(monkeypatch):
    selected, sender = worker(), Sender()
    selected._weight_sender = sender
    monkeypatch.setenv(weight_sync.VERIFY_ENV, "1")
    result = selected.send_weights()
    assert len(sender.calls) == 1
    assert result["sent_manifest"] == {name: weight_sync.tensor_manifest(value) for name, value in sender.calls[0]}
    assert result["trainer_parameters"] == {"numel": 2, "count": 1}
    assert result["evidence_sources"]
    assert result is not sender.result


def test_enabled_send_detects_changed_tensors_without_retry(monkeypatch):
    selected, calls = worker(), []

    class MutatingSender:
        def send(self, weights):
            calls.append(weights)
            weights[0][1].add_(1)
            return {"status": "ok"}

    selected._weight_sender = MutatingSender()
    monkeypatch.setenv(weight_sync.VERIFY_ENV, "1")
    with pytest.raises(RuntimeError, match="actually sent tensors changed"):
        selected.send_weights()
    assert len(calls) == 1
