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
"""Request-scoped allocation tracing and native engine argument routing."""

import pytest
import torch

from arctic_platform.common import deepspeed_engine
from arctic_platform.common.deepspeed_engine import initialize_training_engine
from arctic_platform.common.optimizer import OptimizerSettings


class _ZeroInitializationFixture:
    def __init__(self):
        self.partition_size = [4, 2]
        self.single_partition_of_fp32_groups = [torch.empty(4), torch.empty(2)]
        self.device = "cpu"
        self.cpu_offload = True
        self.cpu_offload_pin_memory = False

    def initialize_optimizer_states(self):
        for index, tensor in enumerate(self.single_partition_of_fp32_groups):
            tensor.grad = torch.zeros(self.partition_size[index], dtype=tensor.dtype, device=self.device)
        return "initialized"


def test_zero_initialization_trace_records_real_allocations_without_global_patch():
    from arctic_platform.debug.deepspeed_zero_initialization import initialize_with_allocation_trace

    optimizer = _ZeroInitializationFixture()
    original_zeros = torch.zeros
    rows = []

    def record(event, **fields):
        assert torch.zeros is original_zeros
        rows.append({"event": event, **fields})

    result = initialize_with_allocation_trace(
        _ZeroInitializationFixture.initialize_optimizer_states, optimizer, record
    )
    assert result == "initialized"
    assert [row["event"] for row in rows] == [
        "INITIALIZATION_START",
        "ALLOCATION_START",
        "ALLOCATION_COMPLETE",
        "ALLOCATION_START",
        "ALLOCATION_COMPLETE",
        "INITIALIZATION_COMPLETE",
    ]
    assert [row["bytes"] for row in rows if row["event"] == "ALLOCATION_COMPLETE"] == [16, 8]
    assert rows[0]["partitions"][0]["partition_size"] == 4
    for tensor in optimizer.single_partition_of_fp32_groups:
        assert torch.equal(tensor.grad, torch.zeros_like(tensor))


def test_zero_initialization_trace_propagates_allocation_error(monkeypatch):
    from arctic_platform.debug.deepspeed_zero_initialization import initialize_with_allocation_trace

    optimizer = _ZeroInitializationFixture()
    rows = []
    failure = RuntimeError("allocation failed")

    def fail(*args, **kwargs):
        raise failure

    monkeypatch.setattr(torch, "zeros", fail)
    with pytest.raises(RuntimeError) as caught:
        initialize_with_allocation_trace(
            _ZeroInitializationFixture.initialize_optimizer_states,
            optimizer,
            lambda event, **fields: rows.append({"event": event, **fields}),
        )
    assert caught.value is failure
    assert [row["event"] for row in rows] == [
        "INITIALIZATION_START",
        "ALLOCATION_START",
        "ALLOCATION_ERROR",
        "INITIALIZATION_ERROR",
    ]


def test_zero_initialization_trace_disabled_keeps_config_identity(monkeypatch):
    from arctic_platform.debug import deepspeed_zero_initialization

    monkeypatch.setattr(
        deepspeed_zero_initialization, "zero_optimizer_class", lambda: pytest.fail("disabled trace imported ZeRO")
    )
    config = {"zero_optimization": {"stage": 2}}
    with deepspeed_zero_initialization.maybe_trace_zero_initialization(config, 0) as init_config:
        assert init_config is config


def test_zero_initialization_trace_rejects_missing_source_before_opening_trace(monkeypatch):
    import inspect

    from arctic_platform.debug import deepspeed_zero_initialization

    original = _ZeroInitializationFixture.initialize_optimizer_states
    monkeypatch.setattr(deepspeed_zero_initialization, "zero_optimizer_class", lambda: _ZeroInitializationFixture)
    monkeypatch.setattr(inspect, "getsourcefile", lambda method: None)
    monkeypatch.setattr(
        deepspeed_zero_initialization, "open_trace", lambda *args: pytest.fail("unidentified source opened a trace")
    )
    config = {
        deepspeed_zero_initialization.TRAINING_OPTION: {
            "request_id": "missing-source",
            "output_dir": "/data-fast/tmp/zero-test",
        }
    }
    with pytest.raises(RuntimeError, match="inspectable optimizer source"):
        with deepspeed_zero_initialization.maybe_trace_zero_initialization(config, 0):
            pytest.fail("unidentified source entered initialization context")
    assert _ZeroInitializationFixture.initialize_optimizer_states is original
    assert deepspeed_zero_initialization.TRAINING_OPTION in config


@pytest.mark.parametrize("fail_before_method", [False, True])
def test_zero_initialization_trace_removes_option_and_restores_method(monkeypatch, fail_before_method):
    import io
    import json

    from arctic_platform.debug import deepspeed_zero_initialization

    stream = io.StringIO()
    monkeypatch.setattr(deepspeed_zero_initialization, "zero_optimizer_class", lambda: _ZeroInitializationFixture)
    monkeypatch.setattr(deepspeed_zero_initialization, "open_trace", lambda directory, rank: stream)
    original = _ZeroInitializationFixture.initialize_optimizer_states
    config = {
        deepspeed_zero_initialization.TRAINING_OPTION: {
            "request_id": "test",
            "output_dir": "/data-fast/tmp/zero-test",
        },
        "zero_optimization": {"stage": 2},
    }

    def initialize():
        if fail_before_method:
            raise RuntimeError("engine failed before optimizer")
        return _ZeroInitializationFixture().initialize_optimizer_states()

    if fail_before_method:
        with pytest.raises(RuntimeError, match="engine failed before optimizer"):
            with deepspeed_zero_initialization.maybe_trace_zero_initialization(config, 0) as init_config:
                assert init_config == {"zero_optimization": {"stage": 2}}
                initialize()
    else:
        with deepspeed_zero_initialization.maybe_trace_zero_initialization(config, 0) as init_config:
            assert init_config == {"zero_optimization": {"stage": 2}}
            assert initialize() == "initialized"
            rows = [json.loads(line) for line in stream.getvalue().splitlines()]
            assert rows[0]["request_id"] == "test"
            assert rows[-1]["event"] == "INITIALIZATION_COMPLETE"
    assert stream.closed
    assert _ZeroInitializationFixture.initialize_optimizer_states is original
    assert deepspeed_zero_initialization.TRAINING_OPTION in config


@pytest.mark.parametrize("with_optimizer", [False, True])
@pytest.mark.parametrize("trace_enabled", [False, True])
@pytest.mark.parametrize("fail_initialize", [False, True])
def test_engine_initializes_once_inside_optional_trace(monkeypatch, with_optimizer, trace_enabled, fail_initialize):
    import io

    from arctic_platform.debug import deepspeed_zero_initialization

    stream = io.StringIO()
    original = _ZeroInitializationFixture.initialize_optimizer_states
    monkeypatch.setattr(deepspeed_zero_initialization, "zero_optimizer_class", lambda: _ZeroInitializationFixture)
    monkeypatch.setattr(deepspeed_zero_initialization, "open_trace", lambda directory, rank: stream)
    monkeypatch.setattr(deepspeed_engine, "preserve_fp8_params_through_ds_cast", lambda model: (0, 0))
    built = object()
    monkeypatch.setattr(deepspeed_engine, "build_training_optimizer", lambda *args, **kwargs: built)
    ds_config = {"zero_optimization": {"stage": 2}}
    if trace_enabled:
        ds_config[deepspeed_zero_initialization.TRAINING_OPTION] = {
            "request_id": "engine-test",
            "output_dir": "/data-fast/tmp/zero-test",
        }
    model = torch.nn.Linear(1, 1)
    mpu = object()
    calls = []
    failure = RuntimeError("engine initialization failed")

    def initialize(**kwargs):
        calls.append(kwargs)
        assert (_ZeroInitializationFixture.initialize_optimizer_states is not original) is trace_enabled
        assert kwargs["config"] == {"zero_optimization": {"stage": 2}}
        if not trace_enabled:
            assert kwargs["config"] is ds_config
        assert kwargs["model"] is model
        assert kwargs["mpu"] is mpu
        if with_optimizer:
            assert kwargs["optimizer"] is built
        else:
            assert "optimizer" not in kwargs
        if fail_initialize:
            raise failure
        assert _ZeroInitializationFixture().initialize_optimizer_states() == "initialized"
        return "engine", kwargs.get("optimizer"), None, None

    monkeypatch.setattr("deepspeed.initialize", initialize)

    def run():
        return initialize_training_engine(
            model,
            ds_config,
            optimizer=OptimizerSettings() if with_optimizer else None,
            ep_size=1,
            mpu=mpu,
            model_path="unused",
            rank=0,
        )

    if fail_initialize:
        with pytest.raises(RuntimeError) as caught:
            run()
        assert caught.value is failure
    else:
        engine, optimizer, choice = run()
        assert engine == "engine"
        assert optimizer is (built if with_optimizer else None)
        assert (choice is not None) is with_optimizer
    assert len(calls) == 1
    assert _ZeroInitializationFixture.initialize_optimizer_states is original
    assert stream.closed is trace_enabled
    assert (deepspeed_zero_initialization.TRAINING_OPTION in ds_config) is trace_enabled


@pytest.mark.parametrize(
    "option, rank",
    [
        ({}, 0),
        ({"request_id": "../escape", "output_dir": "/data-fast/tmp"}, 0),
        ({"request_id": "test", "output_dir": "/tmp"}, 0),
        ({"request_id": "test", "output_dir": "/data-fast/tmp"}, True),
    ],
)
def test_zero_initialization_trace_rejects_invalid_request(option, rank):
    from arctic_platform.debug import deepspeed_zero_initialization

    with pytest.raises(ValueError):
        with deepspeed_zero_initialization.maybe_trace_zero_initialization(
            {deepspeed_zero_initialization.TRAINING_OPTION: option}, rank
        ):
            pytest.fail("invalid trace entered initialization context")
