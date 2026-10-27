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
"""DeepSpeed config, optimizer selection, and engine initialization."""

from __future__ import annotations

import pytest
import torch

import arctic_platform.common.deepspeed_engine as deepspeed_engine
from arctic_platform.common.deepspeed_engine import OptimizerSettings
from arctic_platform.common.deepspeed_engine import TrainingEngineSpec
from arctic_platform.common.deepspeed_engine import build_deepspeed_config
from arctic_platform.common.deepspeed_engine import build_training_optimizer
from arctic_platform.common.deepspeed_engine import cpu_adam_fp32_optimizer_states
from arctic_platform.common.deepspeed_engine import initialize_training_engine
from arctic_platform.common.deepspeed_engine import optimizer_choice
from arctic_platform.common.deepspeed_engine import optimizer_implementation
from arctic_platform.common.deepspeed_engine import resolve_optimizer_name

_ZERO2 = {"zero_optimization": {"stage": 2}}
_OFFLOAD = {"zero_optimization": {"stage": 2, "offload_optimizer": {"device": "cpu"}}}


def _spec(**kwargs) -> TrainingEngineSpec:
    kwargs.setdefault("world_size", 4)
    return TrainingEngineSpec(**kwargs)


def test_bfloat16_job_reduces_gradients_in_float32_with_bf16_masters():
    ds_config = build_deepspeed_config(_spec())

    assert ds_config["bf16"] == {
        "enabled": True,
        "bf16_master_weights_and_grads": True,
        "bf16_optimizer_states": True,
    }
    assert ds_config["fp16"] == {"enabled": False}
    assert ds_config["communication_data_type"] == "fp32"
    assert ds_config["train_micro_batch_size_per_gpu"] == 1
    assert ds_config["gradient_accumulation_steps"] == 1
    assert ds_config["train_batch_size"] == 4
    assert ds_config["zero_optimization"] == {"stage": 2}
    assert "seed" not in ds_config
    assert "sequence_parallel_size" not in ds_config


@pytest.mark.parametrize("dtype", ["bf16", "bfloat16"])
@pytest.mark.parametrize("fp32_masters", [False, True])
def test_bfloat16_job_accumulates_gradients_in_float32(dtype, fp32_masters):
    ds_config = build_deepspeed_config(_spec(dtype=dtype, fp32_masters=fp32_masters))

    assert ds_config["data_types"]["grad_accum_dtype"] == "fp32"


@pytest.mark.parametrize("dtype", ["float16", "fp16", "float32", "fp32"])
def test_other_precisions_leave_accumulation_dtype_to_deepspeed(dtype):
    ds_config = build_deepspeed_config(_spec(dtype=dtype))

    assert "data_types" not in ds_config


@pytest.mark.parametrize("data_types", [{}, {"grad_accum_dtype": "bf16"}, {"grad_accum_dtype": "fp32"}])
def test_explicit_accumulation_override_is_preserved(data_types):
    override = {"data_types": data_types.copy()}
    ds_config = build_deepspeed_config(_spec(ds_overrides=override))

    assert ds_config["data_types"] == data_types
    assert override == {"data_types": data_types}


@pytest.mark.parametrize("fp32_masters", [False, True])
def test_deepspeed_resolves_float32_accumulation_without_changing_model_precision(fp32_masters):
    from deepspeed.runtime.config import DeepSpeedConfig
    from deepspeed.runtime.engine import DeepSpeedEngine

    ds_config = build_deepspeed_config(_spec(world_size=1, fp32_masters=fp32_masters))
    config = DeepSpeedConfig(ds_config)
    engine = DeepSpeedEngine.__new__(DeepSpeedEngine)
    object.__setattr__(engine, "_config", config)

    assert engine.get_data_types() == (torch.bfloat16, torch.float32)
    assert ds_config["bf16"]["bf16_master_weights_and_grads"] is not fp32_masters
    assert ds_config["bf16"]["bf16_optimizer_states"] is not fp32_masters


def test_float32_job_disables_reduced_precision_and_leaves_the_reduction_unset():
    ds_config = build_deepspeed_config(_spec(dtype="float32"))

    assert ds_config["bf16"]["enabled"] is False
    assert ds_config["bf16"]["bf16_master_weights_and_grads"] is False
    assert ds_config["fp16"]["enabled"] is False
    assert ds_config["communication_data_type"] is None


def test_fp32_masters_disable_bf16_master_weights():
    ds_config = build_deepspeed_config(_spec(fp32_masters=True))

    assert ds_config["bf16"]["enabled"] is True
    assert ds_config["bf16"]["bf16_master_weights_and_grads"] is False
    assert ds_config["bf16"]["bf16_optimizer_states"] is False


def test_sequence_parallel_shrinks_the_data_parallel_batch():
    ds_config = build_deepspeed_config(_spec(world_size=8, sequence_parallel=2))

    assert ds_config["train_batch_size"] == 4
    assert ds_config["sequence_parallel_size"] == 2


def test_sequence_parallel_must_divide_the_world():
    with pytest.raises(ValueError, match="must be divisible by sp_size"):
        build_deepspeed_config(_spec(world_size=4, sequence_parallel=3))


def test_seed_and_zero_bucket_are_written_before_client_overrides():
    ds_config = build_deepspeed_config(
        _spec(
            seed=7,
            zero_bucket_size=50_000_000,
            ds_overrides={"zero_optimization": {"stage": 1}, "gradient_clipping": 2.0},
        )
    )

    assert ds_config["seed"] == 7
    assert ds_config["gradient_clipping"] == 2.0
    assert ds_config["zero_optimization"] == {"stage": 1}


@pytest.mark.parametrize("lowered", ["bf16", "bfloat16", "BF16", "fp16"])
def test_lowering_the_reduction_dtype_under_bfloat16_parameters_is_refused(lowered):
    with pytest.raises(ValueError, match="communication_data_type must be fp32"):
        build_deepspeed_config(_spec(ds_overrides={"communication_data_type": lowered}))


def test_removing_the_reduction_dtype_under_bfloat16_parameters_is_refused():
    with pytest.raises(ValueError, match="communication_data_type must be fp32"):
        build_deepspeed_config(_spec(ds_overrides={"communication_data_type": None}))


def test_an_override_that_keeps_float32_is_accepted():
    ds_config = build_deepspeed_config(
        _spec(ds_overrides={"communication_data_type": "fp32", "gradient_clipping": 2.0})
    )

    assert ds_config["communication_data_type"] == "fp32"
    assert ds_config["gradient_clipping"] == 2.0


def test_default_adamw_is_fused_without_offload_and_cpu_adam_with_offload():
    assert resolve_optimizer_name("AdamW", _ZERO2) == "fused_adam"
    assert resolve_optimizer_name("", _ZERO2) == "fused_adam"
    assert resolve_optimizer_name("AdamW", _OFFLOAD) == "deepspeed_cpu_adam"


def test_explicit_optimizer_names_are_honored_under_offload():
    assert resolve_optimizer_name("fused_adam", _OFFLOAD) == "fused_adam"
    assert resolve_optimizer_name("torch_adamw", _OFFLOAD) == "torch_cpu_adamw"
    assert resolve_optimizer_name("torch_adamw", _ZERO2) == "torch_adamw"
    assert resolve_optimizer_name("cpu_adam", _ZERO2) == "deepspeed_cpu_adam"


def test_nonfused_generic_offload_names_select_standard_cpu_arithmetic():
    for name in ("", "Adam", "AdamW"):
        assert resolve_optimizer_name(name, _OFFLOAD, fused=False) == "torch_cpu_adamw"
    assert resolve_optimizer_name("torch_adamw", _OFFLOAD) == "torch_cpu_adamw"
    assert resolve_optimizer_name("cpu_adam", _OFFLOAD, fused=False) == "deepspeed_cpu_adam"


def test_unknown_optimizer_names_are_rejected():
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        resolve_optimizer_name("sgd", _ZERO2)


def test_cpu_adam_uses_fp32_states_unless_bf16_optimizer_states_are_enabled():
    assert cpu_adam_fp32_optimizer_states({"bf16": {"bf16_optimizer_states": True}}) is False
    assert cpu_adam_fp32_optimizer_states({"bf16": {"bf16_optimizer_states": False}}) is True
    assert cpu_adam_fp32_optimizer_states({}) is True


@pytest.mark.parametrize(
    "optimizer, ds_config, fused, expected",
    [
        (
            OptimizerSettings(),
            _ZERO2,
            True,
            ("fused_adam", "AdamW ran as the faster fused GPU implementation (FusedAdam)"),
        ),
        (
            OptimizerSettings(),
            _OFFLOAD,
            True,
            ("deepspeed_cpu_adam", "optimizer state is offloaded to CPU, so the CPU implementation (CPUAdam) ran"),
        ),
        (
            OptimizerSettings(name="cpu_adam"),
            _ZERO2,
            True,
            ("deepspeed_cpu_adam", "you selected 'cpu_adam'"),
        ),
        (
            OptimizerSettings(name="torch_adamw"),
            _ZERO2,
            True,
            ("torch_adamw", "you selected 'torch_adamw'"),
        ),
        (
            OptimizerSettings(fused=False),
            _ZERO2,
            False,
            ("torch_adamw", "you set 'optimizer.fused' to false, so standard AdamW ran"),
        ),
        (
            OptimizerSettings(),
            _ZERO2,
            False,
            ("torch_adamw", "FusedAdam isn't supported for this model, so standard AdamW ran"),
        ),
        (
            OptimizerSettings(fused=False),
            _OFFLOAD,
            False,
            (
                "deepspeed_cpu_adam_nonfused",
                "optimizer state is offloaded to CPU, with standard non-fused AdamW arithmetic",
            ),
        ),
        (
            OptimizerSettings(),
            _OFFLOAD,
            False,
            (
                "deepspeed_cpu_adam_nonfused",
                "optimizer state is offloaded to CPU, with standard non-fused AdamW arithmetic",
            ),
        ),
        (
            OptimizerSettings(name="cpu_adam", fused=False),
            _OFFLOAD,
            False,
            ("deepspeed_cpu_adam", "optimizer state is offloaded to CPU, so the CPU implementation (CPUAdam) ran"),
        ),
    ],
)
def test_optimizer_choice_matches_the_class_built(optimizer, ds_config, fused, expected):
    choice = optimizer_choice(optimizer, ds_config, fused=fused)

    assert (choice["implementation"], choice["reason"]) == expected


def test_optimizer_implementation_rejects_unknown_names():
    with pytest.raises(AssertionError, match="Unhandled optimizer"):
        optimizer_implementation("sgd", fused=True)


def test_build_training_optimizer_disables_decay_on_bias_and_norm(monkeypatch):
    captured = {}

    class _Adam:
        def __init__(self, param_groups, **kwargs):
            captured["param_groups"] = param_groups
            captured["kwargs"] = kwargs

    monkeypatch.setattr("deepspeed.ops.adam.FusedAdam", _Adam)

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(2, 2, bias=False)
            self.norm = torch.nn.LayerNorm(2)

    model = _Model()

    optimizer = build_training_optimizer(
        model,
        OptimizerSettings(lr=3e-4, weight_decay=0.1, eps=1e-6, betas=[0.8, 0.95]),
        _ZERO2,
        fused=True,
    )

    assert isinstance(optimizer, _Adam)
    groups = {group["weight_decay"]: [param.shape for param in group["params"]] for group in captured["param_groups"]}
    assert groups[0.1] == [torch.Size([2, 2])]
    assert groups[0.0] == [torch.Size([2]), torch.Size([2])]
    assert captured["kwargs"] == {"lr": 3e-4, "betas": (0.8, 0.95), "eps": 1e-6, "adam_w_mode": True}


def test_initialize_forwards_the_built_optimizer_and_skips_expert_split(monkeypatch):
    sentinel = object()
    calls = {}

    def _build(model, optimizer, ds_config, *, fused):
        calls["fused"] = fused
        return sentinel

    def _initialize(*, model, config, **kwargs):
        calls["model"] = model
        calls["config"] = config
        calls["kwargs"] = kwargs
        return "engine", kwargs.get("optimizer"), None, None

    monkeypatch.setattr(deepspeed_engine, "build_training_optimizer", _build)
    monkeypatch.setattr(deepspeed_engine, "preserve_fp8_params_through_ds_cast", lambda model: (0, 0))
    monkeypatch.setattr(deepspeed_engine, "is_glm_moe_model", lambda path: False)
    monkeypatch.setattr("deepspeed.initialize", _initialize)
    model = torch.nn.Linear(1, 1)
    ds_config = {"train_micro_batch_size_per_gpu": 1}
    mpu = object()

    engine, optimizer, choice = initialize_training_engine(
        model,
        ds_config,
        optimizer=OptimizerSettings(),
        ep_size=1,
        mpu=mpu,
        model_path="unused",
        rank=0,
    )

    assert engine == "engine"
    assert optimizer is sentinel
    assert choice["implementation"] == "fused_adam"
    assert calls["fused"] is True
    assert calls["kwargs"] == {"optimizer": sentinel, "mpu": mpu}


def test_initialize_without_an_optimizer_omits_it_and_the_mpu(monkeypatch):
    calls = {}

    def _initialize(*, model, config, **kwargs):
        calls["kwargs"] = kwargs
        return "engine", None, None, None

    monkeypatch.setattr(deepspeed_engine, "preserve_fp8_params_through_ds_cast", lambda model: (0, 0))
    monkeypatch.setattr("deepspeed.initialize", _initialize)

    engine, optimizer, choice = initialize_training_engine(
        torch.nn.Linear(1, 1),
        {},
        optimizer=None,
        ep_size=4,
        mpu=None,
        model_path="unused",
        rank=1,
    )

    assert (engine, optimizer, choice) == ("engine", None, None)
    assert calls["kwargs"] == {}


def test_initialize_splits_expert_groups_and_leaves_fused_adam_off_for_glm(monkeypatch):
    calls = {}

    class _Adam:
        def __init__(self, param_groups, **kwargs):
            self.param_groups = param_groups

    def _split(groups):
        calls["split"] = list(groups)
        return [{"params": [], "name": "experts"}]

    monkeypatch.setattr("deepspeed.ops.adam.FusedAdam", _Adam)
    monkeypatch.setattr("torch.optim.AdamW", _Adam)
    monkeypatch.setattr(deepspeed_engine, "is_glm_moe_model", lambda path: True)
    monkeypatch.setattr(deepspeed_engine, "split_expert_param_groups", _split)
    monkeypatch.setattr(deepspeed_engine, "preserve_fp8_params_through_ds_cast", lambda model: (0, 0))

    def _initialize(*, model, config, **kwargs):
        calls["optimizer"] = kwargs["optimizer"]
        return "engine", kwargs["optimizer"], None, None

    monkeypatch.setattr("deepspeed.initialize", _initialize)
    model = torch.nn.Linear(2, 2, bias=False)

    engine, optimizer, choice = initialize_training_engine(
        model,
        _ZERO2,
        optimizer=OptimizerSettings(),
        ep_size=2,
        mpu=None,
        model_path="glm",
        rank=0,
    )

    assert engine == "engine"
    assert optimizer is calls["optimizer"]
    assert choice["implementation"] == "torch_adamw"
    assert calls["split"]
    assert optimizer.param_groups == [{"params": [], "name": "experts"}]


@pytest.mark.parametrize("force_cpu_optimizer", [None, True, False])
def test_initialize_glm_cpu_offload_honors_nonfused_policy(monkeypatch, force_cpu_optimizer):
    calls = {}

    class _CPUAdam:
        def __init__(self, param_groups, **kwargs):
            self.param_groups = param_groups
            self.adam_w_mode = kwargs["adamw_mode"]
            self.ds_opt_adam = object()

    def _initialize(*, model, config, **kwargs):
        calls["config"] = config
        calls["optimizer"] = kwargs["optimizer"]
        return "engine", kwargs["optimizer"], None, None

    monkeypatch.setattr(deepspeed_engine, "is_glm_moe_model", lambda path: True)
    monkeypatch.setattr(deepspeed_engine, "preserve_fp8_params_through_ds_cast", lambda model: (0, 0))
    monkeypatch.setattr("deepspeed.initialize", _initialize)
    monkeypatch.setattr("deepspeed.ops.adam.DeepSpeedCPUAdam", _CPUAdam)
    ds_config = {"zero_optimization": {"stage": 2, "offload_optimizer": {"device": "cpu"}}}
    if force_cpu_optimizer is not None:
        ds_config["zero_force_ds_cpu_optimizer"] = force_cpu_optimizer

    _, optimizer, choice = initialize_training_engine(
        torch.nn.Linear(2, 2, bias=False),
        ds_config,
        optimizer=OptimizerSettings(),
        ep_size=1,
        mpu=None,
        model_path="glm",
        rank=0,
    )

    assert isinstance(optimizer, _CPUAdam)
    assert choice["implementation"] == "deepspeed_cpu_adam_nonfused"
    assert calls["optimizer"] is optimizer
    assert calls["config"] is ds_config
    if force_cpu_optimizer is not None:
        assert calls["config"]["zero_force_ds_cpu_optimizer"] is force_cpu_optimizer
    assert "zero_allow_untested_optimizer" not in calls["config"]
    if force_cpu_optimizer is None:
        assert "zero_force_ds_cpu_optimizer" not in ds_config


def test_build_training_optimizer_keeps_explicit_cpu_adam_when_nonfused(monkeypatch):
    captured = {}

    class _CPUAdam:
        def __init__(self, param_groups, **kwargs):
            captured["kwargs"] = kwargs

    monkeypatch.setattr("deepspeed.ops.adam.DeepSpeedCPUAdam", _CPUAdam)
    optimizer = build_training_optimizer(
        torch.nn.Linear(2, 2, bias=False),
        OptimizerSettings(name="cpu_adam", fused=False),
        _OFFLOAD,
        fused=False,
    )

    assert isinstance(optimizer, _CPUAdam)
    assert captured["kwargs"]["adamw_mode"] is True


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_nonfused_cpu_adam_kernel_matches_torch_adamw_across_steps(dtype, weight_decay):
    from arctic_platform.common.cpu_adam import _NonfusedAdamWOps

    initial = torch.linspace(-0.25, 0.25, 17, dtype=dtype)
    actual = initial.clone()
    expected = torch.nn.Parameter(initial.clone())
    baseline = torch.optim.AdamW(
        [expected],
        lr=0.003,
        betas=(0.8, 0.95),
        eps=1e-6,
        weight_decay=weight_decay,
        foreach=False,
        fused=False,
    )
    moments = [torch.zeros_like(actual), torch.zeros_like(actual)]
    ops = _NonfusedAdamWOps(None)
    for step in range(1, 5):
        gradient = torch.linspace(-0.3, 0.4, 17, dtype=dtype) / step
        expected.grad = gradient.clone()
        baseline.step()
        ops.adam_update(
            0,
            step,
            0.003,
            0.8,
            0.95,
            1e-6,
            weight_decay,
            True,
            actual,
            gradient,
            *moments,
        )
        torch.testing.assert_close(actual, expected.detach(), rtol=0, atol=0)
        for key, value in zip(("exp_avg", "exp_avg_sq"), moments):
            torch.testing.assert_close(value, baseline.state[expected][key], rtol=0, atol=0)


@pytest.mark.parametrize(
    "parameter_dtype, state_dtype",
    [
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ],
)
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_nonfused_cpu_adam_native_counter_and_checkpoint_resume(parameter_dtype, state_dtype, weight_decay):
    from copy import deepcopy

    from deepspeed.ops.adam import DeepSpeedCPUAdam
    from torch.optim.adamw import adamw

    from arctic_platform.common.cpu_adam import configure_nonfused_cpu_adam

    initial = torch.linspace(-0.25, 0.25, 17, dtype=parameter_dtype)
    parameter = torch.nn.Parameter(initial.clone())
    expected = initial.clone()
    expected_moments = [torch.zeros_like(expected, dtype=state_dtype) for _ in range(2)]
    expected_step = torch.tensor(0.0)

    def build(value):
        return configure_nonfused_cpu_adam(
            DeepSpeedCPUAdam(
                [value],
                lr=0.003,
                betas=(0.8, 0.95),
                eps=1e-6,
                weight_decay=weight_decay,
                adamw_mode=True,
                fp32_optimizer_states=state_dtype == torch.float32,
            )
        )

    optimizer = build(parameter)
    for step in range(1, 5):
        gradient = torch.linspace(-0.3, 0.4, 17, dtype=parameter_dtype) / step
        parameter.grad = gradient.clone()
        optimizer.step()
        with torch.no_grad():
            adamw(
                [expected],
                [gradient.to(state_dtype)],
                [expected_moments[0]],
                [expected_moments[1]],
                [],
                [expected_step],
                amsgrad=False,
                beta1=0.8,
                beta2=0.95,
                lr=0.003,
                weight_decay=weight_decay,
                eps=1e-6,
                maximize=False,
                foreach=False,
                capturable=False,
                differentiable=False,
                fused=False,
            )
        state = optimizer.state[parameter]
        assert type(optimizer) is DeepSpeedCPUAdam
        assert type(state["step"]) is int
        assert state["step"] == step
        torch.testing.assert_close(parameter.detach(), expected, rtol=0, atol=0)
        for key, value in zip(("exp_avg", "exp_avg_sq"), expected_moments):
            assert state[key].dtype == state_dtype
            torch.testing.assert_close(state[key], value, rtol=0, atol=0)
        if step == 2:
            checkpoint = deepcopy(optimizer.state_dict())
            parameter = torch.nn.Parameter(parameter.detach().clone())
            optimizer = build(parameter)
            optimizer.load_state_dict(checkpoint)
            restored = optimizer.state[parameter]
            assert type(restored["step"]) is int
            assert restored["step"] == step
            for key, value in zip(("exp_avg", "exp_avg_sq"), expected_moments):
                assert restored[key].dtype == state_dtype
                torch.testing.assert_close(restored[key], value, rtol=0, atol=0)


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
    assert deepspeed_zero_initialization.maybe_initialize_with_zero_allocation_trace(
        lambda **kwargs: kwargs, config, 0, model="model"
    ) == {"config": config, "model": "model"}


@pytest.mark.parametrize("fail_before_method", [False, True])
def test_zero_initialization_trace_removes_option_and_restores_method(monkeypatch, fail_before_method):
    import io
    import json
    from contextlib import nullcontext

    from arctic_platform.debug import deepspeed_zero_initialization

    stream = io.StringIO()
    monkeypatch.setattr(deepspeed_zero_initialization, "zero_optimizer_class", lambda: _ZeroInitializationFixture)
    monkeypatch.setattr(deepspeed_zero_initialization, "open_trace", lambda directory, rank: nullcontext(stream))
    original = _ZeroInitializationFixture.initialize_optimizer_states
    config = {
        deepspeed_zero_initialization.TRAINING_OPTION: {
            "request_id": "test",
            "output_dir": "/data-fast/tmp/zero-test",
        },
        "zero_optimization": {"stage": 2},
    }

    def initialize(**kwargs):
        assert deepspeed_zero_initialization.TRAINING_OPTION not in kwargs["config"]
        if fail_before_method:
            raise RuntimeError("engine failed before optimizer")
        return _ZeroInitializationFixture().initialize_optimizer_states()

    if fail_before_method:
        with pytest.raises(RuntimeError, match="engine failed before optimizer"):
            deepspeed_zero_initialization.maybe_initialize_with_zero_allocation_trace(initialize, config, 0)
    else:
        assert (
            deepspeed_zero_initialization.maybe_initialize_with_zero_allocation_trace(initialize, config, 0)
            == "initialized"
        )
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        assert rows[0]["request_id"] == "test"
        assert rows[-1]["event"] == "INITIALIZATION_COMPLETE"
    assert _ZeroInitializationFixture.initialize_optimizer_states is original
    assert deepspeed_zero_initialization.TRAINING_OPTION in config


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
        deepspeed_zero_initialization.maybe_initialize_with_zero_allocation_trace(
            lambda **kwargs: pytest.fail("invalid trace initialized engine"),
            {deepspeed_zero_initialization.TRAINING_OPTION: option},
            rank,
        )
