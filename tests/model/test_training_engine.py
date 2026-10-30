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


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("cpu_offload", [False, True])
@pytest.mark.parametrize("name", ["", " ", "adam", " AdamW "])
def test_default_optimizer_resolution_preserves_offload_and_fused_policy(name, cpu_offload, fused):
    ds_config = _OFFLOAD if cpu_offload else _ZERO2
    if not cpu_offload:
        expected = "fused_adam"
    elif fused:
        expected = "deepspeed_cpu_adam"
    else:
        expected = "torch_cpu_adamw"
    assert resolve_optimizer_name(name, ds_config, fused=fused) == expected


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("cpu_offload", [False, True])
@pytest.mark.parametrize(
    "name, expected",
    [
        ("fusedadam", "fused_adam"),
        (" Fused-Adam ", "fused_adam"),
        ("torch_adamw", "torch_adamw"),
        (" Torch-AdamW ", "torch_adamw"),
        ("deepspeed_cpu_adam", "deepspeed_cpu_adam"),
        ("deepspeedcpuadam", "deepspeed_cpu_adam"),
        ("cpu_adam", "deepspeed_cpu_adam"),
        ("cpuadam", "deepspeed_cpu_adam"),
        ("deep_speed_cpu_adam", "deepspeed_cpu_adam"),
        (" Deep-Speed-CPU-Adam ", "deepspeed_cpu_adam"),
    ],
)
def test_explicit_optimizer_aliases_preserve_selection_and_cpu_placement(name, expected, cpu_offload, fused):
    ds_config = _OFFLOAD if cpu_offload else _ZERO2
    if expected == "torch_adamw" and cpu_offload:
        expected = "torch_cpu_adamw"
    assert resolve_optimizer_name(name, ds_config, fused=fused) == expected


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


def test_build_training_optimizer_disables_decay_on_every_qwen38_norm():
    pytest.importorskip("transformers.models.qwen4_exp")
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpForCausalLM

    # A GatedDeltaNet layer with PLE and a QSA layer: every RMSNorm the model family has.
    model = Qwen4ExpForCausalLM(
        Qwen4ExpTextConfig(
            vocab_size=32,
            hidden_size=16,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_conv_kernel_dim=2,
            moe_intermediate_size=8,
            shared_expert_intermediate_size=8,
            num_experts=4,
            num_experts_per_tok=2,
            layer_types=["linear_attention", "full_attention"],
            hc_count=2,
            hc_lowrank=4,
            ple_layer_ids=[1],
            ple_embed_dim=8,
            ple_conv_kernel_size=2,
            ngram_size=3,
            heads_per_ngram=1,
            ngram_vocab_size_base=11,
            make_ngram_vocab_size_divisible_by=4,
            split_ngram_parts=2,
            indexer_n_heads=1,
            indexer_kv_heads=1,
            indexer_head_dim=4,
            indexer_budget=4,
            indexer_compress_ratio=2,
            eos_token_id=1,
            pad_token_id=0,
            rope_parameters={"rope_type": "default", "rope_theta": 10000, "partial_rotary_factor": 0.5},
        )
    )

    optimizer = build_training_optimizer(
        model, OptimizerSettings(name="torch_adamw", weight_decay=0.1), _ZERO2, fused=False
    )

    decayed = {id(param) for group in optimizer.param_groups if group["weight_decay"] for param in group["params"]}
    decayed_norms = [
        name
        for module_name, module in model.named_modules()
        if "Norm" in type(module).__name__
        for name, param in module.named_parameters(prefix=module_name, recurse=False)
        if id(param) in decayed
    ]
    assert not decayed_norms, f"norm weights in the weight-decay group: {decayed_norms}"


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


def test_parameters_kept_in_fp32_get_their_own_optimizer_groups(monkeypatch):
    from arctic_platform.model.implementations.fp8 import preserve_fp8_params_through_ds_cast

    captured = {}

    class _Adam:
        def __init__(self, param_groups, **kwargs):
            captured["param_groups"] = param_groups

    monkeypatch.setattr("deepspeed.ops.adam.FusedAdam", _Adam)

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(2, 2, bias=False, dtype=torch.bfloat16)
            self.scale = torch.nn.Parameter(torch.ones(2, 2))
            self.norm = torch.nn.LayerNorm(2, dtype=torch.bfloat16)

        @classmethod
        def keeps_fp32_parameter(cls, name):
            return name == "scale"

    model = _Model()
    assert preserve_fp8_params_through_ds_cast(model) == (0, 1)

    build_training_optimizer(model, OptimizerSettings(weight_decay=0.1), _ZERO2, fused=True)

    groups = [
        (group["weight_decay"], {param.dtype for param in group["params"]}, len(group["params"]))
        for group in captured["param_groups"]
    ]
    assert groups == [(0.1, {torch.bfloat16}, 1), (0.1, {torch.float32}, 1), (0.0, {torch.bfloat16}, 2)]


def test_build_training_optimizer_keeps_flattened_weights_16_byte_aligned():
    model = torch.nn.Module()
    model.first = torch.nn.Linear(16, 16, bias=False)
    model.scale = torch.nn.Parameter(torch.ones(3))  # 3 elements, in the same group as the weights
    model.second = torch.nn.Linear(16, 16, bias=False)

    optimizer = build_training_optimizer(model, OptimizerSettings(name="torch_adamw"), _ZERO2, fused=False)

    (group,) = optimizer.param_groups
    # ZeRO places each parameter at the running sum of the sizes before it.
    assert [param.numel() for param in group["params"]] == [256, 256, 3]
    assert group["params"][0] is model.first.weight and group["params"][1] is model.second.weight


def test_build_training_optimizer_aligns_each_kept_fp32_group():
    from arctic_platform.model.implementations.fp8 import preserve_fp8_params_through_ds_cast

    class _Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
            self.odd = torch.nn.Parameter(torch.ones(3, dtype=torch.bfloat16))
            self.second = torch.nn.Linear(16, 16, bias=False, dtype=torch.bfloat16)
            self.scale = torch.nn.Parameter(torch.ones(3))
            self.base = torch.nn.Parameter(torch.ones(24))

        @classmethod
        def keeps_fp32_parameter(cls, name):
            return name in ("scale", "base")

    model = _Model()
    assert preserve_fp8_params_through_ds_cast(model) == (0, 2)

    optimizer = build_training_optimizer(model, OptimizerSettings(name="torch_adamw"), _ZERO2, fused=False)

    groups = [[(param.dtype, param.numel()) for param in group["params"]] for group in optimizer.param_groups]
    assert groups == [
        [(torch.bfloat16, 256), (torch.bfloat16, 256), (torch.bfloat16, 3)],
        [(torch.float32, 24), (torch.float32, 3)],
    ]


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

    class _CPUAdam(torch.optim.Optimizer):
        def __init__(self, param_groups, **kwargs):
            super().__init__(param_groups, {})
            self.adam_w_mode = kwargs["adamw_mode"]
            self.fp32_optimizer_states = kwargs["fp32_optimizer_states"]
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


def test_cpu_adam_checkpoint_hooks_preserve_mixed_groups_without_aliasing_saved_moments():
    from copy import deepcopy

    from arctic_platform.common.cpu_adam import _CpuAdamCheckpointState

    parameters = [
        torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16)),
        torch.nn.Parameter(torch.zeros(3, dtype=torch.float32)),
    ]
    optimizer = torch.optim.AdamW([{"params": [parameter]} for parameter in parameters])
    optimizer.fp32_optimizer_states = True
    checkpoint_state = _CpuAdamCheckpointState()
    optimizer.register_load_state_dict_pre_hook(checkpoint_state.before_load)
    optimizer.register_load_state_dict_post_hook(checkpoint_state.after_load)
    for parameter in parameters:
        optimizer.state[parameter] = {
            "step": torch.tensor(2.0),
            "exp_avg": torch.tensor([0.123456, 0.234567, 0.345678]),
            "exp_avg_sq": torch.tensor([0.012345, 0.023456, 0.034567]),
        }
    checkpoint = deepcopy(optimizer.state_dict())
    for _ in range(2):
        optimizer.load_state_dict(checkpoint)
        assert checkpoint_state.moments == []
        for index, parameter in enumerate(parameters):
            for name in ("exp_avg", "exp_avg_sq"):
                assert optimizer.state[parameter][name].dtype == torch.float32
                torch.testing.assert_close(
                    optimizer.state[parameter][name], checkpoint["state"][index][name], rtol=0, atol=0
                )
        saved_moment = checkpoint["state"][0]["exp_avg"].clone()
        optimizer.state[parameters[0]]["exp_avg"].add_(1)
        torch.testing.assert_close(checkpoint["state"][0]["exp_avg"], saved_moment, rtol=0, atol=0)


@pytest.mark.parametrize("invalid_groups", [None, [], [{"params": []}]])
def test_cpu_adam_checkpoint_hooks_keep_empty_state_and_native_group_validation(invalid_groups):
    from arctic_platform.common.cpu_adam import _CpuAdamCheckpointState

    parameter = torch.nn.Parameter(torch.zeros(3, dtype=torch.bfloat16))
    optimizer = torch.optim.AdamW([parameter])
    optimizer.fp32_optimizer_states = True
    checkpoint_state = _CpuAdamCheckpointState()
    optimizer.register_load_state_dict_pre_hook(checkpoint_state.before_load)
    optimizer.register_load_state_dict_post_hook(checkpoint_state.after_load)
    checkpoint = optimizer.state_dict()
    if invalid_groups is None:
        optimizer.load_state_dict(checkpoint)
    else:
        checkpoint["param_groups"] = invalid_groups
        with pytest.raises(ValueError, match="parameter group"):
            optimizer.load_state_dict(checkpoint)
    assert not optimizer.state
    assert checkpoint_state.moments == []


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
    monkeypatch.setattr(deepspeed_engine, "is_glm_moe_model", lambda path: False)
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


@pytest.mark.parametrize("fused,expected", [(True, "deepspeed_cpu_adam"), (False, "torch_cpu_adamw")])
def test_cpu_optimizer_log_names_the_selected_implementation(monkeypatch, caplog, fused, expected):
    class Adam:
        def __init__(self, *args, **kwargs):
            pass

    monkeypatch.setattr("deepspeed.ops.adam.DeepSpeedCPUAdam", Adam)
    monkeypatch.setattr("arctic_platform.common.cpu_adam.configure_nonfused_cpu_adam", lambda optimizer: optimizer)
    with caplog.at_level("INFO", logger="arctic_platform.common.deepspeed_engine"):
        build_training_optimizer(torch.nn.Linear(2, 2), OptimizerSettings(), _OFFLOAD, fused=fused)
    assert f"using optimizer='{expected}'" in caplog.text
