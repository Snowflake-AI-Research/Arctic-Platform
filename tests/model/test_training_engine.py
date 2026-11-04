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

import json

import pytest
import torch
from torch.utils.checkpoint import checkpoint

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
from arctic_platform.testing_utils import require_deepspeed
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_equal

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
    monkeypatch.setattr(deepspeed_engine, "is_glm_moe_model", lambda path: False)
    monkeypatch.setattr(deepspeed_engine, "preserve_fp8_params_through_ds_cast", lambda model: (0, 0))
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


@pytest.mark.parametrize("model_type", ["glm_moe_dsa", "glm5_next_text"])
@pytest.mark.parametrize("fused,expected", [(None, "torch_adamw"), (False, "torch_adamw"), (True, "fused_adam")])
def test_initialize_splits_expert_groups_and_respects_glm_optimizer_choice(
    monkeypatch, tmp_path, model_type, fused, expected
):
    calls = {}

    class _Adam:
        def __init__(self, param_groups, **kwargs):
            self.param_groups = param_groups

    def _split(groups):
        calls["split"] = list(groups)
        return [{"params": [], "name": "experts"}]

    monkeypatch.setattr("deepspeed.ops.adam.FusedAdam", _Adam)
    monkeypatch.setattr("torch.optim.AdamW", _Adam)
    monkeypatch.setattr(deepspeed_engine, "split_expert_param_groups", _split)
    monkeypatch.setattr(deepspeed_engine, "preserve_fp8_params_through_ds_cast", lambda model: (0, 0))

    def _initialize(*, model, config, **kwargs):
        calls["optimizer"] = kwargs["optimizer"]
        return "engine", kwargs["optimizer"], None, None

    monkeypatch.setattr("deepspeed.initialize", _initialize)
    model = torch.nn.Linear(2, 2, bias=False)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))

    engine, optimizer, choice = initialize_training_engine(
        model,
        _ZERO2,
        optimizer=OptimizerSettings(fused=fused),
        ep_size=2,
        mpu=None,
        model_path=str(tmp_path),
        rank=0,
    )

    assert engine == "engine"
    assert optimizer is calls["optimizer"]
    assert choice["implementation"] == expected
    assert calls["split"]
    assert optimizer.param_groups == [{"params": [], "name": "experts"}]


@pytest.mark.parametrize("force_cpu_optimizer", [None, True, False])
def test_initialize_cpu_offload_honors_explicit_nonfused_policy(monkeypatch, force_cpu_optimizer):
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

    monkeypatch.setattr(deepspeed_engine, "preserve_fp8_params_through_ds_cast", lambda model: (0, 0))
    monkeypatch.setattr("deepspeed.initialize", _initialize)
    monkeypatch.setattr("deepspeed.ops.adam.DeepSpeedCPUAdam", _CPUAdam)
    ds_config = {"zero_optimization": {"stage": 2, "offload_optimizer": {"device": "cpu"}}}
    if force_cpu_optimizer is not None:
        ds_config["zero_force_ds_cpu_optimizer"] = force_cpu_optimizer

    _, optimizer, choice = initialize_training_engine(
        torch.nn.Linear(2, 2, bias=False),
        ds_config,
        optimizer=OptimizerSettings(fused=False),
        ep_size=1,
        mpu=None,
        model_path="unused",
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


@pytest.mark.parametrize("model_type", ["glm_moe_dsa", "glm5_next_text"])
@pytest.mark.parametrize(
    "name,fused,expected",
    [
        ("AdamW", None, "deepspeed_cpu_adam"),
        ("AdamW", True, "deepspeed_cpu_adam"),
        ("AdamW", False, "deepspeed_cpu_adam_nonfused"),
        ("torch_adamw", None, "deepspeed_cpu_adam_nonfused"),
        ("cpu_adam", False, "deepspeed_cpu_adam"),
    ],
)
def test_glm_gpu_fallback_preserves_cpu_offload_optimizer_choice(
    monkeypatch, tmp_path, model_type, name, fused, expected
):
    native_ops = object()
    calls = {}

    class _CPUAdam(torch.optim.Optimizer):
        def __init__(self, param_groups, **kwargs):
            super().__init__(param_groups, {})
            self.adam_w_mode = kwargs["adamw_mode"]
            self.fp32_optimizer_states = kwargs["fp32_optimizer_states"]
            self.ds_opt_adam = native_ops

    def _initialize(*, model, config, **kwargs):
        calls["config"] = config
        calls["optimizer"] = kwargs["optimizer"]
        return "engine", kwargs["optimizer"], None, None

    monkeypatch.setattr(deepspeed_engine, "preserve_fp8_params_through_ds_cast", lambda model: (0, 0))
    monkeypatch.setattr("deepspeed.initialize", _initialize)
    monkeypatch.setattr("deepspeed.ops.adam.DeepSpeedCPUAdam", _CPUAdam)
    (tmp_path / "config.json").write_text(json.dumps(dict(model_type=model_type)))
    ds_config = dict(zero_optimization=dict(stage=2, offload_optimizer=dict(device="cpu")))
    supplied_config = json.dumps(ds_config, sort_keys=True)

    _, optimizer, choice = initialize_training_engine(
        torch.nn.Linear(2, 2, bias=False),
        ds_config,
        optimizer=OptimizerSettings(name=name, fused=fused),
        ep_size=1,
        mpu=None,
        model_path=str(tmp_path),
        rank=0,
    )

    assert choice["implementation"] == expected
    assert calls["optimizer"] is optimizer
    assert calls["config"] is ds_config
    assert json.dumps(ds_config, sort_keys=True) == supplied_config
    if expected == "deepspeed_cpu_adam":
        assert optimizer.ds_opt_adam is native_ops
    else:
        assert optimizer.ds_opt_adam.native_ops is native_ops


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
@pytest.mark.parametrize("elements", [0, 1, 5, 6, 19])
def test_nonfused_cpu_adam_bounds_updates_with_exact_partition_parity(
    monkeypatch, parameter_dtype, state_dtype, weight_decay, elements
):
    from torch.optim.adamw import adamw

    from arctic_platform.common import cpu_adam

    monkeypatch.setattr(cpu_adam, "_UPDATE_CHUNK_ELEMENTS", 5, raising=False)
    parameter = torch.linspace(-0.25, 0.25, elements, dtype=parameter_dtype)
    expected = parameter.clone()
    moments = [torch.zeros_like(parameter, dtype=state_dtype) for _ in range(2)]
    expected_moments = [moment.clone() for moment in moments]
    expected_step = torch.tensor(0.0)
    calls = []

    def bounded_adamw(parameters, gradients, first_moments, second_moments, max_moments, steps, **kwargs):
        calls.append((parameters[0].numel(), steps[0].item()))
        assert parameters[0].numel() <= 5
        return adamw(parameters, gradients, first_moments, second_moments, max_moments, steps, **kwargs)

    monkeypatch.setattr(cpu_adam, "adamw", bounded_adamw)
    for step in range(1, 5):
        gradient = torch.linspace(-0.3, 0.4, elements, dtype=torch.float32) / step
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
        calls.clear()
        cpu_adam._NonfusedAdamWOps(None).adam_update(
            0, step, 0.003, 0.8, 0.95, 1e-6, weight_decay, True, parameter, gradient, *moments
        )
        torch_assert_equal(parameter, expected)
        for actual, baseline in zip(moments, expected_moments):
            torch_assert_equal(actual, baseline)
        assert sum(count for count, _ in calls) == elements
        assert all(counter == step - 1 for _, counter in calls)


@pytest.mark.parametrize("layout", ["matrix", "parameter_view", "gradient_view", "moment_view"])
def test_nonfused_cpu_adam_preserves_matrix_and_strided_updates(monkeypatch, layout):
    from torch.optim.adamw import adamw

    from arctic_platform.common import cpu_adam

    monkeypatch.setattr(cpu_adam, "_UPDATE_CHUNK_ELEMENTS", 5)
    parameter = torch.linspace(-0.25, 0.25, 21).reshape(3, 7)
    gradient = torch.linspace(-0.3, 0.4, 21).reshape(3, 7)
    moments = [torch.zeros_like(parameter) for _ in range(2)]

    def strided_view(value):
        return value.T.contiguous().T

    if layout == "parameter_view":
        parameter = strided_view(parameter)
    elif layout == "gradient_view":
        gradient = strided_view(gradient)
    elif layout == "moment_view":
        moments = [strided_view(moment) for moment in moments]
    expected = parameter.clone()
    expected_moments = [moment.clone() for moment in moments]
    calls = []

    def observed_adamw(parameters, *args, **kwargs):
        calls.append(parameters[0].numel())
        return adamw(parameters, *args, **kwargs)

    monkeypatch.setattr(cpu_adam, "adamw", observed_adamw)
    for step in range(1, 4):
        with torch.no_grad():
            adamw(
                [expected],
                [gradient],
                [expected_moments[0]],
                [expected_moments[1]],
                [],
                [torch.tensor(step - 1, dtype=torch.float32)],
                amsgrad=False,
                beta1=0.8,
                beta2=0.95,
                lr=0.003,
                weight_decay=0.1,
                eps=1e-6,
                maximize=False,
                foreach=False,
                capturable=False,
                differentiable=False,
                fused=False,
            )
        calls.clear()
        cpu_adam._NonfusedAdamWOps(None).adam_update(
            0, step, 0.003, 0.8, 0.95, 1e-6, 0.1, True, parameter, gradient, *moments
        )
        assert calls == ([5, 5, 5, 5, 1] if layout == "matrix" else [21])
        torch_assert_equal(parameter, expected)
        for actual, baseline in zip(moments, expected_moments):
            torch_assert_equal(actual, baseline)


@pytest.mark.parametrize(
    "parameter_dtype, state_dtype",
    [
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ],
)
@pytest.mark.parametrize("weight_decay", [0.0, 0.1])
def test_nonfused_cpu_adam_native_counter_and_checkpoint_resume(
    monkeypatch, parameter_dtype, state_dtype, weight_decay
):
    from copy import deepcopy

    from deepspeed.ops.adam import DeepSpeedCPUAdam
    from torch.optim.adamw import adamw

    from arctic_platform.common import cpu_adam
    from arctic_platform.common.cpu_adam import configure_nonfused_cpu_adam

    monkeypatch.setattr(cpu_adam, "_UPDATE_CHUNK_ELEMENTS", 5, raising=False)
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
    monkeypatch.setattr(deepspeed_engine, "is_glm_moe_model", lambda path: False)
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


_PARTITION_WIDTH = 64


def test_offloaded_file_snapshot_reuses_bounded_storage_and_restores_exactly(monkeypatch, tmp_path):
    monkeypatch.setattr(deepspeed_engine, "_OFFLOADED_SNAPSHOT_CHUNK_BYTES", 64)
    monkeypatch.setattr(deepspeed_engine.tempfile, "gettempdir", lambda: str(tmp_path))
    storage = deepspeed_engine._OffloadedGradientStorage()
    source = torch.arange(55, dtype=torch.float32).reshape(5, 11)
    snapshot = storage.snapshot(source)
    assert isinstance(snapshot, deepspeed_engine._OffloadedGradientFile)
    assert snapshot.size == source.numel() * source.element_size()
    assert storage.staging.numel() * storage.staging.element_size() == 64
    descriptor = snapshot.file.fileno()
    initial_buffer = storage.staging.data_ptr()
    destination = torch.empty_like(source)
    snapshot.restore(destination)
    torch_assert_equal(destination, source)
    source.add_(1.25)
    snapshot.copy_(source)
    snapshot.restore(destination)
    torch_assert_equal(destination, source)
    assert snapshot.file.fileno() == descriptor
    assert storage.staging.data_ptr() == initial_buffer
    snapshot.file.close()


@pytest.mark.parametrize("operation", ["pwritev", "preadv"])
def test_offloaded_file_snapshot_completes_partial_io(monkeypatch, tmp_path, operation):
    monkeypatch.setattr(deepspeed_engine, "_OFFLOADED_SNAPSHOT_CHUNK_BYTES", 64)
    monkeypatch.setattr(deepspeed_engine.tempfile, "gettempdir", lambda: str(tmp_path))
    native = getattr(deepspeed_engine.os, operation)
    calls = []

    def partial(descriptor, views, offset):
        calls.append(offset)
        return native(descriptor, [views[0][: min(12, len(views[0]))]], offset)

    monkeypatch.setattr(deepspeed_engine.os, operation, partial)
    storage = deepspeed_engine._OffloadedGradientStorage()
    source = torch.arange(55, dtype=torch.float32)
    snapshot = storage.snapshot(source)
    destination = torch.empty_like(source)
    snapshot.restore(destination)
    torch_assert_equal(destination, source)
    assert len(calls) > 4
    snapshot.file.close()


def test_offloaded_file_snapshot_closes_failed_write(monkeypatch, tmp_path):
    monkeypatch.setattr(deepspeed_engine, "_OFFLOADED_SNAPSHOT_CHUNK_BYTES", 64)
    monkeypatch.setattr(deepspeed_engine.tempfile, "gettempdir", lambda: str(tmp_path))
    native = deepspeed_engine.tempfile.TemporaryFile
    files = []

    def owned_file(*args, **kwargs):
        file = native(*args, **kwargs)
        files.append(file)
        return file

    monkeypatch.setattr(deepspeed_engine.tempfile, "TemporaryFile", owned_file)
    monkeypatch.setattr(deepspeed_engine.os, "pwritev", lambda *args: 0)
    with pytest.raises(OSError, match="made no progress"):
        deepspeed_engine._OffloadedGradientStorage().snapshot(torch.ones(55, dtype=torch.float32))
    assert len(files) == 1
    assert files[0].closed


def test_offloaded_file_snapshot_refuses_truncated_restore(monkeypatch, tmp_path):
    monkeypatch.setattr(deepspeed_engine, "_OFFLOADED_SNAPSHOT_CHUNK_BYTES", 64)
    monkeypatch.setattr(deepspeed_engine.tempfile, "gettempdir", lambda: str(tmp_path))
    source = torch.arange(55, dtype=torch.float32)
    snapshot = deepspeed_engine._OffloadedGradientStorage().snapshot(source)
    deepspeed_engine.os.ftruncate(snapshot.file.fileno(), 12)
    with pytest.raises(OSError, match="made no progress"):
        snapshot.restore(torch.empty_like(source))
    snapshot.file.close()


class _PartialGradientModel(torch.nn.Module):
    def __init__(self, *, participation, reentrant):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            torch.nn.Linear(_PARTITION_WIDTH, _PARTITION_WIDTH, bias=False) for _ in range(6)
        )
        self.participation = participation
        self.reentrant = reentrant
        self.forwards = 0

    def forward(self, inputs):
        self.forwards += 1
        for index, layer in enumerate(self.layers):
            if self.participation == "unused" and index in (2, 5):
                continue
            if self.participation == "alternating" and index == (2 if self.forwards % 2 else 5):
                continue
            inputs = checkpoint(layer, inputs, use_reentrant=self.reentrant)
            inputs = torch.nn.functional.gelu(inputs)
        return inputs


def _partition_release_engine(monkeypatch, *, stage, fp32_masters, release, participation=None, reentrant=False):
    set_seed(0)
    if participation is None:
        model = torch.nn.Sequential(
            torch.nn.Linear(_PARTITION_WIDTH, 2 * _PARTITION_WIDTH),
            torch.nn.GELU(),
            torch.nn.Linear(2 * _PARTITION_WIDTH, _PARTITION_WIDTH),
        )
    else:
        model = _PartialGradientModel(participation=participation, reentrant=reentrant)
    spec = TrainingEngineSpec(
        dtype="bfloat16",
        world_size=1,
        fp32_masters=fp32_masters,
        gradient_clipping=0,
        ds_overrides={
            "zero_optimization": {"stage": stage, "offload_optimizer": {"device": "cpu", "pin_memory": False}},
            "zero_force_ds_cpu_optimizer": False,
        },
    )
    ds_config = build_deepspeed_config(spec)
    assert ds_config["data_types"] == {"grad_accum_dtype": "fp32"}
    with monkeypatch.context() as patch:
        if not release:
            patch.setattr(deepspeed_engine, "release_offloaded_grad_partitions", lambda optimizer: False)
        engine, _, _ = initialize_training_engine(
            model,
            ds_config,
            optimizer=OptimizerSettings(name="AdamW", lr=1e-3, fused=False),
            ep_size=1,
            mpu=None,
            model_path="unused",
            rank=0,
        )
    return engine


@require_torch_gpu
@require_deepspeed
@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("fp32_masters", [False, True])
def test_offloaded_partition_release_evicts_nonboundary_gradients(monkeypatch, single_rank_group, stage, fp32_masters):
    import inspect

    engine = _partition_release_engine(monkeypatch, stage=stage, fp32_masters=fp32_masters, release=True)
    retained = inspect.getclosurevars(engine.optimizer.step).nonlocals["retained"]
    copy_grads = engine.optimizer.copy_grads_in_partition
    completed_groups = set()

    def observe_copy(param, group_idx=None):
        if group_idx is None:
            group_idx = engine.optimizer._get_param_group_idx(param)
        result = copy_grads(param, group_idx)
        if all(
            engine.optimizer.params_already_reduced[engine.optimizer.get_param_id(partition_param)]
            for partition_param in engine.optimizer.params_in_partition[group_idx]
        ):
            assert engine.optimizer.averaged_gradients[group_idx] is None
            completed_groups.add(group_idx)
        return result

    monkeypatch.setattr(engine.optimizer, "copy_grads_in_partition", observe_copy)
    engine.set_gradient_accumulation_boundary(False)
    inputs = torch.ones(8, _PARTITION_WIDTH, device="cuda", dtype=torch.bfloat16)
    engine.backward(engine(inputs).float().square().mean())
    remaining = sum(
        partition.flat_partition.numel() * partition.flat_partition.element_size()
        for partition in engine.optimizer.averaged_gradients.values()
        if partition is not None
    )
    print(
        f"NONBOUNDARY_PARTITION_MEMORY stage={stage} fp32_masters={fp32_masters} remaining_bytes={remaining}",
        flush=True,
    )
    assert remaining == 0, "offloaded FP32 sums remain on GPU between accumulation microsteps"
    # ZeRO-1 can defer reduction to the boundary; ZeRO-2 reduces every microstep.
    if stage == 2:
        assert len(retained) == len(engine.optimizer.bit16_groups)
        assert completed_groups == set(range(len(engine.optimizer.bit16_groups)))


@require_torch_gpu
@require_deepspeed
@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("fp32_masters", [False, True])
@pytest.mark.parametrize("boundaries", [(True,), (False, True), (True, True), (True, False)])
@pytest.mark.parametrize("file_snapshots", [False, True])
def test_offloaded_partition_release_preserves_repeated_backwards_and_steps(
    monkeypatch, single_rank_group, stage, fp32_masters, boundaries, file_snapshots
):
    import inspect

    from deepspeed.utils import safe_get_full_grad

    if file_snapshots:
        monkeypatch.setattr(deepspeed_engine, "_OFFLOADED_SNAPSHOT_CHUNK_BYTES", 64)
    evicted = _partition_release_engine(monkeypatch, stage=stage, fp32_masters=fp32_masters, release=True)
    baseline = _partition_release_engine(monkeypatch, stage=stage, fp32_masters=fp32_masters, release=False)
    retained = inspect.getclosurevars(evicted.optimizer.step).nonlocals["retained"]
    inputs = [
        torch.arange(8 * _PARTITION_WIDTH, device="cuda").reshape(8, _PARTITION_WIDTH).bfloat16() / 512,
        torch.full((8, _PARTITION_WIDTH), 0.123456, device="cuda", dtype=torch.bfloat16),
    ]
    # A second window detects snapshots that accidentally survive an optimizer step.
    for window in range(2):
        for index, boundary in enumerate(boundaries):
            losses = []
            for engine in (evicted, baseline):
                engine.set_gradient_accumulation_boundary(boundary)
                loss = engine(inputs[(index + window) % len(inputs)]).float().square().mean()
                losses.append(loss.detach())
                engine.backward(loss)
            torch_assert_equal(*losses)
            groups = list(evicted.optimizer.averaged_gradients.values())
            assert all(partition is None for partition in groups)
            if boundary or stage == 2:
                assert retained
                assert all(
                    isinstance(snapshot, deepspeed_engine._OffloadedGradientFile) is file_snapshots
                    for snapshot in retained.values()
                )
        if not boundaries[-1]:
            for engine in (evicted, baseline):
                engine.optimizer.finalize_gradient_accumulation_boundary()
                engine.set_gradient_accumulation_boundary(True)
        pairs = list(zip(evicted.module.named_parameters(), baseline.module.named_parameters()))
        for (name, parameter), (_, reference) in pairs:
            torch_assert_equal(
                safe_get_full_grad(parameter),
                safe_get_full_grad(reference),
                msg=lambda message, name=name: f"{name} gradient: {message}",
            )
        evicted.step()
        baseline.step()
        assert retained == {}
        torch_assert_equal(
            torch.as_tensor(evicted.optimizer._global_grad_norm),
            torch.as_tensor(baseline.optimizer._global_grad_norm),
        )
        for (name, parameter), (_, reference) in pairs:
            torch_assert_equal(parameter, reference, msg=lambda message, name=name: f"{name}: {message}")
        actual, expected = evicted.optimizer.optimizer.state_dict(), baseline.optimizer.optimizer.state_dict()
        for param_id, state in actual["state"].items():
            for key, value in state.items():
                reference = expected["state"][param_id][key]
                if isinstance(value, torch.Tensor):
                    torch_assert_equal(value, reference)
                else:
                    assert value == reference


@require_torch_gpu
@require_deepspeed
@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("fp32_masters", [False, True])
@pytest.mark.parametrize("reentrant", [False, True])
@pytest.mark.parametrize("participation", ["unused", "alternating"])
@pytest.mark.parametrize("boundaries", [(True, True), (False, True), (True, False)])
def test_offloaded_partition_release_evicts_partial_checkpoint_groups(
    monkeypatch, single_rank_group, stage, fp32_masters, reentrant, participation, boundaries
):
    from deepspeed.utils import safe_get_full_grad

    options = dict(stage=stage, fp32_masters=fp32_masters, participation=participation, reentrant=reentrant)
    evicted = _partition_release_engine(monkeypatch, release=True, **options)
    baseline = _partition_release_engine(monkeypatch, release=False, **options)
    for window in range(2):
        for index, boundary in enumerate(boundaries):
            losses = []
            for engine in (evicted, baseline):
                engine.set_gradient_accumulation_boundary(boundary)
                inputs = torch.full(
                    (8, _PARTITION_WIDTH),
                    0.125 + (index + window) / 32,
                    device="cuda",
                    dtype=torch.bfloat16,
                    requires_grad=True,
                )
                loss = engine(inputs).float().square().mean()
                losses.append(loss.detach().clone())
                engine.backward(loss)
            torch_assert_equal(*losses)
            assert all(partition is None for partition in evicted.optimizer.averaged_gradients.values())
        if not boundaries[-1]:
            for engine in (evicted, baseline):
                engine.optimizer.finalize_gradient_accumulation_boundary()
                engine.set_gradient_accumulation_boundary(True)
        for parameter, reference in zip(evicted.module.parameters(), baseline.module.parameters()):
            actual, expected = safe_get_full_grad(parameter), safe_get_full_grad(reference)
            if actual is None:
                assert expected is None
            else:
                torch_assert_equal(actual, expected)
        evicted.step()
        baseline.step()
        assert evicted.optimizer.accumulated_grads_in_cpu == {}
        torch_assert_equal(
            torch.as_tensor(evicted.optimizer._global_grad_norm),
            torch.as_tensor(baseline.optimizer._global_grad_norm),
        )
        for parameter, reference in zip(evicted.module.parameters(), baseline.module.parameters()):
            torch_assert_equal(parameter, reference)
        actual, expected = evicted.optimizer.optimizer.state_dict(), baseline.optimizer.optimizer.state_dict()
        assert actual["param_groups"] == expected["param_groups"]
        assert actual["state"].keys() == expected["state"].keys()
        for param_id, state in actual["state"].items():
            for key, value in state.items():
                reference = expected["state"][param_id][key]
                if isinstance(value, torch.Tensor):
                    torch_assert_equal(value, reference)
                else:
                    assert value == reference


@require_torch_gpu
@require_deepspeed
@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("fp32_masters", [False, True])
@pytest.mark.parametrize("file_snapshots", [False, True])
def test_offloaded_partition_reset_preserves_checkpoint_continuation(
    monkeypatch, single_rank_group, tmp_path, stage, fp32_masters, file_snapshots
):
    import inspect

    if file_snapshots:
        monkeypatch.setattr(deepspeed_engine, "_OFFLOADED_SNAPSHOT_CHUNK_BYTES", 64)
    continued = _partition_release_engine(monkeypatch, stage=stage, fp32_masters=fp32_masters, release=True)
    inputs = torch.full((8, _PARTITION_WIDTH), 0.125, device="cuda", dtype=torch.bfloat16)

    def update(engine):
        for boundary in (False, True, True):
            engine.set_gradient_accumulation_boundary(boundary)
            loss = engine(inputs).float().square().mean()
            engine.backward(loss)
        retained = inspect.getclosurevars(engine.optimizer.step).nonlocals["retained"]
        assert retained
        assert all(
            isinstance(snapshot, deepspeed_engine._OffloadedGradientFile) is file_snapshots
            for snapshot in retained.values()
        )
        engine.step()
        assert retained == {}
        assert engine.optimizer.accumulated_grads_in_cpu == {}
        return loss.detach(), torch.as_tensor(engine.optimizer._global_grad_norm)

    update(continued)
    checkpoint_root = str(tmp_path / "checkpoint")
    assert continued.save_checkpoint(checkpoint_root, tag="first-update")
    expected_loss, expected_norm = update(continued)

    restored = _partition_release_engine(monkeypatch, stage=stage, fp32_masters=fp32_masters, release=True)
    loaded, _ = restored.load_checkpoint(checkpoint_root, tag="first-update")
    assert loaded is not None
    actual_loss, actual_norm = update(restored)
    torch_assert_equal(actual_loss, expected_loss)
    torch_assert_equal(actual_norm, expected_norm)
    for parameter, reference in zip(restored.module.parameters(), continued.module.parameters()):
        torch_assert_equal(parameter, reference)
    for parameter, reference in zip(
        restored.optimizer.single_partition_of_fp32_groups, continued.optimizer.single_partition_of_fp32_groups
    ):
        torch_assert_equal(parameter, reference)
    actual = restored.optimizer.optimizer.state_dict()
    expected = continued.optimizer.optimizer.state_dict()
    assert actual["param_groups"] == expected["param_groups"]
    assert actual["state"].keys() == expected["state"].keys()
    for param_id, state in actual["state"].items():
        assert state.keys() == expected["state"][param_id].keys()
        for key, value in state.items():
            reference = expected["state"][param_id][key]
            if isinstance(value, torch.Tensor):
                torch_assert_equal(value, reference)
            else:
                assert value == reference


@require_torch_gpu
@require_deepspeed
@pytest.mark.parametrize("file_snapshots", [False, True])
def test_offloaded_partition_release_clears_snapshots_after_overflow(monkeypatch, single_rank_group, file_snapshots):
    import inspect

    if file_snapshots:
        monkeypatch.setattr(deepspeed_engine, "_OFFLOADED_SNAPSHOT_CHUNK_BYTES", 64)
    evicted = _partition_release_engine(monkeypatch, stage=2, fp32_masters=True, release=True)
    baseline = _partition_release_engine(monkeypatch, stage=2, fp32_masters=True, release=False)
    retained = inspect.getclosurevars(evicted.optimizer.step).nonlocals["retained"]
    inputs = torch.ones(8, _PARTITION_WIDTH, device="cuda", dtype=torch.bfloat16)
    for overflow in (True, False):
        # The control never accumulates the discarded gradients: installed DeepSpeed can retain its compact buffer
        # after overflow, so repeating that path would compare against a stale-gradient update.
        engines = (evicted,) if overflow else (evicted, baseline)
        for engine in engines:
            engine.set_gradient_accumulation_boundary(True)
            engine.backward(engine(inputs).float().square().mean())
            if engine is evicted:
                assert retained
                assert all(
                    isinstance(snapshot, deepspeed_engine._OffloadedGradientFile) is file_snapshots
                    for snapshot in retained.values()
                )
            if overflow:
                engine.optimizer.check_grad_overflow = True

                def force_overflow(**kwargs):
                    engine.optimizer.overflow = True

                with monkeypatch.context() as patch:
                    patch.setattr(engine.optimizer, "check_overflow", force_overflow)
                    engine.step()
            else:
                engine.step()
            if engine is evicted:
                assert retained == {}
        for parameter, reference in zip(evicted.module.parameters(), baseline.module.parameters()):
            torch_assert_equal(parameter, reference)


@require_torch_gpu
@require_deepspeed
@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("fp32_masters", [False, True])
def test_offloaded_partition_snapshots_allocate_each_group_once_per_update(
    monkeypatch, single_rank_group, stage, fp32_masters
):
    import inspect
    from collections import Counter

    from torch.utils._python_dispatch import TorchDispatchMode

    engine = _partition_release_engine(monkeypatch, stage=stage, fp32_masters=fp32_masters, release=True)
    inputs = torch.ones(8, _PARTITION_WIDTH, device="cuda", dtype=torch.bfloat16)
    snapshots = Counter()
    retained = inspect.getclosurevars(engine.optimizer.step).nonlocals["retained"]

    class CountSnapshots(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            group_idx = None
            if func == torch.ops.aten._to_copy.default and args[0].device.type == "cuda":
                for index, partition in engine.optimizer.averaged_gradients.items():
                    if (
                        partition is not None
                        and args[0].numel() == partition.flat_partition.numel()
                        and args[0].data_ptr() == partition.flat_partition.data_ptr()
                    ):
                        group_idx = index
                        break
            result = func(*args, **(kwargs or {}))
            if group_idx is not None and result.device.type == "cpu":
                snapshots[group_idx] += 1
            return result

    for _ in range(2):
        snapshots.clear()
        storage_ids = None
        for _ in range(3):
            engine.set_gradient_accumulation_boundary(True)
            with CountSnapshots():
                engine.backward(engine(inputs).float().square().mean())
            current_ids = {index: snapshot.data_ptr() for index, snapshot in retained.items()}
            if storage_ids is None:
                storage_ids = current_ids
            else:
                assert current_ids == storage_ids
        expected = {index: 1 for index in range(len(engine.optimizer.bit16_groups))}
        print(f"PARTITION_SNAPSHOT_ALLOCATIONS actual={dict(snapshots)} expected={expected}", flush=True)
        assert snapshots == expected
        engine.step()
        assert retained == {}


@require_torch_gpu
@require_deepspeed
@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("fp32_masters", [False, True])
def test_offloaded_partition_snapshots_end_before_moment_allocation(
    monkeypatch, single_rank_group, stage, fp32_masters
):
    import inspect
    import weakref

    engine = _partition_release_engine(monkeypatch, stage=stage, fp32_masters=fp32_masters, release=True)
    engine.set_gradient_accumulation_boundary(True)
    inputs = torch.ones(8, _PARTITION_WIDTH, device="cuda", dtype=torch.bfloat16)
    engine.backward(engine(inputs).float().square().mean())
    retained = inspect.getclosurevars(engine.optimizer.step).nonlocals["retained"]
    snapshots = {index: weakref.ref(snapshot) for index, snapshot in retained.items()}
    assert len(snapshots) == len(engine.optimizer.bit16_groups)
    native_step = engine.optimizer.optimizer.step
    consumed = []

    def observe_step(*args, **kwargs):
        selected = engine.optimizer.optimizer.param_groups[0]["params"][0]
        group_idx = next(
            index
            for index, master in enumerate(engine.optimizer.single_partition_of_fp32_groups)
            if master is selected
        )
        live_bytes = sum(
            snapshot().numel() * snapshot().element_size() for snapshot in snapshots.values() if snapshot() is not None
        )
        print(f"PARTITION_SNAPSHOT_MOMENT_ALLOCATION group={group_idx} retained_bytes={live_bytes}", flush=True)
        assert snapshots[group_idx]() is None, "FP32 snapshot overlaps lazy optimizer-state allocation"
        consumed.append(group_idx)
        return native_step(*args, **kwargs)

    monkeypatch.setattr(engine.optimizer.optimizer, "step", observe_step)
    engine.step()
    assert consumed == list(range(len(snapshots)))
    assert all(snapshot() is None for snapshot in snapshots.values())


@require_torch_gpu
@require_deepspeed
@pytest.mark.parametrize("during_update", [False, True])
def test_offloaded_partition_failure_retains_only_unconsumed_snapshots(monkeypatch, single_rank_group, during_update):
    import inspect

    engine = _partition_release_engine(monkeypatch, stage=2, fp32_masters=False, release=True)
    engine.set_gradient_accumulation_boundary(True)
    inputs = torch.ones(8, _PARTITION_WIDTH, device="cuda", dtype=torch.bfloat16)
    engine.backward(engine(inputs).float().square().mean())
    retained = inspect.getclosurevars(engine.optimizer.step).nonlocals["retained"]
    original_ids = {index: id(snapshot) for index, snapshot in retained.items()}
    assert len(original_ids) > 1
    failure = RuntimeError("injected native step failure")
    consumed = []

    def fail_before_update(*args, **kwargs):
        raise failure

    def fail_during_update(*args, **kwargs):
        selected = engine.optimizer.optimizer.param_groups[0]["params"][0]
        group_idx = next(
            index
            for index, master in enumerate(engine.optimizer.single_partition_of_fp32_groups)
            if master is selected
        )
        consumed.append(group_idx)
        raise failure

    if during_update:
        monkeypatch.setattr(engine.optimizer.optimizer, "step", fail_during_update)
    else:
        monkeypatch.setattr(engine.optimizer, "unscale_and_clip_grads", fail_before_update)
    with pytest.raises(RuntimeError) as caught:
        engine.step()
    assert caught.value is failure
    expected = {index: identity for index, identity in original_ids.items() if index not in consumed}
    assert {index: id(snapshot) for index, snapshot in retained.items()} == expected


def test_partition_release_leaves_non_zero_optimizers_unchanged():
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))])
    original = set(vars(optimizer))
    assert not deepspeed_engine.release_offloaded_grad_partitions(optimizer)
    assert set(vars(optimizer)) == original


@require_deepspeed
@pytest.mark.parametrize("compact", [False, True])
def test_partition_reset_preserves_native_noncompact_accumulators(monkeypatch, compact):
    class NativeOptimizer:
        def __init__(self):
            self.cpu_offload = True
            self.bit16_groups = [[0], [1]]
            self.accumulated_grads_in_cpu = {0: object(), 1: object()}
            self.averaged_gradients = {0: object(), 1: object()}
            self.reset_calls = []

        def _use_compact_grad_partition_buffer(self, group_idx):
            return compact and group_idx == 0

        def get_param_id(self, param):
            return param

        def copy_grads_in_partition(self, param, group_idx=None):
            return param, group_idx

        def _get_compact_grad_partition_buffer(self, group_idx):
            return self.averaged_gradients[group_idx]

        def step(self):
            return "stepped"

        def reset_cpu_buffers(self, *args, **kwargs):
            self.reset_calls.append((args, kwargs))
            return "reset"

        def _optimizer_step(self, group_idx):
            return group_idx

        def independent_gradient_partition_epilogue(self):
            return "epilogue"

    monkeypatch.setattr("deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer", NativeOptimizer)
    optimizer = NativeOptimizer()
    original_attributes = dict(vars(optimizer))
    original_accumulators = dict(optimizer.accumulated_grads_in_cpu)
    original_partitions = dict(optimizer.averaged_gradients)
    assert deepspeed_engine.release_offloaded_grad_partitions(optimizer) is compact
    if not compact:
        assert vars(optimizer) == original_attributes
    assert optimizer.reset_cpu_buffers("request", boundary=True) == "reset"
    assert optimizer.reset_calls == [(("request",), {"boundary": True})]
    expected_ids = {1} if compact else {0, 1}
    assert optimizer.accumulated_grads_in_cpu.keys() == expected_ids
    for param_id in expected_ids:
        assert optimizer.accumulated_grads_in_cpu[param_id] is original_accumulators[param_id]
    assert optimizer.averaged_gradients[1] is original_partitions[1]
    if compact:
        assert optimizer.averaged_gradients[0] is None
    else:
        assert optimizer.averaged_gradients[0] is original_partitions[0]


@require_deepspeed
@pytest.mark.parametrize("boundary", [False, True])
@pytest.mark.parametrize("file_snapshots", [False, True])
@pytest.mark.parametrize("failure_phase", [None, "native", "wait"])
def test_partition_epilogue_waits_for_partial_copies_and_restores_snapshot(
    monkeypatch, boundary, file_snapshots, failure_phase
):
    import inspect
    from types import SimpleNamespace

    if file_snapshots:
        monkeypatch.setattr(deepspeed_engine, "_OFFLOADED_SNAPSHOT_CHUNK_BYTES", 4)
    events = []
    failure = RuntimeError("injected epilogue failure")

    class NativeOptimizer:
        def __init__(self):
            self.cpu_offload = True
            self.bit16_groups = [[0, 1], [2]]
            self.accumulated_grads_in_cpu = {}
            self.averaged_gradients = {
                0: SimpleNamespace(flat_partition=torch.tensor([1.0, 2.0])),
                1: object(),
            }
            self.pending = False

        def _use_compact_grad_partition_buffer(self, group_idx):
            return group_idx == 0

        def get_param_id(self, param):
            return param

        def copy_grads_in_partition(self, param, group_idx=None):
            return None

        def _get_compact_grad_partition_buffer(self, group_idx):
            if self.averaged_gradients[group_idx] is None:
                self.averaged_gradients[group_idx] = SimpleNamespace(flat_partition=torch.zeros(2))
            return self.averaged_gradients[group_idx]

        def is_gradient_accumulation_boundary(self):
            return boundary

        def independent_gradient_partition_epilogue(self, *args, **kwargs):
            events.append(("native", args, kwargs))
            if failure_phase == "native":
                raise failure
            self.pending = True
            return "reduced"

        def _wait_for_offload_copies(self):
            events.append(("wait",))
            assert self.pending
            if failure_phase == "wait":
                raise failure
            self.averaged_gradients[0].flat_partition.add_(2)
            self.pending = False

        def step(self):
            return None

        def reset_cpu_buffers(self):
            return None

        def _optimizer_step(self, group_idx):
            return group_idx

    monkeypatch.setattr("deepspeed.runtime.zero.stage_1_and_2.DeepSpeedZeroOptimizer", NativeOptimizer)
    optimizer = NativeOptimizer()
    native_partition = optimizer.averaged_gradients[0]
    noncompact_partition = optimizer.averaged_gradients[1]
    assert deepspeed_engine.release_offloaded_grad_partitions(optimizer)
    retained = inspect.getclosurevars(optimizer.step).nonlocals["retained"]
    if failure_phase in ("native", "wait"):
        with pytest.raises(RuntimeError) as caught:
            optimizer.independent_gradient_partition_epilogue("request", complete=True)
        assert caught.value is failure
        assert events == [("native", ("request",), {"complete": True})] + (
            [("wait",)] if failure_phase == "wait" else []
        )
        assert optimizer.averaged_gradients[0] is native_partition
        assert optimizer.averaged_gradients[1] is noncompact_partition
        assert retained == {}
        return
    assert optimizer.independent_gradient_partition_epilogue("request", complete=True) == "reduced"
    assert events == [("native", ("request",), {"complete": True}), ("wait",)]
    assert optimizer.averaged_gradients[1] is noncompact_partition
    assert optimizer.averaged_gradients[0] is None
    snapshot = retained[0]
    assert isinstance(snapshot, deepspeed_engine._OffloadedGradientFile) is file_snapshots
    rebuilt = optimizer._get_compact_grad_partition_buffer(0)
    assert rebuilt is not native_partition
    torch_assert_equal(rebuilt.flat_partition, torch.tensor([3.0, 4.0]))
    optimizer.independent_gradient_partition_epilogue()
    assert retained[0] is snapshot
    torch_assert_equal(optimizer._get_compact_grad_partition_buffer(0).flat_partition, torch.tensor([5.0, 6.0]))
    optimizer.reset_cpu_buffers()
    assert retained == {}
