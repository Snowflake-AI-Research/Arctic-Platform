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
    assert resolve_optimizer_name("torch_adamw", _OFFLOAD) == "torch_adamw"
    assert resolve_optimizer_name("cpu_adam", _ZERO2) == "deepspeed_cpu_adam"


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
def test_initialize_splits_expert_groups_and_fuses_adam_for_glm(monkeypatch, tmp_path, model_type):
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
        optimizer=OptimizerSettings(),
        ep_size=2,
        mpu=None,
        model_path=str(tmp_path),
        rank=0,
    )

    assert engine == "engine"
    assert optimizer is calls["optimizer"]
    assert choice["implementation"] == "fused_adam"
    assert calls["split"]
    assert optimizer.param_groups == [{"params": [], "name": "experts"}]


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


@pytest.mark.parametrize("failure", ["mkdir", "open"])
def test_offloaded_file_snapshot_falls_back_from_unwritable_scratch(monkeypatch, tmp_path, failure):
    directory = tmp_path / "unavailable"
    monkeypatch.setenv("ARCTIC_OFFLOADED_GRADIENT_TMPDIR", str(directory))
    monkeypatch.setattr(deepspeed_engine.tempfile, "gettempdir", lambda: str(tmp_path))
    native_mkdir = deepspeed_engine.Path.mkdir
    native_file = deepspeed_engine.tempfile.TemporaryFile
    opened = []

    def mkdir(path, *args, **kwargs):
        if path == directory and failure == "mkdir":
            raise PermissionError("unwritable scratch directory")
        return native_mkdir(path, *args, **kwargs)

    def temporary_file(*args, **kwargs):
        opened.append(str(kwargs["dir"]))
        if str(kwargs["dir"]) == str(directory) and failure == "open":
            raise PermissionError("unwritable scratch file")
        return native_file(*args, **kwargs)

    monkeypatch.setattr(deepspeed_engine.Path, "mkdir", mkdir)
    monkeypatch.setattr(deepspeed_engine.tempfile, "TemporaryFile", temporary_file)
    source = torch.arange(8, dtype=torch.float32)
    snapshot = deepspeed_engine._OffloadedGradientFile(source, deepspeed_engine._OffloadedGradientStorage())
    try:
        snapshot.copy_(source)
        restored = torch.empty_like(source)
        snapshot.restore(restored)
        torch_assert_equal(restored, source)
        assert opened[-1] == str(tmp_path)
    finally:
        snapshot.file.close()


def test_offloaded_file_snapshot_honors_scratch_override(monkeypatch, tmp_path):
    directory = tmp_path / "scratch"
    monkeypatch.setenv("ARCTIC_OFFLOADED_GRADIENT_TMPDIR", str(directory))
    native_file = deepspeed_engine.tempfile.TemporaryFile
    opened = []

    def temporary_file(*args, **kwargs):
        opened.append(kwargs["dir"])
        return native_file(*args, **kwargs)

    monkeypatch.setattr(deepspeed_engine.tempfile, "TemporaryFile", temporary_file)
    source = torch.arange(8, dtype=torch.float32)
    snapshot = deepspeed_engine._OffloadedGradientFile(source, deepspeed_engine._OffloadedGradientStorage())
    snapshot.file.close()
    assert opened == [directory]


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
            "data_types": {"grad_accum_dtype": "fp32"},
        },
    )
    ds_config = build_deepspeed_config(spec)
    assert ds_config["data_types"] == {"grad_accum_dtype": "fp32"}
    with monkeypatch.context() as patch:
        if not release:
            patch.setattr(
                deepspeed_engine, "release_offloaded_grad_partitions", lambda optimizer: False, raising=False
            )
        engine, _, _ = initialize_training_engine(
            model,
            ds_config,
            optimizer=OptimizerSettings(name="AdamW", lr=1e-3, fused=True),
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
@pytest.mark.parametrize("release_mode", ["epilogue", "copy"])
def test_partition_epilogue_waits_for_partial_copies_and_restores_snapshot(
    monkeypatch, boundary, file_snapshots, failure_phase, release_mode
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
            self.params_in_partition = {0: [0, 1]}
            self.params_already_reduced = {0: True, 1: True}

        def _use_compact_grad_partition_buffer(self, group_idx):
            return group_idx == 0

        def get_param_id(self, param):
            return param

        def copy_grads_in_partition(self, param, group_idx=None):
            return NativeOptimizer.independent_gradient_partition_epilogue(self, "request", complete=True)

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

    def reduce():
        if release_mode == "copy":
            optimizer.copy_grads_in_partition(0, 0)
        else:
            assert optimizer.independent_gradient_partition_epilogue("request", complete=True) == "reduced"

    if failure_phase in ("native", "wait"):
        with pytest.raises(RuntimeError) as caught:
            reduce()
        assert caught.value is failure
        assert events == [("native", ("request",), {"complete": True})] + (
            [("wait",)] if failure_phase == "wait" else []
        )
        assert optimizer.averaged_gradients[0] is native_partition
        assert optimizer.averaged_gradients[1] is noncompact_partition
        assert retained == {}
        return
    reduce()
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


@require_torch_gpu
@require_deepspeed
@pytest.mark.parametrize("stage", [1, 2])
@pytest.mark.parametrize("fp32_masters", [False, True])
def test_offloaded_partition_release_evicts_boundary_gradients(monkeypatch, single_rank_group, stage, fp32_masters):
    engine = _partition_release_engine(monkeypatch, stage=stage, fp32_masters=fp32_masters, release=True)
    engine.set_gradient_accumulation_boundary(True)
    inputs = torch.ones(8, _PARTITION_WIDTH, device="cuda", dtype=torch.bfloat16)
    engine.backward(engine(inputs).float().square().mean())
    remaining = sum(
        partition.flat_partition.numel() * partition.flat_partition.element_size()
        for partition in engine.optimizer.averaged_gradients.values()
        if partition is not None
    )
    print(
        f"BOUNDARY_PARTITION_MEMORY stage={stage} fp32_masters={fp32_masters} remaining_bytes={remaining}", flush=True
    )
    assert remaining == 0, "completed offload partitions remain on GPU between backward requests"
