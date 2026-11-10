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

from copy import deepcopy

import pytest
import torch

from arctic_platform.common.deepspeed_engine import TrainingEngineSpec
from arctic_platform.common.deepspeed_engine import apply_default_grad_accum_dtype
from arctic_platform.common.deepspeed_engine import build_deepspeed_config
from arctic_platform.testing_utils import get_unique_port_number
from arctic_platform.testing_utils import require_torch_gpu
from arctic_platform.testing_utils import reserve_free_port
from arctic_platform.testing_utils import set_seed
from arctic_platform.testing_utils import torch_assert_equal


@pytest.fixture
def worker():
    from arctic_platform.common.deepspeed_worker import DeepSpeedWorker

    return object.__new__(DeepSpeedWorker.__ray_metadata__.modified_class)


_ZERO2 = {"zero_optimization": {"stage": 2}}
_REFUSED_ZERO = [{"stage": 0}, {"stage": 1, "compute_grad_norm": False}]


def _entry_point_config(worker, entry_point, overrides, dtype="bfloat16"):
    if entry_point == "builder":
        return build_deepspeed_config(
            TrainingEngineSpec(world_size=1, dtype=dtype, fp32_masters=True, ds_overrides=overrides)
        )
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "bf16": {"enabled": dtype in ("bf16", "bfloat16")},
        "fp16": {"enabled": dtype in ("fp16", "float16")},
        **deepcopy(_ZERO2),
    }
    config.update(overrides)
    return worker.ds_training_config({}, config, {})


@pytest.mark.parametrize("job_config", [{}, {"training_config": {}}])
def test_missing_data_types_retains_worker_fp32_accumulation_default(worker, job_config):
    config = deepcopy(_ZERO2)
    assert worker.ds_training_config(job_config, config, {}) is config
    assert config["data_types"] == {"grad_accum_dtype": "fp32"}


@pytest.mark.parametrize(
    "data_types", [{"grad_accum_dtype": "bf16"}, {"grad_accum_dtype": "fp32", "other_setting": True}]
)
@pytest.mark.parametrize("zero", [{"stage": 2}, *_REFUSED_ZERO])
@pytest.mark.parametrize("entry_point", ["builder", "worker"])
def test_explicit_accumulation_precision_is_preserved(worker, entry_point, data_types, zero):
    supplied = {"data_types": deepcopy(data_types), "zero_optimization": deepcopy(zero)}
    config = _entry_point_config(worker, entry_point, supplied)
    assert config["data_types"] == data_types
    assert config["data_types"] is supplied["data_types"]
    assert supplied == {"data_types": data_types, "zero_optimization": zero}


@pytest.mark.parametrize("data_types", [None, {}, {"other_setting": True}, {"grad_accum_dtype": None}])
@pytest.mark.parametrize(
    "entry_point, dtype",
    [("builder", "bfloat16"), ("worker", "bfloat16"), ("worker", "float16"), ("worker", "float32")],
)
def test_supplied_block_without_accumulation_precision_gets_fp32_default(worker, entry_point, dtype, data_types):
    supplied = {"data_types": deepcopy(data_types)}
    config = _entry_point_config(worker, entry_point, supplied, dtype)
    assert config["data_types"] == {**(data_types or {}), "grad_accum_dtype": "fp32"}
    assert supplied == {"data_types": data_types}


@pytest.mark.parametrize("job_config", [{}, {"training_config": {}}])
def test_worker_without_zero_does_not_insert_accumulation_precision(worker, job_config):
    config = {}
    assert worker.ds_training_config(job_config, config, {}) is config
    assert "data_types" not in config


@pytest.mark.parametrize("data_types", [{}, {"grad_accum_dtype": "bf16"}, {"grad_accum_dtype": "fp32"}])
def test_worker_without_zero_preserves_supplied_accumulation_block(worker, data_types):
    config = {"data_types": deepcopy(data_types)}
    worker.ds_training_config({}, config, {})
    assert config["data_types"] == data_types


@pytest.mark.parametrize("zero", _REFUSED_ZERO)
def test_worker_leaves_accumulation_unset_where_deepspeed_refuses_fp32(worker, zero):
    config = {"zero_optimization": deepcopy(zero)}
    worker.ds_training_config({}, config, {})
    assert config["bf16"] == {"enabled": True}
    assert "data_types" not in config


@pytest.mark.parametrize("entry_point", ["builder", "worker"])
@pytest.mark.parametrize(
    "dtype, overrides, model_dtype, accumulation_dtype",
    [
        ("float16", {}, torch.float16, torch.float16),
        ("fp16", {}, torch.float16, torch.float16),
        ("float32", {}, torch.float32, torch.float32),
        ("fp32", {}, torch.float32, torch.float32),
        (
            "bfloat16",
            {"bf16": {"enabled": False}, "fp16": {"enabled": True}},
            torch.float16,
            torch.float16,
        ),
        ("bfloat16", {"bf16": {"enabled": False}}, torch.float32, torch.float32),
        (
            "float16",
            {"bf16": {"enabled": True}, "fp16": {"enabled": False}, "communication_data_type": "fp32"},
            torch.bfloat16,
            torch.float32,
        ),
        (
            "float32",
            {"bf16": {"enabled": True}, "communication_data_type": "fp32"},
            torch.bfloat16,
            torch.float32,
        ),
        ("float16", {"data_types": {"grad_accum_dtype": "fp32"}}, torch.float16, torch.float32),
        ("float32", {"data_types": {"grad_accum_dtype": "bf16"}}, torch.float32, torch.bfloat16),
    ],
)
def test_effective_precision_controls_default_and_preserves_explicit_dtype(
    worker, entry_point, dtype, overrides, model_dtype, accumulation_dtype
):
    supplied = deepcopy(overrides)
    config = _entry_point_config(worker, entry_point, supplied, dtype)
    if entry_point == "worker" and "data_types" not in overrides:
        accumulation_dtype = torch.float32
    assert supplied == overrides
    if entry_point == "builder" and model_dtype != torch.bfloat16 and "data_types" not in overrides:
        assert "data_types" not in config
    else:
        expected_accumulation = "bf16" if accumulation_dtype == torch.bfloat16 else "fp32"
        assert config["data_types"] == {"grad_accum_dtype": expected_accumulation}
    assert _resolved_engine(config).get_data_types() == (model_dtype, accumulation_dtype)


@pytest.mark.parametrize(
    "zero",
    [
        {"stage": 0},
        {"stage": 1},
        {"stage": 1, "compute_grad_norm": False},
        {"stage": 1, "compute_grad_norm": False, "offload_optimizer": {"device": "cpu"}},
        {"stage": 2},
        {"stage": 2, "offload_optimizer": {"device": "cpu"}},
        {"stage": 3},
    ],
)
@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
def test_default_is_written_only_for_supported_bfloat16_accumulation(zero, dtype):
    """Check the policy against the installed DeepSpeed optimizer sanity check, without initializing ranks."""
    overrides = {"zero_optimization": deepcopy(zero), "zero_allow_untested_optimizer": True}
    defaulted = build_deepspeed_config(
        TrainingEngineSpec(world_size=1, dtype=dtype, fp32_masters=True, ds_overrides=overrides)
    )
    explicit = {**defaulted, "data_types": {"grad_accum_dtype": "fp32"}}
    engine = _resolved_engine(explicit)
    try:
        engine._do_optimizer_sanity_check(torch.optim.AdamW([torch.nn.Parameter(torch.zeros(8))]))
        accepted = True
    except (NotImplementedError, ValueError):
        accepted = False
    should_default = accepted and dtype == "bfloat16"
    assert ("data_types" in defaulted) is should_default
    if should_default:
        assert defaulted["data_types"] == {"grad_accum_dtype": "fp32"}
    model_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[dtype]
    assert _resolved_engine(defaulted).get_data_types() == (
        model_dtype,
        torch.float32 if should_default else model_dtype,
    )


def _resolved_engine(config):
    from deepspeed.runtime.config import DeepSpeedConfig
    from deepspeed.runtime.engine import DeepSpeedEngine

    engine = DeepSpeedEngine.__new__(DeepSpeedEngine)
    torch.nn.Module.__init__(engine)
    object.__setattr__(engine, "_config", DeepSpeedConfig(config))
    object.__setattr__(engine, "pipeline_parallelism", False)
    object.__setattr__(engine, "global_rank", 0)
    engine.optimizer = None
    engine.checkpoint_engine = None
    return engine


@pytest.mark.parametrize(
    "raw_boolean, resolved_boolean",
    [(True, True), (False, False), ("true", True), ("false", False), (1, True), (0, False)],
)
@pytest.mark.parametrize("entry_point", ["builder", "worker"])
def test_typed_grad_norm_policy_matches_real_deepspeed_sanity_check(
    worker, raw_boolean, resolved_boolean, entry_point
):
    from deepspeed.runtime.zero.config import get_zero_config

    zero = {"stage": 1, "compute_grad_norm": raw_boolean}
    overrides = {
        "zero_optimization": zero,
        "gradient_clipping": 0.0,
        "zero_allow_untested_optimizer": True,
    }
    assert get_zero_config(overrides).compute_grad_norm is resolved_boolean
    defaulted = _entry_point_config(worker, entry_point, deepcopy(overrides))
    assert ("data_types" in defaulted) is resolved_boolean
    engine = _resolved_engine(defaulted)
    assert engine._config.zero_config.compute_grad_norm is resolved_boolean
    assert engine.get_data_types() == (torch.bfloat16, torch.float32 if resolved_boolean else torch.bfloat16)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(8))])
    engine._do_optimizer_sanity_check(optimizer)
    explicit_engine = _resolved_engine({**defaulted, "data_types": {"grad_accum_dtype": "fp32"}})
    if resolved_boolean:
        explicit_engine._do_optimizer_sanity_check(optimizer)
    else:
        with pytest.raises(ValueError, match="compute_grad_norm=false"):
            explicit_engine._do_optimizer_sanity_check(optimizer)


@pytest.mark.parametrize(
    "raw_boolean, resolved_boolean",
    [(True, True), (False, False), ("true", True), ("false", False), (1, True), (0, False)],
)
@pytest.mark.parametrize("precision", ["bf16", "bfloat16", "fp16"])
def test_typed_precision_policy_matches_real_deepspeed_parser(precision, raw_boolean, resolved_boolean):
    config = {
        "train_micro_batch_size_per_gpu": 1,
        "zero_optimization": {"stage": 2},
        precision: {"enabled": raw_boolean},
    }
    apply_default_grad_accum_dtype(config, bf16_only=True)
    should_default = resolved_boolean and precision != "fp16"
    assert ("data_types" in config) is should_default
    engine = _resolved_engine(config)
    parsed = engine._config.float16_config if precision == "fp16" else engine._config.bfloat16_config
    assert parsed.enabled is resolved_boolean
    model_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    expected_dtype = model_dtype if resolved_boolean else torch.float32
    assert engine.get_data_types() == (expected_dtype, torch.float32 if should_default else expected_dtype)
    engine._do_optimizer_sanity_check(torch.optim.AdamW([torch.nn.Parameter(torch.zeros(8))]))


@pytest.mark.parametrize("entry_point", ["builder", "worker"])
@pytest.mark.parametrize(
    "zero, resolved_stage, cpu_offload, fp32",
    [
        ({"stage": "auto"}, 0, False, False),
        (True, 1, False, True),
        (False, 0, False, False),
        ({"stage": 1, "compute_grad_norm": False, "cpu_offload": True}, 1, True, True),
        ({"stage": 1, "compute_grad_norm": "false", "cpu_offload": "true"}, 1, True, True),
        ({"stage": 1, "compute_grad_norm": 0, "cpu_offload": 1}, 1, True, True),
        ({"stage": 1, "compute_grad_norm": False, "cpu_offload": False}, 1, False, False),
        ({"stage": 1, "compute_grad_norm": "false", "cpu_offload": "false"}, 1, False, False),
        ({"stage": 1, "compute_grad_norm": 0, "cpu_offload": 0}, 1, False, False),
    ],
)
def test_auto_and_legacy_zero_policy_matches_real_deepspeed_parser(
    worker, entry_point, zero, resolved_stage, cpu_offload, fp32
):
    from deepspeed.runtime.zero.config import get_zero_config

    overrides = {
        "zero_optimization": deepcopy(zero),
        "gradient_clipping": 0.0,
        "zero_allow_untested_optimizer": True,
    }
    original = deepcopy(overrides)
    parsed = get_zero_config(overrides)
    assert parsed.stage == resolved_stage
    assert (parsed.offload_optimizer is not None) is cpu_offload
    if cpu_offload:
        assert parsed.offload_optimizer.device == "cpu"
    defaulted = _entry_point_config(worker, entry_point, overrides)
    assert overrides == original
    assert defaulted["zero_optimization"] == zero
    assert ("data_types" in defaulted) is fp32
    engine = _resolved_engine(defaulted)
    assert engine._config.zero_config.stage == resolved_stage
    assert engine.zero_cpu_offload() is cpu_offload
    assert engine.get_data_types() == (torch.bfloat16, torch.float32 if fp32 else torch.bfloat16)
    engine._do_optimizer_sanity_check(torch.optim.AdamW([torch.nn.Parameter(torch.zeros(8))]))


@pytest.mark.parametrize("entry_point", ["helper", "builder"])
def test_malformed_stage_retains_deepspeed_structured_validation_error(entry_point):
    from deepspeed.runtime.zero.config import get_zero_config
    from pydantic import ValidationError

    config = {"zero_optimization": {"stage": "invalid"}}
    original = deepcopy(config)
    with pytest.raises(ValidationError) as native:
        get_zero_config(config)
    with pytest.raises(ValidationError) as candidate:
        if entry_point == "helper":
            apply_default_grad_accum_dtype(config)
        else:
            build_deepspeed_config(TrainingEngineSpec(world_size=1, ds_overrides=config))
    assert candidate.value.errors() == native.value.errors()
    assert config == original


@pytest.mark.parametrize("data_types", ["invalid", [], 17])
def test_invalid_accumulation_blocks_are_left_for_deepspeed_validation(data_types):
    config = {**deepcopy(_ZERO2), "data_types": data_types}
    apply_default_grad_accum_dtype(config)
    assert config["data_types"] is data_types


def test_defaulting_is_idempotent_and_preserves_the_callers_block():
    supplied = {"other_setting": True}
    config = {**deepcopy(_ZERO2), "bf16": {"enabled": True}, "data_types": supplied}
    apply_default_grad_accum_dtype(config)
    defaulted = config["data_types"]
    apply_default_grad_accum_dtype(config)
    assert config["data_types"] is defaulted
    assert defaulted == {"other_setting": True, "grad_accum_dtype": "fp32"}
    assert supplied == {"other_setting": True}


def test_bfloat16_alias_keeps_zero0_native_accumulation_default():
    config = {"bfloat16": {"enabled": True}}
    apply_default_grad_accum_dtype(config)
    assert config == {"bfloat16": {"enabled": True}}


@pytest.mark.parametrize("dtype", ["bf16", "bfloat16"])
@pytest.mark.parametrize("fp32_masters", [False, True])
def test_deepspeed_resolves_float32_accumulation_without_changing_model_precision(fp32_masters, dtype):
    ds_config = build_deepspeed_config(TrainingEngineSpec(world_size=1, dtype=dtype, fp32_masters=fp32_masters))
    assert ds_config["data_types"] == {"grad_accum_dtype": "fp32"}
    assert _resolved_engine(ds_config).get_data_types() == (torch.bfloat16, torch.float32)
    assert ds_config["bf16"]["bf16_master_weights_and_grads"] is not fp32_masters
    assert ds_config["bf16"]["bf16_optimizer_states"] is not fp32_masters


@require_torch_gpu
def test_worker_default_matches_explicit_fp32_loss_and_gradient_norm(worker):
    """The worker default, with or without a supplied ``data_types`` block, matches explicit FP32 exactly."""
    import deepspeed
    import torch.distributed as dist

    owns_group = not dist.is_initialized()
    if owns_group:
        port = reserve_free_port(get_unique_port_number() + 1, span=7)
        deepspeed.init_distributed(
            dist_backend="nccl",
            auto_mpi_discovery=False,
            init_method=f"tcp://127.0.0.1:{port}",
            rank=0,
            world_size=1,
        )
    measurements = {}
    try:
        for label in ("worker_default", "block_without_dtype", "explicit_fp32"):
            set_seed(17)
            model = torch.nn.Linear(16, 8, bias=False, device="cuda", dtype=torch.bfloat16)
            config = dict(
                train_micro_batch_size_per_gpu=1,
                gradient_accumulation_steps=4,
                bf16=dict(enabled=True),
                zero_optimization=dict(stage=2),
                gradient_clipping=0.0,
                steps_per_print=1000,
            )
            if label == "explicit_fp32":
                config["data_types"] = dict(grad_accum_dtype="fp32")
            elif label == "block_without_dtype":
                config["data_types"] = {}
            worker.ds_training_config({"training_config": {"optimizer": {"torch_adam": True, "lr": 0.01}}}, config, {})
            engine, *_ = deepspeed.initialize(model=model, model_parameters=model.parameters(), config=config)
            assert engine.get_data_types() == (torch.bfloat16, torch.float32)
            losses = []
            norms = []
            for _ in range(8):
                inputs = torch.randn(1, 16, device="cuda", dtype=torch.bfloat16)
                targets = torch.randn(1, 8, device="cuda", dtype=torch.bfloat16)
                loss = (engine(inputs).float() - targets.float()).square().mean()
                losses.append(loss.detach().cpu())
                engine.backward(loss)
                boundary = engine.is_gradient_accumulation_boundary()
                engine.step()
                if boundary:
                    norms.append(float(engine.get_global_grad_norm()))
            measurements[label] = dict(
                loss=torch.stack(losses),
                grad_norm=torch.tensor(norms),
                accumulation_dtype=str(engine.get_data_types()[1]),
            )
            print(
                f"ACCUMULATION_CONTROL {label}: dtype={measurements[label]['accumulation_dtype']} "
                f"loss={measurements[label]['loss'].tolist()} grad_norm={norms}",
                flush=True,
            )
            del engine, model
        for metric in ("loss", "grad_norm"):
            assert torch.isfinite(measurements["worker_default"][metric]).all()
            for label in ("worker_default", "block_without_dtype"):
                torch_assert_equal(measurements[label][metric], measurements["explicit_fp32"][metric])
    finally:
        if owns_group:
            dist.destroy_process_group()
