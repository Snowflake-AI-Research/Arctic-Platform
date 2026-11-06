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


@pytest.mark.parametrize("job_config", [{}, {"training_config": {}}])
def test_missing_data_types_retains_worker_fp32_accumulation_default(worker, job_config):
    config = deepcopy(_ZERO2)
    assert worker.ds_training_config(job_config, config, {}) is config
    assert config["data_types"] == {"grad_accum_dtype": "fp32"}


@pytest.mark.parametrize(
    "data_types", [{"grad_accum_dtype": "bf16"}, {"grad_accum_dtype": "fp32", "other_setting": True}]
)
@pytest.mark.parametrize("zero", [{"stage": 2}, *_REFUSED_ZERO])
def test_explicit_accumulation_precision_is_preserved(worker, data_types, zero):
    supplied = deepcopy(data_types)
    config = {"data_types": supplied, "zero_optimization": deepcopy(zero)}
    worker.ds_training_config({}, config, {})
    assert config["data_types"] == data_types
    assert supplied == data_types


@pytest.mark.parametrize("data_types", [None, {}, {"other_setting": True}, {"grad_accum_dtype": None}])
def test_supplied_block_without_accumulation_precision_gets_worker_fp32_default(worker, data_types):
    supplied = deepcopy(data_types)
    config = {"data_types": supplied, **deepcopy(_ZERO2)}
    worker.ds_training_config({}, config, {})
    assert config["data_types"] == {**(data_types or {}), "grad_accum_dtype": "fp32"}
    assert supplied == data_types


@pytest.mark.parametrize("zero", _REFUSED_ZERO)
def test_worker_leaves_accumulation_unset_where_deepspeed_refuses_fp32(worker, zero):
    config = {"zero_optimization": deepcopy(zero)}
    worker.ds_training_config({}, config, {})
    assert config["bf16"] == {"enabled": True}
    assert "data_types" not in config


@pytest.mark.parametrize("dtype", ["bf16", "bfloat16"])
@pytest.mark.parametrize("fp32_masters", [False, True])
def test_bfloat16_config_builder_defaults_accumulation_to_fp32(dtype, fp32_masters):
    ds_config = build_deepspeed_config(TrainingEngineSpec(world_size=4, dtype=dtype, fp32_masters=fp32_masters))
    assert ds_config["data_types"] == {"grad_accum_dtype": "fp32"}


@pytest.mark.parametrize("dtype", ["float16", "fp16", "float32", "fp32"])
def test_other_precisions_default_accumulation_to_fp32_under_zero(dtype):
    ds_config = build_deepspeed_config(TrainingEngineSpec(world_size=4, dtype=dtype))
    assert ds_config["data_types"] == {"grad_accum_dtype": "fp32"}


@pytest.mark.parametrize("data_types", [{"grad_accum_dtype": "bf16"}, {"grad_accum_dtype": "fp32"}])
@pytest.mark.parametrize("zero", [{"stage": 2}, *_REFUSED_ZERO])
def test_explicit_accumulation_override_is_preserved(data_types, zero):
    override = {"data_types": deepcopy(data_types), "zero_optimization": deepcopy(zero)}
    ds_config = build_deepspeed_config(TrainingEngineSpec(world_size=4, ds_overrides=override))
    assert ds_config["data_types"] == data_types
    assert override == {"data_types": data_types, "zero_optimization": zero}


@pytest.mark.parametrize("data_types", [None, {}, {"other_setting": True}, {"grad_accum_dtype": None}])
def test_override_block_without_accumulation_precision_gets_fp32_default(data_types):
    override = {"data_types": deepcopy(data_types)}
    ds_config = build_deepspeed_config(TrainingEngineSpec(world_size=4, ds_overrides=override))
    assert ds_config["data_types"] == {**(data_types or {}), "grad_accum_dtype": "fp32"}
    assert override == {"data_types": data_types}


@pytest.mark.parametrize(
    "dtype, zero",
    [("bfloat16", _REFUSED_ZERO[0]), ("float16", _REFUSED_ZERO[0]), ("bfloat16", _REFUSED_ZERO[1])],
)
def test_config_builder_leaves_accumulation_unset_where_deepspeed_refuses_fp32(dtype, zero):
    ds_config = build_deepspeed_config(
        TrainingEngineSpec(world_size=4, dtype=dtype, ds_overrides={"zero_optimization": deepcopy(zero)})
    )
    assert "data_types" not in ds_config


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
def test_default_is_written_exactly_where_deepspeed_accepts_fp32_accumulation(zero, dtype):
    """DeepSpeed's own optimizer sanity check decides which configs accept explicit FP32 accumulation."""
    from deepspeed.runtime.config import DeepSpeedConfig
    from deepspeed.runtime.engine import DeepSpeedEngine

    overrides = {"zero_optimization": deepcopy(zero), "zero_allow_untested_optimizer": True}
    defaulted = build_deepspeed_config(
        TrainingEngineSpec(world_size=1, dtype=dtype, fp32_masters=True, ds_overrides=overrides)
    )
    explicit = {**defaulted, "data_types": {"grad_accum_dtype": "fp32"}}
    engine = DeepSpeedEngine.__new__(DeepSpeedEngine)
    torch.nn.Module.__init__(engine)
    object.__setattr__(engine, "_config", DeepSpeedConfig(explicit))
    object.__setattr__(engine, "pipeline_parallelism", False)
    object.__setattr__(engine, "global_rank", 0)
    try:
        engine._do_optimizer_sanity_check(torch.optim.AdamW([torch.nn.Parameter(torch.zeros(8))]))
        accepted = True
    except (NotImplementedError, ValueError):
        accepted = False
    assert ("data_types" in defaulted) is accepted
    if accepted:
        assert defaulted["data_types"] == {"grad_accum_dtype": "fp32"}


@pytest.mark.parametrize("override", [None, {"data_types": {"grad_accum_dtype": "fp32"}}])
@pytest.mark.parametrize("fp32_masters", [False, True])
def test_deepspeed_resolves_float32_accumulation_without_changing_model_precision(fp32_masters, override):
    from deepspeed.runtime.config import DeepSpeedConfig
    from deepspeed.runtime.engine import DeepSpeedEngine

    ds_config = build_deepspeed_config(
        TrainingEngineSpec(
            world_size=1,
            fp32_masters=fp32_masters,
            ds_overrides=override,
        )
    )
    config = DeepSpeedConfig(ds_config)
    engine = DeepSpeedEngine.__new__(DeepSpeedEngine)
    torch.nn.Module.__init__(engine)
    object.__setattr__(engine, "_config", config)
    engine.optimizer = None
    engine.checkpoint_engine = None
    assert engine.get_data_types() == (torch.bfloat16, torch.float32)
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
