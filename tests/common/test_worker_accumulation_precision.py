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


@pytest.mark.parametrize("job_config", [{}, {"training_config": {}}])
def test_missing_data_types_retains_worker_fp32_accumulation_default(worker, job_config):
    config = {}
    assert worker.ds_training_config(job_config, config, {}) is config
    assert config["data_types"] == {"grad_accum_dtype": "fp32"}


@pytest.mark.parametrize(
    "data_types", [{}, {"grad_accum_dtype": "bf16"}, {"grad_accum_dtype": "fp32"}, {"other_setting": True}]
)
def test_explicit_accumulation_precision_is_preserved(worker, data_types):
    supplied = deepcopy(data_types)
    config = {"data_types": supplied}
    worker.ds_training_config({}, config, {})
    assert config["data_types"] == data_types
    assert config["data_types"] is supplied


@pytest.mark.parametrize("dtype", ["bf16", "bfloat16"])
@pytest.mark.parametrize("fp32_masters", [False, True])
def test_bfloat16_config_builder_does_not_insert_worker_accumulation_default(dtype, fp32_masters):
    ds_config = build_deepspeed_config(TrainingEngineSpec(world_size=4, dtype=dtype, fp32_masters=fp32_masters))
    assert "data_types" not in ds_config


@pytest.mark.parametrize("dtype", ["float16", "fp16", "float32", "fp32"])
def test_other_precisions_leave_accumulation_dtype_to_deepspeed(dtype):
    assert "data_types" not in build_deepspeed_config(TrainingEngineSpec(world_size=4, dtype=dtype))


@pytest.mark.parametrize("data_types", [{}, {"grad_accum_dtype": "bf16"}, {"grad_accum_dtype": "fp32"}])
def test_explicit_accumulation_override_is_preserved(data_types):
    override = {"data_types": deepcopy(data_types)}
    ds_config = build_deepspeed_config(TrainingEngineSpec(world_size=4, ds_overrides=override))
    assert ds_config["data_types"] == data_types
    assert override == {"data_types": data_types}


@pytest.mark.parametrize("fp32_masters", [False, True])
def test_deepspeed_resolves_explicit_float32_accumulation_without_changing_model_precision(fp32_masters):
    from deepspeed.runtime.config import DeepSpeedConfig
    from deepspeed.runtime.engine import DeepSpeedEngine

    ds_config = build_deepspeed_config(
        TrainingEngineSpec(
            world_size=1,
            fp32_masters=fp32_masters,
            ds_overrides={"data_types": {"grad_accum_dtype": "fp32"}},
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
    """The retained worker default and explicit FP32 must produce identical training metrics."""
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
        for label in ("worker_default", "explicit_fp32"):
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
            torch_assert_equal(measurements["worker_default"][metric], measurements["explicit_fp32"][metric])
    finally:
        if owns_group:
            dist.destroy_process_group()
