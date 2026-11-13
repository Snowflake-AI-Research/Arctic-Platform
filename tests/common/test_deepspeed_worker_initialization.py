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

"""The on-prem worker builds its DeepSpeed engine from typed training settings."""

import json
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.common.optimizer import OptimizerSettings
from arctic_platform.common.training_settings import MicroBatchSpec
from arctic_platform.common.training_settings import WorkerTrainingSettings


def test_onprem_config_builds_one_microbatch_per_boundary_and_keeps_overrides():
    from arctic_platform.common.deepspeed_worker import onprem_deepspeed_config

    training = WorkerTrainingSettings(
        optimizer=OptimizerSettings(lr=3e-4),
        gradient_clipping=0.5,
        mb_spec=MicroBatchSpec(num_micro_batches=3),
        ds_config={
            "zero_optimization": {"stage": 1, "offload_optimizer": {"device": "cpu"}},
            "scheduler": {"type": "WarmupLR", "params": {"warmup_num_steps": 10}},
        },
    )

    resolved = onprem_deepspeed_config(training, dtype="bfloat16", world_size=4, seed=7)

    assert resolved["train_micro_batch_size_per_gpu"] == 1
    assert resolved["gradient_accumulation_steps"] == 1
    assert resolved["train_batch_size"] == 4
    assert resolved["gradient_clipping"] == 0.5
    assert resolved["zero_optimization"] == training.ds_config["zero_optimization"]
    assert "optimizer" not in resolved
    assert resolved["scheduler"] == training.ds_config["scheduler"]
    assert resolved["seed"] == 7
    assert resolved["torch_autocast"] == {"enabled": True, "dtype": "bfloat16"}
    assert resolved["data_types"] == {"grad_accum_dtype": "fp32"}


def test_onprem_config_sizes_the_batch_for_a_ds_config_sequence_parallel_degree():
    from arctic_platform.common.deepspeed_worker import onprem_deepspeed_config

    training = WorkerTrainingSettings(ds_config={"sequence_parallel_size": 2})
    resolved = onprem_deepspeed_config(training, dtype="bfloat16", world_size=8)

    assert resolved["sequence_parallel_size"] == 2
    assert resolved["train_batch_size"] == 4


def test_expert_parallel_worker_builds_the_moe_spec_and_splits_expert_groups(monkeypatch, tmp_path):
    pytest.importorskip("ray")
    pytest.importorskip("deepspeed")
    from arctic_platform.common import deepspeed_worker as mod

    model_path = tmp_path / "glm5_next"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps({"model_type": "glm5_next"}))
    worker = object.__new__(mod.DeepSpeedWorker.__ray_metadata__.modified_class)
    worker.rank, worker.world_size, worker.master_port = 0, 8, 12345
    model = torch.nn.Identity()
    observed = {}
    monkeypatch.setattr(mod.deepspeed, "init_distributed", lambda: None)
    monkeypatch.setattr(mod, "pr0", lambda *args: None)

    def build(spec):
        # Spec validation imports kernels that query the real device, so fake it only afterwards.
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda gpu: SimpleNamespace(uuid="test"))
        observed["spec"] = spec
        return SimpleNamespace(model=model)

    def initialize(model, ds_config, **kwargs):
        observed["ds_config"] = ds_config
        observed["engine"] = kwargs
        return SimpleNamespace(local_rank=0, global_steps=0), None, None

    monkeypatch.setattr(mod, "build_model", build)
    monkeypatch.setattr(mod, "initialize_training_engine", initialize)
    monkeypatch.setattr(mod, "get_accelerator", lambda: SimpleNamespace(device_name=lambda rank: "cuda:0"))

    assert worker.initialize(
        "localhost",
        {
            "model_name": str(model_path),
            "job_type": "train",
            "max_seq_len": 16384,
            "training_config": {"model_provider": "arctic", "attn_implementation": "sparse_mla", "ep_size": 8},
            "ds_worker_config": {"sp_size": 8},
        },
    )

    assert observed["spec"].loader == "glm5_next"
    assert observed["spec"].parallelism.expert_parallel == observed["spec"].parallelism.sequence_parallel == 8
    assert observed["spec"].loader_options["seq_len"] == 16384
    assert observed["engine"]["ep_size"] == 8
    assert observed["engine"]["mpu"] is None
    assert observed["ds_config"]["train_batch_size"] == 8


@pytest.mark.parametrize(
    "training_config, message",
    [
        ({"optimizer": None}, "a frozen model runs on Cortex only"),
        ({"mb_spec": {"max_tokens_per_mb": 1024}}, "max_tokens_per_mb is supported on Cortex only"),
        ({"ds_config": {"gradient_accumulation_steps": 2}}, "set training.mb_spec.num_micro_batches"),
    ],
)
def test_onprem_training_rejects_what_it_cannot_run(monkeypatch, training_config, message):
    pytest.importorskip("ray")
    pytest.importorskip("deepspeed")
    from arctic_platform.common import deepspeed_worker as mod

    worker = object.__new__(mod.DeepSpeedWorker.__ray_metadata__.modified_class)
    worker.rank, worker.world_size, worker.master_port = 0, 1, 12345
    monkeypatch.setattr(mod.deepspeed, "init_distributed", lambda: None)
    monkeypatch.setattr(mod, "pr0", lambda *args: None)

    with pytest.raises(ValueError, match=message):
        worker.initialize("localhost", {"model_name": "m", "job_type": "training", "training_config": training_config})
