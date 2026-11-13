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

"""The on-prem worker hands DeepSpeed a TrainingEngineSpec."""

import json
from types import SimpleNamespace

import pytest
import torch

from arctic_platform.common.deepspeed_engine import TrainingEngineSpec
from arctic_platform.common.deepspeed_engine import build_deepspeed_config


def test_onprem_engine_adapter_translates_optimizer_and_preserves_scheduler_and_gas():
    ds_config = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 3,
        "gradient_clipping": 0.5,
        "bf16": {"enabled": True},
        "communication_data_type": "fp32",
        "zero_optimization": {"stage": 1, "offload_optimizer": {"device": "cpu"}},
        "optimizer": {"type": "AdamW", "params": {"lr": 3e-4, "torch_adam": True}},
        "scheduler": {"type": "WarmupLR", "params": {"warmup_num_steps": 10}},
    }

    spec = TrainingEngineSpec.from_deepspeed_config(ds_config, dtype="bfloat16", world_size=4, seed=7)
    resolved = build_deepspeed_config(spec)

    assert spec.optimizer.name == "torch_adamw"
    assert spec.optimizer.lr == 3e-4
    assert spec.optimizer.fused is False
    assert resolved["train_batch_size"] == 24
    assert resolved["zero_optimization"] == ds_config["zero_optimization"]
    assert "optimizer" not in resolved
    assert resolved["scheduler"] == ds_config["scheduler"]
    assert resolved["seed"] == 7


def test_onprem_engine_adapter_keeps_the_worker_accumulation_dtype():
    ds_config = {
        "bf16": {"enabled": True},
        "communication_data_type": "fp32",
        "data_types": {"grad_accum_dtype": "fp32"},
        "optimizer": {"type": "AdamW", "params": {"lr": 1e-5}},
    }

    resolved = build_deepspeed_config(
        TrainingEngineSpec.from_deepspeed_config(ds_config, dtype="bfloat16", world_size=2)
    )

    assert resolved["data_types"] == {"grad_accum_dtype": "fp32"}


def test_expert_parallel_worker_builds_the_moe_spec_and_splits_expert_groups(monkeypatch, tmp_path):
    pytest.importorskip("ray")
    pytest.importorskip("deepspeed")
    from arctic_platform.common import deepspeed_worker as mod
    from arctic_platform.model.parallel import groups

    ep, sp, mpu = object(), object(), object()
    monkeypatch.setattr(
        mod.dist, "all_gather_object", lambda output, host: output.__setitem__(slice(None), [host] * 8)
    )
    monkeypatch.setattr(groups, "expert_parallel_group", lambda size: ep)

    def setup(sp_size, ep_size, **kwargs):
        assert ep_size == 8
        assert kwargs["world_size"] == 8
        return groups.SPGroups(mpu if sp_size > 1 else None, sp if sp_size > 1 else None, sp_size, 0)

    monkeypatch.setattr(groups, "setup_parallel_groups", setup)

    model_path = tmp_path / "glm5_next"
    model_path.mkdir()
    (model_path / "config.json").write_text(json.dumps({"model_type": "glm5_next"}))
    worker = object.__new__(mod.DeepSpeedWorker.__ray_metadata__.modified_class)
    worker.rank, worker.world_size, worker.master_port = 0, 8, 12345
    model = torch.nn.Identity()
    observed = {}
    monkeypatch.setattr(mod.deepspeed, "init_distributed", lambda: None)
    monkeypatch.setattr(mod, "pr0", lambda *args: None)
    monkeypatch.setattr(worker, "ds_training_config", lambda *args: {"train_micro_batch_size_per_gpu": 1})

    def build(spec, *, parallel_groups):
        # Spec validation imports kernels that query the real device, so fake it only afterwards.
        monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda gpu: SimpleNamespace(uuid="test"))
        observed["spec"] = spec
        assert parallel_groups == {"ep_group": ep, "sp_group": sp}
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
            "ds_worker_config": {
                "model_provider": "prime_rl",
                "attn_implementation": "sparse_mla",
                "ep_size": 8,
                "sp_size": 8,
            },
        },
    )

    assert observed["spec"].loader == "glm5_next"
    assert observed["spec"].parallelism.expert_parallel == observed["spec"].parallelism.sequence_parallel == 8
    assert observed["engine"]["ep_size"] == 8
    assert observed["engine"]["mpu"] is mpu
    assert observed["ds_config"]["train_batch_size"] == 1
