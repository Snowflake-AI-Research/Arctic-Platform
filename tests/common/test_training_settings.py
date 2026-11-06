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

"""Typed training-engine settings and the clients that send them."""

from types import SimpleNamespace

import pytest

from arctic_platform.client import ArcticClientConfig
from arctic_platform.client import CortexConfig
from arctic_platform.client import MicroBatchSpec
from arctic_platform.client import OptimizerSettings
from arctic_platform.client import TrainingConfig
from arctic_platform.common.deepspeed_engine import TrainingEngineSpec
from arctic_platform.common.deepspeed_engine import training_engine_spec
from arctic_platform.common.training_settings import TrainingEngineSettings
from arctic_platform.common.training_settings import WorkerTrainingSettings
from arctic_platform.common.utils.server_models import JobConfig
from arctic_platform.rl.config import ArcticRLClientConfig


@pytest.mark.parametrize(
    "key, hint",
    [
        ("optimizer", "set training.optimizer"),
        ("gradient_clipping", "set training.gradient_clipping"),
        ("gradient_accumulation_steps", "set training.mb_spec.num_micro_batches"),
        ("train_micro_batch_size_per_gpu", "set training.mb_spec.num_micro_batches"),
        ("train_batch_size", "the engine derives it from the data-parallel size"),
    ],
)
def test_ds_config_may_not_set_what_the_engine_owns(key, hint):
    with pytest.raises(ValueError, match=f"may not set {key} \\({hint}"):
        WorkerTrainingSettings(ds_config={key: 1})
    with pytest.raises(ValueError, match=f"may not set {key}"):
        TrainingConfig(ds_config={key: 1})


def test_shared_settings_accept_an_older_clients_ds_config():
    settings = TrainingEngineSettings(ds_config={"optimizer": {"type": "AdamW"}, "gradient_accumulation_steps": 2})
    assert settings.ds_config["gradient_accumulation_steps"] == 2


def test_micro_batch_spec_takes_one_policy():
    with pytest.raises(ValueError, match="Use one"):
        MicroBatchSpec(num_micro_batches=2, max_tokens_per_mb=1024)
    with pytest.raises(ValueError):
        MicroBatchSpec(num_micro_batches=0)


def test_training_engine_spec_is_the_one_translation():
    settings = TrainingEngineSettings(
        optimizer=OptimizerSettings(lr=3e-4),
        gradient_clipping=0.5,
        fp32_masters=True,
        zero_bucket_size=1024,
        ds_config={"zero_optimization": {"stage": 1}},
    )

    assert training_engine_spec(settings, dtype="bfloat16", world_size=8, sequence_parallel=2, seed=3) == (
        TrainingEngineSpec(
            dtype="bfloat16",
            world_size=8,
            sequence_parallel=2,
            gradient_clipping=0.5,
            seed=3,
            ds_overrides={"zero_optimization": {"stage": 1}},
            fp32_masters=True,
            zero_bucket_size=1024,
            optimizer=OptimizerSettings(lr=3e-4),
        )
    )
    frozen = training_engine_spec(TrainingEngineSettings(optimizer=None), dtype="bfloat16", world_size=1)
    assert frozen.optimizer is None
    assert frozen.ds_overrides is None


def test_onprem_payload_sends_typed_training_settings():
    cfg = ArcticClientConfig(
        model_name="m",
        training_gpus=2,
        training=TrainingConfig(
            optimizer=OptimizerSettings(lr=2e-5),
            gradient_clipping=0.0,
            mb_spec=MicroBatchSpec(num_micro_batches=4),
            ds_config={"zero_optimization": {"stage": 3}, "sequence_parallel_size": 2},
        ),
    )

    payload = cfg.to_onprem("training")

    assert "ds_config" not in payload
    training = WorkerTrainingSettings.model_validate(payload["training_config"])
    assert training.optimizer.lr == 2e-5
    assert training.gradient_clipping == 0.0
    assert training.mb_spec.num_micro_batches == 4
    assert JobConfig.model_validate(payload).sp_size == 2


def test_onprem_log_prob_inherits_the_training_sp_degree_unless_it_sets_one():
    def payload(log_prob_ds_config):
        cfg = ArcticClientConfig(
            model_name="m",
            training_gpus=2,
            log_prob_gpus=2,
            training=TrainingConfig(ds_config={"sequence_parallel_size": 2}),
            sampling={"log_prob_engine": "deepspeed", "log_prob_ds_config": log_prob_ds_config},
        )
        return cfg.to_onprem("log_prob")

    assert payload(None)["log_prob_config"] == {"sequence_parallel_size": 2}
    assert payload({"sequence_parallel_size": 1})["log_prob_config"] == {"sequence_parallel_size": 1}
    assert "ds_config" not in payload(None)


def test_worker_config_mb_spec_moved_to_training():
    with pytest.raises(ValueError, match="moved to training.mb_spec"):
        TrainingConfig(ds_worker_config={"mb_spec": {"max_tokens_per_mb": 1024}})


def _cortex(training: TrainingConfig) -> dict:
    cfg = ArcticClientConfig(
        model_name="m", training_gpus=1, backend=CortexConfig(base_url="https://x", pat="y"), training=training
    )
    return cfg._cortex_training_sub_job()["training_config"]


def test_cortex_payload_sends_typed_training_settings():
    training = _cortex(
        TrainingConfig(
            optimizer=OptimizerSettings(lr=1e-6, betas=[0.9, 0.95]),
            mb_spec=MicroBatchSpec(max_tokens_per_mb=4096),
            ds_config={"zero_optimization": {"stage": 2}},
        )
    )

    assert training["optimizer"] == {
        "name": "AdamW",
        "lr": 1e-6,
        "weight_decay": 0.0,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
    }
    assert training["gradient_clipping"] == 1.0
    assert training["mb_spec"] == {"max_tokens_per_mb": 4096}
    assert training["ds_config"] == {"zero_optimization": {"stage": 2}}
    assert "train_batch_size" not in training
    assert _cortex(TrainingConfig(optimizer=None))["optimizer"] is None


def test_cortex_rejects_a_microbatch_count():
    with pytest.raises(ValueError, match="num_micro_batches is supported on-prem only"):
        _cortex(TrainingConfig(mb_spec=MicroBatchSpec(num_micro_batches=2)))


def _skyrl_legacy(**overrides) -> ArcticRLClientConfig:
    """The shape SkyRL's arctic_rl integration builds."""
    fields = dict(
        model_name="Qwen/Qwen3-0.6B",
        training_gpus=4,
        log_prob_gpus=2,
        log_prob_engine="deepspeed",
        ds_config={
            "train_micro_batch_size_per_gpu": 1,
            "train_batch_size": 64,
            "gradient_accumulation_steps": 16,
            "sequence_parallel_size": 2,
            "bf16": {"enabled": True},
            "gradient_clipping": 1.0,
            "optimizer": {
                "type": "AdamW",
                "params": {"lr": 1e-6, "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
            },
            "zero_optimization": {"stage": 2, "offload_optimizer": {"device": "none"}},
        },
        ds_worker_config={"attn_implementation": "sdpa", "mb_spec": {"max_tokens_per_mb": 4096}},
        training_config={
            "training_horizon": 100,
            "optimizer": {"lr": 2e-6, "weight_decay": 0.01, "betas": [0.9, 0.95], "gradient_clipping": 0.5},
            "lr_scheduler": {"type": "cosine", "warmup_ratio": 0.1},
            "max_length": 1536,
            "model_config": None,
        },
    )
    fields.update(overrides)
    return ArcticRLClientConfig(**fields)


def test_legacy_config_translates_to_typed_settings():
    settings = _skyrl_legacy().training_settings()

    assert settings.optimizer == OptimizerSettings(lr=2e-6, weight_decay=0.01, betas=[0.9, 0.95])
    assert settings.gradient_clipping == 0.5
    assert settings.mb_spec == MicroBatchSpec(num_micro_batches=16)
    assert settings.ds_config == {
        "sequence_parallel_size": 2,
        "bf16": {"enabled": True},
        "zero_optimization": {"stage": 2, "offload_optimizer": {"device": "none"}},
        "scheduler": {
            "type": "WarmupCosineLR",
            "params": {
                "total_num_steps": 100,
                "warmup_num_steps": 10,
                "warmup_min_ratio": 0.0,
                "cos_min_ratio": 0.0,
                "warmup_type": "linear",
            },
        },
    }


def test_legacy_ds_config_alone_translates_and_unset_clipping_stays_off():
    settings = ArcticRLClientConfig(
        model_name="m",
        training_gpus=1,
        ds_config={"gradient_accumulation_steps": 1, "optimizer": {"type": "AdamW", "params": {"lr": 3e-5}}},
    ).training_settings()

    assert settings.optimizer == OptimizerSettings(lr=3e-5)
    assert settings.gradient_clipping == 0.0
    assert settings.ds_config is None

    warmup = _skyrl_legacy(
        training_config={"training_horizon": 10, "optimizer": {"lr": 1e-5}, "lr_scheduler": {"warmup_steps": 2.4}}
    ).training_settings()
    assert warmup.ds_config["scheduler"]["type"] == "WarmupLR"
    assert warmup.ds_config["scheduler"]["params"]["warmup_num_steps"] == 2
    assert warmup.gradient_clipping == 1.0


def test_legacy_torch_adam_is_rejected():
    with pytest.raises(ValueError, match="torch_adam is no longer supported"):
        _skyrl_legacy(training_config={"optimizer": {"torch_adam": True}}).training_settings()


def test_legacy_onprem_client_sends_typed_settings(monkeypatch):
    from arctic_platform.rl import ray_client

    sent = []
    client = object.__new__(ray_client.ArcticRLRayClient)
    client.config = _skyrl_legacy()
    client._arctic_rl_ray_server_state = SimpleNamespace(
        initialize=SimpleNamespace(remote=lambda job_config: sent.append(job_config) or job_config)
    )
    monkeypatch.setattr(ray_client.ray, "get", lambda value: value)

    training = client._post_initialize("training")
    log_prob = client._post_initialize("log_prob")

    assert "ds_config" not in training and "ds_config" not in log_prob
    assert WorkerTrainingSettings.model_validate(training["training_config"]) == client.config.training_settings()
    assert log_prob["log_prob_config"] == {"sequence_parallel_size": 2}
    assert JobConfig.model_validate(training).sp_size == 2
    assert JobConfig.model_validate(log_prob).sp_size == 2


def test_legacy_cortex_dispatch_sends_typed_settings_and_the_token_budget(monkeypatch):
    from arctic_platform.integrations._cortex_dispatch import _to_unified_config

    monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://example.invalid")
    unified = _to_unified_config(_skyrl_legacy())

    assert unified.training.mb_spec == MicroBatchSpec(max_tokens_per_mb=4096)
    assert unified.training.ds_worker_config == {"attn_implementation": "sdpa"}
    training = unified._cortex_training_sub_job()["training_config"]
    assert training["optimizer"]["lr"] == 2e-6
    assert training["gradient_clipping"] == 0.5
    assert training["mb_spec"] == {"max_tokens_per_mb": 4096}
    assert training["ds_config"]["zero_optimization"] == {"stage": 2}
    assert training["ds_config"]["scheduler"]["type"] == "WarmupCosineLR"
