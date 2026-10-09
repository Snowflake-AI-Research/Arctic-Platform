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

"""The hosted training payload uses the same determinism set as the gateway payload."""

from pathlib import Path

from arctic_platform.correctness.harness.config import LoadedConfig


def test_hosted_training_sub_job_pins_determinism(monkeypatch) -> None:
    captured = {}

    class SubJobConfig:
        @staticmethod
        def training_job(*args, **kwargs):
            captured.update(kwargs)
            return kwargs

    import dss_client.neutrino_client as client

    monkeypatch.setattr(client, "SubJobConfig", SubJobConfig)
    from arctic_platform.correctness.harness.hosted import training_sub_job

    cfg = LoadedConfig(
        config_id="x",
        path=Path("x"),
        sub_job={
            "model_name": "model",
            "training_config": {
                "optimizer": {"type": "adamw"},
                "max_seq_len": 8,
                "train_batch_size": 1,
                "n_gpus": 1,
                "debug": {"full_determinism": False, "full_determinism_must_comply": True},
            },
            "global_batch_size": 1,
            "dtype": "bfloat16",
            "seed": 1,
        },
    )

    training_sub_job(cfg)

    debug = captured["extra_training"]["debug"]
    assert debug["full_determinism"] is True
    assert debug["full_determinism_must_comply"] is False
