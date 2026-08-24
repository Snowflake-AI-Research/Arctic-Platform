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
"""``ArcticSFTClientConfig`` validation — pure CPU, no GPU / Ray / model load.

`ArcticSFTClientConfig` adds no fields to `ArcticClientConfig`; it only enforces
that an SFT run actually trains. These tests pin that the guards fire for SFT and
that the shared config stays permissive, since RL relies on both exemptions
(sampling-only clients run with `training_gpus=0`).
"""

from __future__ import annotations

from pydantic import ValidationError

from arctic_platform.client import ArcticClientConfig
from arctic_platform.client import TrainingConfig
from arctic_platform.common.utils.server_models import JobConfig
from arctic_platform.sft import ArcticSFTClientConfig
from arctic_platform.testing_utils import TestCasePlus

_CKPT = TrainingConfig(checkpoint_path="/tmp/c")


class TestArcticSFTClientConfig(TestCasePlus):
    def test_requires_training_gpus(self):
        with self.assertRaises(ValidationError):
            ArcticSFTClientConfig(model_name="m", training_gpus=0, training=_CKPT)

    def test_reconnect_bypasses_gpu_guard(self):
        cfg = ArcticSFTClientConfig(model_name="m", training_gpus=0, training_job_id=7)
        self.assertEqual(cfg.training_job_id, 7)

    def test_new_job_requires_checkpoint_path(self):
        with self.assertRaises(ValidationError):
            ArcticSFTClientConfig(model_name="m", training_gpus=1)

    def test_reconnect_bypasses_checkpoint_path_guard(self):
        cfg = ArcticSFTClientConfig(model_name="m", training_gpus=1, training_job_id=7)
        self.assertIsNone(cfg.training.checkpoint_path)

    def test_accepts_a_complete_config(self):
        cfg = ArcticSFTClientConfig(model_name="m", training_gpus=2, training=_CKPT)
        self.assertEqual(cfg.training.checkpoint_path, "/tmp/c")
        self.assertEqual(cfg.backend.protocol, "http")
        self.assertEqual(cfg.backend.type, "onprem")

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            ArcticSFTClientConfig(model_name="m", training_gpus=1, training=_CKPT, bogus=1)

    def test_shared_config_does_not_enforce_the_sft_guards(self):
        """RL needs both exemptions, so the guards must live on the SFT subclass only."""
        cfg = ArcticClientConfig(model_name="m", training_gpus=0, sampling_gpus=2)
        self.assertEqual(cfg.sampling_gpus, 2)
        self.assertIsNone(ArcticClientConfig(model_name="m", training_gpus=1).training.checkpoint_path)

    def test_job_config_null_seed_uses_default(self):
        """JSON null is accepted on /initialize; missing seed uses the server default."""
        job = JobConfig.model_validate({"model_name": "m", "seed": None})
        self.assertEqual(job.seed, 42)
        omitted = JobConfig.model_validate({"model_name": "m"})
        self.assertEqual(omitted.seed, 42)

    def test_job_config_explicit_seed_preserved(self):
        job = JobConfig.model_validate({"model_name": "m", "seed": 7})
        self.assertEqual(job.seed, 7)
        job0 = JobConfig.model_validate({"model_name": "m", "seed": 0})
        self.assertEqual(job0.seed, 0)

    def test_default_seed_is_omitted_from_initialize_payload(self):
        """Unset client seed is omitted on /initialize, same as Cortex; JobConfig still accepts the payload."""
        cfg = ArcticSFTClientConfig(model_name="m", training_gpus=2, training=_CKPT)
        self.assertIsNone(cfg.seed)
        payload = cfg.to_onprem("training")
        self.assertNotIn("seed", payload)
        job = JobConfig.model_validate(payload)
        self.assertEqual(job.seed, 42)

    def test_explicit_seed_is_sent_on_initialize_payload(self):
        cfg = ArcticSFTClientConfig(model_name="m", training_gpus=2, training=_CKPT, seed=0)
        payload = cfg.to_onprem("training")
        self.assertEqual(payload["seed"], 0)
        self.assertEqual(JobConfig.model_validate(payload).seed, 0)
