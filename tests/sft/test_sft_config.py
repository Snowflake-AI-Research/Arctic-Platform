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
"""``ArcticSFTClientConfig`` validation — pure CPU, no GPU / Ray / model load."""

from __future__ import annotations

from pydantic import ValidationError

from arctic_platform.sft import ArcticSFTClientConfig
from arctic_platform.testing_utils import TestCasePlus


class TestArcticSFTClientConfig(TestCasePlus):
    def test_requires_training_gpus(self):
        with self.assertRaises(ValidationError):
            ArcticSFTClientConfig(model_name="m", training_gpus=0)

    def test_reconnect_bypasses_gpu_guard(self):
        cfg = ArcticSFTClientConfig(model_name="m", training_gpus=0, training_job_id=7)
        self.assertEqual(cfg.training_job_id, 7)

    def test_defaults(self):
        cfg = ArcticSFTClientConfig(model_name="m", training_gpus=2, checkpoint_path="/tmp/c")
        self.assertEqual(cfg.comm_protocol, "http")
        self.assertEqual(cfg.backend, "onprem")
        self.assertFalse(cfg.launch_local_server)
        self.assertIsNone(cfg.server_cuda_visible_devices)

    def test_new_job_requires_checkpoint_path(self):
        with self.assertRaises(ValidationError):
            ArcticSFTClientConfig(model_name="m", training_gpus=1)

    def test_invalid_comm_protocol_rejected(self):
        with self.assertRaises(ValidationError):
            ArcticSFTClientConfig(
                model_name="m",
                training_gpus=1,
                checkpoint_path="/tmp/c",
                comm_protocol="carrier-pigeon",
            )

    def test_extra_fields_forbidden(self):
        with self.assertRaises(ValidationError):
            ArcticSFTClientConfig(model_name="m", training_gpus=1, checkpoint_path="/tmp/c", bogus=1)

    def test_sampling_fields_forwarded(self):
        cfg = ArcticSFTClientConfig(
            model_name="m",
            training_gpus=1,
            checkpoint_path="/tmp/c",
            sampling_gpus=1,
            colocate=True,
            vllm_config={"tensor_parallel_size": 1},
        )
        rl = cfg.to_rl_config()
        self.assertEqual(rl.sampling_gpus, 1)
        self.assertTrue(rl.backend_config.colocate)
        self.assertEqual(rl.sampling.vllm["tensor_parallel_size"], 1)
