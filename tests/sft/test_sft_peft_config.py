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
"""Typed PEFT settings reach the on-prem trainer without changing the reference model."""

import copy

from arctic_platform.client import ArcticClientConfig
from arctic_platform.client import CortexConfig
from arctic_platform.client import SamplingConfig
from arctic_platform.client import TrainingConfig
from arctic_platform.client.base import _check_weight_format
from arctic_platform.model import ModelSpec
from arctic_platform.testing_utils import TestCasePlus


class TestPeftConfig(TestCasePlus):
    def test_onprem_training_payload_reaches_model_patch(self):
        peft = {"peft_type": "LORA", "r": 8, "target_modules": ["q_proj"]}
        worker = {"attn_implementation": "eager"}
        config = ArcticClientConfig(
            model_name="unused", training_gpus=1, training=TrainingConfig(peft=peft, ds_worker_config=worker)
        )
        expected = copy.deepcopy(config)
        payload = config.to_onprem("training")
        self.assertEqual(payload["ds_worker_config"]["peft_config"], peft)
        self.assertEqual(config, expected)
        spec = ModelSpec.from_ds_worker_config(payload["model_name"], payload["ds_worker_config"])
        self.assertEqual(spec.patches.peft, peft)
        self.assertEqual(ArcticClientConfig.model_validate_json(config.model_dump_json()), config)

    def test_reference_job_stays_unadapted(self):
        worker = {"attn_implementation": "eager", "peft_config": {"peft_type": "LORA"}}
        config = ArcticClientConfig(
            model_name="unused",
            training_gpus=1,
            log_prob_gpus=1,
            training=TrainingConfig(ds_worker_config=worker),
            sampling=SamplingConfig(log_prob_engine="deepspeed"),
        )
        self.assertEqual(config.training.peft, worker["peft_config"])
        self.assertNotIn("peft_config", config.to_onprem("log_prob")["ds_worker_config"])
        self.assertNotIn("peft_config", config.to_onprem("sampling"))
        self.assertIn("peft_config", worker)

    def test_empty_configs_are_rejected(self):
        for kwargs in ({"peft": {}}, {"ds_worker_config": {"peft_config": {}}}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "peft_type"):
                TrainingConfig(**kwargs)

    def test_conflicting_configs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            TrainingConfig(
                peft={"peft_type": "LORA", "r": 4}, ds_worker_config={"peft_config": {"peft_type": "LORA", "r": 8}}
            )

    def test_onprem_sampler_sync_is_rejected_before_launch(self):
        for jobs in ({"sampling_gpus": 1}, {"sampling_job_id": 1}):
            for settings in (
                {"peft": {"peft_type": "LORA"}},
                {"ds_worker_config": {"peft_config": {"peft_type": "LORA"}}},
            ):
                with self.subTest(jobs=jobs, settings=settings), self.assertRaisesRegex(ValueError, "adapter sync"):
                    ArcticClientConfig(model_name="unused", training=TrainingConfig(**settings), **jobs)

    def test_cortex_keeps_training_and_sampling_peft(self):
        peft = {"peft_type": "LORA", "target_modules": ["q_proj"]}
        config = ArcticClientConfig(
            model_name="unused",
            training_gpus=1,
            sampling_gpus=1,
            training=TrainingConfig(peft=peft),
            backend=CortexConfig(base_url="http://localhost"),
        )
        subs = {sub["job_type"]: sub for sub in config.to_cortex()}
        self.assertEqual(subs["training"]["training_config"]["peft_config"], peft)
        self.assertEqual(subs["sampling"]["inference_config"]["peft_config"], peft)

    def test_onprem_sync_cannot_fall_back_to_full_weights(self):
        config = ArcticClientConfig(model_name="unused", training=TrainingConfig(peft={"peft_type": "LORA"}))
        for weight_format in (None, "lora"):
            with self.subTest(weight_format=weight_format), self.assertRaisesRegex(ValueError, "adapter sync"):
                _check_weight_format(config, weight_format)
