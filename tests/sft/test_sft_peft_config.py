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
from arctic_platform.common.training_settings import WorkerTrainingSettings
from arctic_platform.model import settings_dense_model_spec
from arctic_platform.testing_utils import TestCasePlus


class TestPeftConfig(TestCasePlus):
    def test_onprem_training_payload_reaches_model_patch(self):
        peft = {"peft_type": "LORA", "r": 8, "target_modules": ["q_proj"]}
        config = ArcticClientConfig(
            model_name="unused", training_gpus=1, training=TrainingConfig(peft=peft, attn_implementation="eager")
        )
        expected = copy.deepcopy(config)
        payload = config.to_onprem("training")
        self.assertEqual(payload["training_config"]["peft_config"], peft)
        self.assertNotIn("ds_worker_config", payload)
        self.assertEqual(config, expected)
        settings = WorkerTrainingSettings.model_validate(payload["training_config"])
        spec = settings_dense_model_spec(
            settings, model_path=payload["model_name"], dtype="bfloat16", sequence_parallel=1
        )
        self.assertEqual(spec.patches.peft, peft)
        self.assertEqual(spec.attn_implementation, "eager")
        self.assertEqual(ArcticClientConfig.model_validate_json(config.model_dump_json()), config)

    def test_reference_job_stays_unadapted(self):
        config = ArcticClientConfig(
            model_name="unused",
            training_gpus=1,
            log_prob_gpus=1,
            training=TrainingConfig(attn_implementation="eager", peft={"peft_type": "LORA"}),
            sampling=SamplingConfig(log_prob_engine="deepspeed"),
        )
        self.assertEqual(config.to_onprem("log_prob")["training_config"], {"attn_implementation": "eager"})
        self.assertNotIn("peft_config", config.to_onprem("sampling"))

    def test_empty_configs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "peft_type"):
            TrainingConfig(peft={})

    def test_peft_has_one_spelling(self):
        for kwargs, message in (
            ({"peft_config": {"peft_type": "LORA"}}, "set training.peft, not training.peft_config"),
            ({"ds_worker_config": {"peft_config": {"peft_type": "LORA"}}}, "training.peft instead of peft_config"),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, message):
                TrainingConfig(**kwargs)

    def test_onprem_sampler_sync_is_rejected_before_launch(self):
        for jobs in ({"sampling_gpus": 1}, {"sampling_job_id": 1}):
            with self.subTest(jobs=jobs), self.assertRaisesRegex(ValueError, "adapter sync"):
                ArcticClientConfig(model_name="unused", training=TrainingConfig(peft={"peft_type": "LORA"}), **jobs)

    def test_cortex_forwards_arctic_and_use_liger(self):
        arctic = ArcticClientConfig(
            model_name="unused", training_gpus=1, training=TrainingConfig(model_provider="arctic", ep_size=4)
        )
        training = arctic._cortex_training_sub_job()["training_config"]
        self.assertEqual(training["model_provider"], "arctic")
        self.assertNotIn("model_options", training)

        liger = ArcticClientConfig(model_name="unused", training_gpus=1, training=TrainingConfig(use_liger=True))
        training = liger._cortex_training_sub_job()["training_config"]
        self.assertIs(training["use_liger"], True)
        self.assertNotIn("model_provider", training)

        legacy = ArcticClientConfig(
            model_name="unused", training_gpus=1, training=TrainingConfig(model_provider="prime_rl", ep_size=4)
        )
        training = legacy._cortex_training_sub_job()["training_config"]
        self.assertEqual(training["model_provider"], "arctic")

    def test_retired_model_providers_are_rejected(self):
        cases = (
            ("liger", "model_options.liger=true"),
            ("transformers", "must be 'huggingface', 'arctic' or"),
        )
        for provider, message in cases:
            with self.subTest(provider=provider), self.assertRaisesRegex(ValueError, message):
                TrainingConfig(model_provider=provider)

        with self.assertRaisesRegex(ValueError, "model_options.lm_head.cross_entropy"):
            TrainingConfig(model_provider="arctic", use_liger=True)

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
