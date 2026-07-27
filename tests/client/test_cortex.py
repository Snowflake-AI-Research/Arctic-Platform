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
"""Cortex transport translation-layer tests (pure logic, no network).

Covers the CreateJob wire schema builders, GPU-gated sub-job assembly, and the
result decoders. The HTTP submit/poll layer needs a live server and is not
exercised here.
"""

from __future__ import annotations

import base64
import hashlib

import pytest
import torch

from arctic_platform.client import wire
from arctic_platform.client.config import ArcticRLClientConfig
from arctic_platform.client.transports import cortex
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.client.transports.cortex import InferenceConfig
from arctic_platform.client.transports.cortex import JobType
from arctic_platform.client.transports.cortex import SubJobConfig
from arctic_platform.client.transports.cortex import TrainingConfig


class TestWireSchema:
    def test_training_sub_job_to_wire(self):
        """A training sub-job serializes model + training_config to the wire schema."""
        sub = SubJobConfig.training_job(
            "Qwen/Qwen3-0.6B",
            optimizer={"name": "AdamW", "lr": 1e-5},
            max_seq_len=128,
            train_batch_size=8,
            n_gpus=4,
            dtype="bfloat16",
        )
        sub.validate()
        wired = sub.to_wire()
        assert wired["job_type"] == "training"
        assert wired["model_name"] == "Qwen/Qwen3-0.6B"
        assert wired["dtype"] == "bfloat16"
        assert wired["training_config"] == {
            "optimizer": {"name": "AdamW", "lr": 1e-5},
            "max_seq_len": 128,
            "train_batch_size": 8,
            "n_gpus": 4,
        }

    def test_sampling_sub_job_to_wire(self):
        """A sampling sub-job serializes an inference_config block."""
        sub = SubJobConfig.sampling_job("m", max_seq_len=64, n_gpus=2)
        sub.validate()
        wired = sub.to_wire()
        assert wired["job_type"] == "sampling"
        assert wired["inference_config"] == {"max_seq_len": 64, "n_gpus": 2}

    def test_sampling_job_rejects_training_type(self):
        """sampling_job() only accepts SAMPLING or LOG_PROBABILITY."""
        with pytest.raises(ValueError, match="SAMPLING or LOG_PROBABILITY"):
            SubJobConfig.sampling_job("m", max_seq_len=8, n_gpus=1, job_type=JobType.TRAINING)

    def test_training_config_validate_requires_optimizer(self):
        """An empty optimizer is rejected."""
        with pytest.raises(ValueError, match="optimizer"):
            TrainingConfig(optimizer={}, max_seq_len=8, train_batch_size=1, n_gpus=1).validate()

    def test_inference_config_validate_positive_gpus(self):
        """n_gpus must be > 0."""
        with pytest.raises(ValueError, match="n_gpus"):
            InferenceConfig(max_seq_len=8, n_gpus=0).validate()


class TestBuildSubJobs:
    def _transport(self, **gpus) -> CortexTransport:
        cfg = ArcticRLClientConfig(
            backend="cortex",
            model_name="m",
            cortex_base_url="http://mock",
            max_seq_len=128,
            training_config={"optimizer": {"lr": 1e-5}, "train_batch_size": 8},
            **gpus,
        )
        return CortexTransport(cfg)

    def test_gpu_gating_omits_zero_count_jobs(self):
        """A sub-job is created only when its GPU count is > 0."""
        subs = self._transport(training_gpus=4)._build_sub_jobs()
        assert [s.job_type for s in subs] == [JobType.TRAINING]

    def test_creation_order_inference_before_training(self):
        """Order is sampling, log_probability, then training (NCCL root last)."""
        subs = self._transport(training_gpus=4, sampling_gpus=2, log_prob_gpus=1)._build_sub_jobs()
        assert [s.job_type for s in subs] == [JobType.SAMPLING, JobType.LOG_PROBABILITY, JobType.TRAINING]


class TestResultDecoders:
    def test_decode_result_payload_roundtrip(self):
        """A base64 DSSST1 result payload decodes back to its dict."""
        result = {
            "wire_format": wire.WIRE_FORMAT_VERSION,
            "encoding": "base64",
            "payload_b64": base64.b64encode(wire.dumps({"loss": 1.5})).decode(),
        }
        assert cortex._decode_result_payload(result) == {"loss": 1.5}

    def test_decode_result_payload_passthrough_when_not_wire(self):
        """A plain result (no wire_format) returns None so the caller keeps it."""
        assert cortex._decode_result_payload({"loss": 0.1}) is None

    def test_decode_result_payload_rejects_non_base64(self):
        """A non-base64 encoding is an error."""
        with pytest.raises(RuntimeError, match="base64"):
            cortex._decode_result_payload(
                {"wire_format": wire.WIRE_FORMAT_VERSION, "encoding": "gzip", "payload_b64": "x"}
            )

    def test_restore_generate_result_lists(self):
        """Tensors in a nested generate result become plain Python lists."""
        out = cortex._restore_generate_result_lists({"a": torch.tensor([1, 2]), "b": [torch.tensor([3])]})
        assert out == {"a": [1, 2], "b": [[3]]}

    def test_result_chunk_event_sha256_mismatch(self):
        """A result_chunk event whose payload sha256 disagrees is rejected."""
        event = {"type": "result_chunk", "payload_b64": base64.b64encode(b"hello").decode(), "payload_sha256": "bad"}
        with pytest.raises(RuntimeError, match="sha256"):
            cortex._decode_result_chunk_event(event)

    def test_result_chunk_event_valid(self):
        """A well-formed result_chunk event returns its raw bytes."""
        payload = b"hello"
        event = {
            "type": "result_chunk",
            "payload_b64": base64.b64encode(payload).decode(),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
        }
        assert cortex._decode_result_chunk_event(event) == payload

    def test_non_result_chunk_event_ignored(self):
        """Non result_chunk events yield None."""
        assert cortex._decode_result_chunk_event({"type": "status"}) is None
