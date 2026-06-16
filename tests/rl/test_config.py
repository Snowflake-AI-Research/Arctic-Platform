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

"""``ArcticRLClientConfig`` validation -- pure CPU, no GPU / Ray cluster / model load.

Guards the cheap-but-load-bearing config contract the heavyweight GPU tests rely on: the local-backend
"at least one engine" rule, the reconnect-mode bypass, comm-protocol host/port derivation, and literal enums.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from arctic_platform.rl.config import ArcticRLClientConfig
from arctic_platform.testing_utils import TestCasePlus


@pytest.mark.cpu
class TestArcticRLClientConfig(TestCasePlus):
    def test_local_requires_at_least_one_engine(self):
        """Local backend with every engine at 0 GPUs is rejected (no job could ever be created)."""
        with self.assertRaises(ValidationError):
            ArcticRLClientConfig(backend="local", model_name="dummy", training_gpus=0, sampling_gpus=0, log_prob_gpus=0)

    def test_single_engine_is_valid(self):
        """One engine > 0 satisfies the rule; the others may stay 0 (training-only / sampling-only topologies)."""
        for gpus in (dict(training_gpus=1), dict(sampling_gpus=1), dict(log_prob_gpus=1)):
            config = ArcticRLClientConfig(backend="local", model_name="dummy", **gpus)
            self.assertEqual(config.training_gpus + config.sampling_gpus + config.log_prob_gpus, 1)

    def test_reconnect_mode_skips_gpu_validation(self):
        """Reconnect mode (job ids preset) attaches to existing jobs, so the 0-GPU guard is bypassed."""
        config = ArcticRLClientConfig(backend="local", model_name="dummy", training_job_id=7)
        self.assertEqual(config.training_job_id, 7)
        self.assertEqual(config.training_gpus + config.sampling_gpus + config.log_prob_gpus, 0)

    def test_ray_protocol_derives_no_host_port(self):
        """ray comms are in-process actors -- host/port stay None."""
        config = ArcticRLClientConfig(model_name="dummy", comm_protocol="ray", training_gpus=1)
        self.assertIsNone(config.host)
        self.assertIsNone(config.port)

    def test_http_protocol_derives_host_and_port(self):
        """http binds the server on the node's routable IP at the default port 7000."""
        config = ArcticRLClientConfig(model_name="dummy", comm_protocol="http", training_gpus=1)
        self.assertIsNotNone(config.host)
        self.assertEqual(config.port, 7000)

    def test_explicit_host_port_preserved(self):
        """An explicitly supplied host/port is never overridden by derivation."""
        config = ArcticRLClientConfig(
            model_name="dummy", comm_protocol="http", host="1.2.3.4", port=9999, training_gpus=1
        )
        self.assertEqual(config.host, "1.2.3.4")
        self.assertEqual(config.port, 9999)

    def test_invalid_comm_protocol_rejected(self):
        with self.assertRaises(ValidationError):
            ArcticRLClientConfig(model_name="dummy", comm_protocol="carrier-pigeon", training_gpus=1)
