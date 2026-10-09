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

from __future__ import annotations

import pytest

from arctic_platform.correctness.harness.dss_driver import sampling_payload


def test_sampling_job_selects_fla_gdn_prefill() -> None:
    pytest.importorskip(
        "arctic_inference.server.config",
        reason="Arctic Inference is not installed with the [sft,testing] extras.",
    )
    from arctic_platform.common.utils.server_models import build_model_config

    payload = sampling_payload("/tmp/weights", 0, dtype="bfloat16", n_gpus=1, max_seq_len=10241)
    engine = build_model_config(payload["model_name"], payload["inference_config"]["vllm_config"])

    assert engine.to_engine_kwargs()["gdn_prefill_backend"] == "triton"
