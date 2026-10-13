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
"""Job comment and id file. Any Cortex recipe uses these."""

from __future__ import annotations

from arctic_platform.client.config import CortexConfig
from arctic_platform.client.transports.cortex import job_create_payload
from arctic_platform.client.transports.cortex import record_job_id


def test_comment_and_id_file_read_from_the_cortex_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ARCTIC_CORTEX_JOB_COMMENT", "karthik openhands")
    monkeypatch.setenv("ARCTIC_CORTEX_JOB_ID_FILE", str(tmp_path / "cortex_job_id"))
    config = CortexConfig(base_url="http://localhost:1")
    assert config.job_comment == "karthik openhands"
    assert config.job_id_file == str(tmp_path / "cortex_job_id")


def test_empty_comment_env_is_unset(monkeypatch):
    monkeypatch.setenv("ARCTIC_CORTEX_JOB_COMMENT", "")
    monkeypatch.delenv("ARCTIC_CORTEX_JOB_ID_FILE", raising=False)
    config = CortexConfig(base_url="http://localhost:1")
    assert config.job_comment is None
    assert config.job_id_file is None


def test_job_comment_is_added_only_when_set():
    body = job_create_payload([{"role": "training"}], "karthik openhands")
    assert body["comment"] == "karthik openhands"
    assert "comment" not in job_create_payload([{"role": "training"}], None)
    assert "comment" not in job_create_payload([{"role": "training"}], "")


def test_record_job_id_skips_a_missing_path(tmp_path):
    record_job_id(None, "job-1")
    record_job_id("", "job-1")
    destination = tmp_path / "cortex_job_id"
    record_job_id(str(destination), None)
    assert not destination.exists()
    record_job_id(str(destination), "job-1")
    assert destination.read_text() == "job-1\n"
