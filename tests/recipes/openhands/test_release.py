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
"""The launcher's exit trap cancels the job named in the id file."""

from __future__ import annotations

from openhands_localization import release

from arctic_platform.client.transports.cortex import CortexTransport


def test_release_cancels_each_job_id(monkeypatch):
    monkeypatch.setenv("ARCTIC_CORTEX_BASE_URL", "http://example.invalid")
    cancelled = []
    monkeypatch.setattr(CortexTransport, "cancel_job", lambda self, job_id: cancelled.append(job_id))
    release.main(["job-1", "job-2"])
    assert cancelled == ["job-1", "job-2"]
