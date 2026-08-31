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
"""The ``cortex logs`` browser.

`format` and `log_cache` are pure stdlib and always importable; `app` needs
``textual``, so it is imported only when the browser actually runs.
"""

from __future__ import annotations

from typing import Any


def run_log_browser(jobs: Any, reader: Any, job_id: str | None = None, *, sub_job_id: str | None = None) -> None:
    """Open the log browser, on the job picker unless ``job_id`` is given."""
    try:
        from arctic_platform.client.cortex.tui.app import CortexLogApp
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "the log browser needs textual: pip install 'arctic_platform[tui]' (or use 'cortex logs --plain')"
        ) from exc

    CortexLogApp(jobs, reader, job_id, sub_job_id=sub_job_id, poll_interval=reader.poll_interval).run()
