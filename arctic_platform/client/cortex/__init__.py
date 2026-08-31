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
"""Cortex control plane and tooling: everything you do *about* a job.

`arctic_platform.client` is a session client -- constructing an `ArcticClient`
creates jobs and `shutdown()` cancels them. This package covers the rest of the
Cortex surface: listing and inspecting jobs you don't own, submitting raw
CreateJob bodies, capacity, checkpoints, logs, and the `cortex` CLI.

It shares the transport's `CortexSession` (auth + retry) and the client's
`CortexConfig`, so there is one connection story across the library and the CLI.
"""

from arctic_platform._dependency_groups import require_any_dep_group

require_any_dep_group("cortex")

from arctic_platform.client.cortex.connection import login  # noqa: E402
from arctic_platform.client.cortex.connection import login_path  # noqa: E402
from arctic_platform.client.cortex.connection import logout  # noqa: E402
from arctic_platform.client.cortex.connection import read_connection_file  # noqa: E402
from arctic_platform.client.cortex.connection import redacted  # noqa: E402
from arctic_platform.client.cortex.connection import resolve  # noqa: E402
from arctic_platform.client.cortex.jobs import Capacity  # noqa: E402
from arctic_platform.client.cortex.jobs import CortexJobs  # noqa: E402

__all__ = [
    "Capacity",
    "CortexJobs",
    "login",
    "login_path",
    "logout",
    "read_connection_file",
    "redacted",
    "resolve",
]
