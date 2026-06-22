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

"""Pytest fixtures shared by the Arctic RL tests under ``tests/rl/``."""

from __future__ import annotations

import os

import pytest
from rl_harness import gpu_serial_lock

# Per-test wall-clock budget for the heavyweight GPU tests. Healthy spin-up is well under a minute, but the http
# session retries a wedged init up to 3 times on fresh ports (job_ready_timeout=240s each), so the test must be
# allowed to outlast that worst case. Overrides the global ``timeout=300`` from pyproject for these tests only.
_GPU_SERIAL_TIMEOUT = 900


def pytest_collection_modifyitems(items):
    """Give every ``gpu_serial`` test the larger timeout budget (set at collection, before pytest-timeout arms)."""
    for item in items:
        if item.get_closest_marker("gpu_serial") is not None:
            item.add_marker(pytest.mark.timeout(_GPU_SERIAL_TIMEOUT))


@pytest.fixture(autouse=True)
def _scrub_stale_nccl_topo_file():
    """Drop an inherited per-process ``NCCL_TOPO_FILE=/proc/self/fd/<N>`` left in the driver env by a prior GPU test.

    Belt-and-suspenders with the same guard in ``deepspeed_worker``: scrubbing the driver env before a test spawns
    its server subprocess keeps the stale handle from propagating to any NCCL init (vLLM as well as DeepSpeed).
    """
    topo_file = os.environ.get("NCCL_TOPO_FILE", "")
    if topo_file.startswith("/proc/self/fd/"):
        os.environ.pop("NCCL_TOPO_FILE", None)
    yield


@pytest.fixture(autouse=True)
def _serialize_gpu_work(request):
    """Serialize GPU-heavy test bodies across xdist workers via the host-wide lock.

    Autouse, but only engages for tests marked ``gpu_serial`` (the heavyweight GPU modules) so the fast CPU tests in
    this directory are never made to queue behind a multi-minute GPU test under ``-n N``.
    """
    if request.node.get_closest_marker("gpu_serial") is None:
        yield
        return
    with gpu_serial_lock():
        yield
