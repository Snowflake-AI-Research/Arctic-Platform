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

"""Pytest fixtures shared by the Arctic SFT tests under ``tests/sft/``.

Mirrors ``tests/rl/conftest.py`` for ``gpu_serial``: the mark is a no-op unless an
autouse fixture in this directory engages :func:`gpu_serial_lock`.
"""

from __future__ import annotations

import os

import pytest

from arctic_platform.testing_utils import gpu_serial_lock

# Local HTTP SFT demos are lighter than full RL e2e, but DeepSpeed spin-up + a few
# training steps can still exceed the global ``timeout=300`` from pyproject.
_GPU_SERIAL_TIMEOUT = 900


def pytest_collection_modifyitems(items):
    """Give every ``gpu_serial`` test the larger timeout budget."""
    for item in items:
        if item.get_closest_marker("gpu_serial") is not None:
            item.add_marker(pytest.mark.timeout(_GPU_SERIAL_TIMEOUT))


@pytest.fixture(autouse=True)
def _scrub_stale_nccl_topo_file():
    """Drop an inherited ``NCCL_TOPO_FILE=/proc/self/fd/<N>`` left by a prior GPU test."""
    topo_file = os.environ.get("NCCL_TOPO_FILE", "")
    if topo_file.startswith("/proc/self/fd/"):
        os.environ.pop("NCCL_TOPO_FILE", None)
    yield


@pytest.fixture(autouse=True)
def _serialize_gpu_work(request):
    """Serialize ``gpu_serial`` bodies across xdist workers via the host-wide lock."""
    if request.node.get_closest_marker("gpu_serial") is None:
        yield
        return
    with gpu_serial_lock():
        yield
