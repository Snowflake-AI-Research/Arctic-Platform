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

"""CPU tests for WorkerPatch.sleep's vLLM 0.30 suspend path.

``tests/unit_tests/test_spec_dec_sleep.py`` covers sleep/wake with a real
LLM and is GPU-only; it is not run in this package's CPU CI.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from arctic_inference.vllm.patches import WorkerPatch


def _worker(*, enable_nccl_comm_suspend: bool, level2: bool = False):
    model = SimpleNamespace(named_buffers=lambda: iter(()))
    worker = SimpleNamespace(
        sleep_mode_backend=MagicMock(),
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                enable_nccl_comm_suspend=enable_nccl_comm_suspend
            )
        ),
        model_runner=SimpleNamespace(drafter=None, model=model),
    )
    if level2:
        worker._skip_drafter_param_snapshot = False
    return worker


def test_sleep_suspends_backend_without_nccl_when_flag_off():
    worker = _worker(enable_nccl_comm_suspend=False)
    with patch(
        "vllm.distributed.parallel_state.suspend_device_comms"
    ) as suspend_comms:
        WorkerPatch.sleep(worker, level=1)

    worker.sleep_mode_backend.suspend.assert_called_once_with(1)
    suspend_comms.assert_not_called()


def test_sleep_suspends_device_comms_when_flag_on():
    worker = _worker(enable_nccl_comm_suspend=True)
    with patch(
        "vllm.distributed.parallel_state.suspend_device_comms"
    ) as suspend_comms:
        WorkerPatch.sleep(worker, level=1)

    worker.sleep_mode_backend.suspend.assert_called_once_with(1)
    suspend_comms.assert_called_once_with()


def test_sleep_level2_still_routes_through_suspend_backend():
    worker = _worker(enable_nccl_comm_suspend=False, level2=True)
    WorkerPatch.sleep(worker, level=2)

    worker.sleep_mode_backend.suspend.assert_called_once_with(2)
    assert worker._sleep_level == 2
    assert worker._sleep_saved_drafter_state == {}
    assert worker._sleep_saved_buffers == {}
