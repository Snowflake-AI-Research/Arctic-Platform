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
"""A launched local HTTP server that dies must fail the client fast with its exit code."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from arctic_platform.client.transports.onprem_http import HttpTransport

# Never bound or dialed: session is a MagicMock and Popen is monkeypatched, so the
# value is irrelevant to these tests.
_PORT = 65500
_BASE_URL = f"http://localhost:{_PORT}"


def test_wait_server_healthy_fails_fast_on_process_exit():
    transport = HttpTransport.__new__(HttpTransport)
    transport.proc = SimpleNamespace(poll=lambda: 1, returncode=1)
    transport.session = MagicMock()
    transport.timeout = 1.0
    transport.base_url = _BASE_URL

    with pytest.raises(RuntimeError, match="exited with code 1"):
        transport._wait_server_healthy(timeout=30)


def test_launch_server_reraises_exit_code_and_reaps_orphan(monkeypatch):
    transport = HttpTransport.__new__(HttpTransport)
    transport.timeout = 1.0
    transport.base_url = _BASE_URL
    transport.session = MagicMock()
    transport.proc = None
    transport.config = SimpleNamespace(
        training_gpus=1,
        sampling_gpus=0,
        log_prob_gpus=0,
        backend=SimpleNamespace(
            port=_PORT,
            colocate=False,
            server_cuda_visible_devices=None,
            startup_timeout=5.0,
        ),
    )

    terminated = {"called": False}

    class FakeProc:
        returncode = 17

        def poll(self):
            return 17  # already exited

    def fake_popen(*args, **kwargs):
        # No stdout/stderr redirection: server inherits the client's fds.
        assert "stdout" not in kwargs and "stderr" not in kwargs
        return FakeProc()

    monkeypatch.setattr("subprocess.Popen", fake_popen)
    monkeypatch.setattr(transport, "_terminate_server", lambda: terminated.__setitem__("called", True))

    with pytest.raises(RuntimeError, match="exited with code 17"):
        transport._launch_server()

    assert terminated["called"], "orphaned server process must be reaped on launch failure"
