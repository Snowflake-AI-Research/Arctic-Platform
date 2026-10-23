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
"""The SkyRL driver cancels the Cortex jobs it created, however it exits."""

from __future__ import annotations

import os
import signal
from types import SimpleNamespace

import pytest

pytest.importorskip("ray")

from arctic_platform.integrations import _cortex_dispatch  # noqa: E402
from arctic_platform.integrations.skyrl import entrypoint  # noqa: E402


class FakeClient:
    def __init__(self) -> None:
        self.shutdowns = 0

    def shutdown(self) -> None:
        self.shutdowns += 1


@pytest.fixture
def created(monkeypatch) -> list[FakeClient]:
    clients: list[FakeClient] = []

    def create(config, *, wait=True):
        clients.append(FakeClient())
        return clients[-1]

    monkeypatch.setattr(_cortex_dispatch, "create_cortex_client", create)
    monkeypatch.setattr(entrypoint, "_created_clients", [])
    return clients


def _config(**job_ids) -> SimpleNamespace:
    fields = dict(training_job_id=None, sampling_job_id=None, log_prob_job_id=None)
    return SimpleNamespace(**(fields | job_ids))


def test_only_clients_that_created_jobs_are_released(created):
    with entrypoint._releasing_created_jobs():
        entrypoint._create_cortex_client(_config())
        entrypoint._create_cortex_client(_config(training_job_id="job-1:training:0"))
    assert [client.shutdowns for client in created] == [1, 0]


def test_sigterm_releases_the_jobs_and_restores_the_handler(created):
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as exited:
        with entrypoint._releasing_created_jobs():
            entrypoint._create_cortex_client(_config())
            os.kill(os.getpid(), signal.SIGTERM)
    assert exited.value.code == 128 + signal.SIGTERM
    assert created[0].shutdowns == 1
    assert signal.getsignal(signal.SIGTERM) is previous


def test_a_failed_cancel_does_not_hide_the_original_error(created):
    def fail() -> None:
        raise RuntimeError("cancel failed")

    with pytest.raises(ValueError, match="training failed"):
        with entrypoint._releasing_created_jobs():
            entrypoint._create_cortex_client(_config()).shutdown = fail
            raise ValueError("training failed")
