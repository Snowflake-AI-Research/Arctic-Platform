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
"""Cortex job ownership: what a job is created with, and when it is cancelled."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from arctic_platform.client import ArcticClient
from arctic_platform.client import ArcticClientConfig
from arctic_platform.client.config import CortexConfig
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.client.transports.cortex import job_create_payload

JOB = "job-1"
TRAINING = f"{JOB}:training:0"


def _backend(**fields) -> CortexConfig:
    return CortexConfig(base_url="http://localhost:1", **fields)


def test_comment_is_read_from_the_cortex_env(monkeypatch):
    monkeypatch.setenv("ARCTIC_CORTEX_JOB_COMMENT", "karthik openhands")
    assert _backend().job_comment == "karthik openhands"


def test_launcher_comment_name_is_read(monkeypatch):
    monkeypatch.delenv("ARCTIC_CORTEX_JOB_COMMENT", raising=False)
    monkeypatch.setenv("CORTEX_JOB_COMMENT", "karthik r2e-gym")
    assert _backend().job_comment == "karthik r2e-gym"


def test_empty_comment_env_is_unset(monkeypatch):
    monkeypatch.delenv("CORTEX_JOB_COMMENT", raising=False)
    monkeypatch.setenv("ARCTIC_CORTEX_JOB_COMMENT", "")
    assert _backend().job_comment is None


def test_create_body_carries_only_the_fields_that_are_set(monkeypatch):
    monkeypatch.delenv("ARCTIC_CORTEX_JOB_COMMENT", raising=False)
    monkeypatch.delenv("CORTEX_JOB_COMMENT", raising=False)
    sub_jobs = [{"role": "training"}]
    assert job_create_payload(sub_jobs, _backend()) == {"sub_job_configs": sub_jobs}
    assert job_create_payload(sub_jobs, _backend(job_comment="")) == {"sub_job_configs": sub_jobs}
    assert job_create_payload(
        sub_jobs, _backend(job_comment="karthik", idle_timeout_seconds=600, pending_timeout_seconds=3600)
    ) == {
        "sub_job_configs": sub_jobs,
        "comment": "karthik",
        "idle_timeout_seconds": 600,
        "pending_timeout_seconds": 3600,
    }


def test_idle_timeout_zero_disables_reclaim_and_is_sent():
    assert job_create_payload([], _backend(idle_timeout_seconds=0))["idle_timeout_seconds"] == 0


@pytest.mark.parametrize("fields", [{"idle_timeout_seconds": 60}, {"pending_timeout_seconds": 60}])
def test_timeouts_the_server_would_reject_fail_at_config_time(fields):
    with pytest.raises(ValidationError):
        _backend(**fields)


class FakeServer:
    """Answers create, get and cancel."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, str]] = []

    async def __call__(self, method: str, url: str, **kwargs):
        self.sent.append((method, url.rsplit("/", 1)[-1]))
        if url.endswith(":cancel"):
            return {}
        if method == "POST":
            return {"job_id": JOB}
        return {"status": "JOB_STATE_RUNNING", "sub_jobs": [{"job_type": "JOB_TYPE_TRAINING", "sub_job_id": TRAINING}]}

    @property
    def cancels(self) -> int:
        return sum(path.endswith(":cancel") for _, path in self.sent)


@pytest.fixture
def server(monkeypatch) -> FakeServer:
    fake = FakeServer()
    monkeypatch.setattr(CortexTransport, "_send", lambda self, *a, **k: fake(*a, **k))
    return fake


def _config() -> ArcticClientConfig:
    return ArcticClientConfig(model_name="m", backend=_backend(), training_gpus=1)


def test_start_returns_the_job_before_it_is_ready(server):
    transport = CortexTransport(_backend())
    transport.start(_config())
    assert transport.job_id == JOB
    assert transport.jobs.training is None
    assert server.sent == [("POST", "cortex-training")]

    assert transport.wait_until_ready().training == TRAINING


def _interrupt_the_wait(monkeypatch, interrupt: BaseException) -> None:
    def wait(self) -> None:
        raise interrupt

    monkeypatch.setattr(CortexTransport, "_wait_running", wait)


def test_client_without_wait_can_cancel_during_the_wait(server, monkeypatch):
    client = ArcticClient(_config(), wait=False)
    _interrupt_the_wait(monkeypatch, KeyboardInterrupt())
    with pytest.raises(KeyboardInterrupt):
        with client:
            client.wait_until_ready()
    assert server.cancels == 1


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(143)])
def test_blocking_client_cancels_when_the_wait_is_interrupted(server, monkeypatch, interrupt):
    _interrupt_the_wait(monkeypatch, interrupt)
    with pytest.raises(type(interrupt)):
        ArcticClient(_config())
    assert server.cancels == 1


def test_shutdown_cancels_once(server):
    client = ArcticClient(_config())
    client.shutdown()
    client.shutdown()
    assert server.cancels == 1
