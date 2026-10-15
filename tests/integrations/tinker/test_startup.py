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

from __future__ import annotations

import logging
import warnings

import aiohttp

from arctic_platform.client import ArcticClientConfig
from arctic_platform.client import CortexConfig
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.integrations.tinker import _enable_progress_log
from arctic_platform.integrations.tinker import _QuietHub


def test_wait_running_logs_each_status_once(caplog, monkeypatch):
    transport = CortexTransport.__new__(CortexTransport)
    states = iter(["initializing", "initializing", "running"])
    transport._job = lambda: {"status": next(states)}
    transport.config = CortexConfig(base_url="http://mock", job_ready_timeout=30)
    transport.poll_interval = 0
    monkeypatch.setattr("arctic_platform.client.transports.cortex.time.sleep", lambda _delay: None)

    with caplog.at_level(logging.INFO, logger="arctic_platform.client.transports.cortex"):
        transport._wait_running()

    messages = [record.message for record in caplog.records]
    assert messages[0] == "Cortex job is initializing. Workers are starting and loading the model."
    assert messages[1].startswith("Cortex job is running (")
    assert len(messages) == 2


def test_shutdown_closes_the_session():
    transport = CortexTransport(CortexConfig(base_url="http://mock"))

    async def open_session() -> aiohttp.ClientSession:
        transport._session = aiohttp.ClientSession()
        return transport._session

    session = transport._on_loop(open_session()).result()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ResourceWarning)
        transport.shutdown()

    assert session.closed
    assert not any("Unclosed" in str(item.message) for item in caught)


def test_initialize_preserves_job_comment_with_client_config(monkeypatch):
    config = ArcticClientConfig(
        backend=CortexConfig(base_url="http://mock"),
        model_name="test-model",
        training_gpus=1,
    )
    transport = CortexTransport(config.backend)
    sent = {}

    async def send(method, url, **kwargs):
        sent.update(kwargs["json"])
        return {"job_id": "job-123"}

    monkeypatch.setenv("CORTEX_JOB_COMMENT", "tinker-test")
    monkeypatch.setattr(transport, "_send", send)
    monkeypatch.setattr(transport, "_wait_running", lambda: None)
    monkeypatch.setattr(transport, "_capture_sub_jobs", lambda: {"training": "job-123:training"})
    jobs = transport.initialize(config)
    assert sent == {"sub_job_configs": config.to_cortex(), "comment": "tinker-test"}
    assert jobs.training == "job-123:training"


def test_quiet_hub_restores_the_logger_level():
    hub = logging.getLogger("huggingface_hub.utils._http")
    previous = hub.level
    hub.setLevel(logging.WARNING)
    try:
        with _QuietHub():
            assert hub.level == logging.ERROR
        assert hub.level == logging.WARNING
    finally:
        hub.setLevel(previous)


def test_progress_log_attaches_when_logging_is_unconfigured():
    root = logging.getLogger()
    saved_root = list(root.handlers)
    log = logging.getLogger("arctic_platform")
    saved_level = log.level
    for handler in saved_root:
        root.removeHandler(handler)
    try:
        _enable_progress_log()
        assert any(getattr(handler, "_arctic_progress", False) for handler in log.handlers)
    finally:
        for handler in list(log.handlers):
            if getattr(handler, "_arctic_progress", False):
                log.removeHandler(handler)
                handler.close()
        log.setLevel(saved_level)
        for handler in saved_root:
            root.addHandler(handler)
