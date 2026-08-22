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

import pytest

from arctic_platform.client import CortexConfig
from arctic_platform.client import JobHandles
from arctic_platform.client import OnPremConfig
from arctic_platform.client import Request
from arctic_platform.client import SyncArcticRLClient
from arctic_platform.client import client as rl_client_module
from arctic_platform.opd import DEFAULT_PROCESSING
from arctic_platform.opd import ArcticOPDClient
from arctic_platform.opd import ArcticOPDClientConfig
from arctic_platform.opd.client import _fwd_bwd_body


class FakeTransport:
    def __init__(self, config, server_state=None):
        self.config = config
        self.server_state = server_state
        self.jobs = JobHandles()
        self.calls: list[Request] = []
        self.stopped = False

    def initialize(self):
        if self.config.training_gpus:
            self.jobs = JobHandles(training=11, sampling=12)
        else:
            self.jobs = JobHandles(sampling=21)
        return self.jobs

    def call(self, request):
        self.calls.append(request)
        return {"results": [{"ok": True}], "metrics": {}}

    def shutdown(self):
        self.stopped = True


def config(**updates):
    base = ArcticOPDClientConfig(
        student_model="student",
        teacher_model="teacher",
        training_gpus=2,
        sampling_gpus=2,
        teacher_sampling_gpus=4,
        backend=CortexConfig(base_url="http://example"),
    )
    return base.model_copy(update=updates)


@pytest.fixture
def client(monkeypatch):
    transports = []

    def make_transport(cfg, server_state=None):
        transport = FakeTransport(cfg, server_state=server_state)
        transports.append(transport)
        return transport

    monkeypatch.setattr(rl_client_module, "make_transport", make_transport)
    return ArcticOPDClient(config()), transports


def test_composes_two_sync_rl_clients(client):
    opd, transports = client
    assert isinstance(opd.student, SyncArcticRLClient)
    assert isinstance(opd.teacher, SyncArcticRLClient)
    assert opd.student.config.training_gpus == 2
    assert opd.student.config.sampling_gpus == 2
    assert opd.teacher.config.training_gpus == 0
    assert opd.teacher.config.sampling_gpus == 4
    assert opd.teacher.config.model_name == "teacher"
    assert opd.student.transport is transports[0]
    assert opd.teacher.transport is transports[1]


def test_routes_student_and_teacher_generate(client):
    opd, transports = client
    assert opd.generate([[1, 2]]) == [{"ok": True}]
    assert transports[0].calls[-1].op == "generate"
    assert transports[0].calls[-1].job_id == 12
    assert opd.generate_teacher([[1, 2, 3]]) == [{"ok": True}]
    assert transports[1].calls[-1].op == "generate"
    assert transports[1].calls[-1].job_id == 21


def test_fwd_bwd_defaults_distill_processing(client):
    opd, transports = client
    opd.fwd_bwd({"input_ids": [1]})
    request = transports[0].calls[-1]
    assert request.job_id == 11
    assert request.binary is True
    assert request.body["processing"] == DEFAULT_PROCESSING
    assert request.body["kwargs"] == {"input_ids": [1]}
    assert request.body["context"] == {"input_ids": [1]}


def test_onprem_fwd_bwd_uses_structured_batch_envelope():
    cfg = ArcticOPDClientConfig(
        student_model="student",
        teacher_model="teacher",
        training_gpus=1,
        sampling_gpus=1,
        teacher_sampling_gpus=1,
    )
    body = _fwd_bwd_body(cfg, {"input_ids": [1], "loss_mask": [True]}, None)
    assert body == {
        "batch": {"input_ids": [1], "loss_mask": [True]},
        "meta": {"zorro_train_enable": False},
        "processing": DEFAULT_PROCESSING,
    }


def test_cortex_sync_is_hf_and_never_targets_teacher(client):
    opd, transports = client
    opd.sync_weights()
    # Cortex has no wake-inference: sync then reset cache, student only.
    assert [request.op for request in transports[0].calls] == ["operation", "operation"]
    sync_request = transports[0].calls[-2]
    assert sync_request.body["payload"]["source_sub_job_id"] == 11
    assert sync_request.body["payload"]["target_sub_job_ids"] == [12]
    assert sync_request.body["payload"]["weight_format"] == "hf"
    assert 21 not in sync_request.body["payload"]["target_sub_job_ids"]
    assert transports[1].calls == []


def test_onprem_sync_weights_delegates_to_student(monkeypatch):
    transports = []

    def make_transport(cfg, server_state=None):
        transport = FakeTransport(cfg, server_state=server_state)
        transports.append(transport)
        return transport

    monkeypatch.setattr(rl_client_module, "make_transport", make_transport)
    opd = ArcticOPDClient(
        ArcticOPDClientConfig(
            student_model="student",
            teacher_model="teacher",
            training_gpus=1,
            sampling_gpus=1,
            teacher_sampling_gpus=1,
        )
    )
    opd.sync_weights()
    assert [request.op for request in transports[0].calls] == [
        "wake-inference",
        "operation",
        "wake-inference",
        "operation",
    ]
    sync_request = transports[0].calls[-3]
    assert sync_request.body["operation_type"] == "weight-sync"
    assert sync_request.body["payload"]["source_sub_job_id"] == 11
    assert sync_request.body["payload"]["target_sub_job_ids"] == [12]
    assert transports[1].calls == []


def test_reconnect_config_contains_all_three_ids(client):
    opd, _ = client
    reconnect = opd.reconnect_config()
    assert (reconnect.training_job_id, reconnect.sampling_job_id, reconnect.teacher_job_id) == (11, 12, 21)


def test_teacher_init_failure_cleans_up_student(monkeypatch):
    student = FakeTransport(config().student_transport_config())
    calls = 0

    def make_transport(_cfg, server_state=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            return student
        raise RuntimeError("teacher failed")

    monkeypatch.setattr(rl_client_module, "make_transport", make_transport)
    with pytest.raises(RuntimeError, match="teacher failed"):
        ArcticOPDClient(config())
    assert student.stopped


def test_partial_reconnect_ids_rejected():
    with pytest.raises(ValueError, match="requires training_job_id"):
        ArcticOPDClientConfig(
            student_model="student",
            teacher_model="teacher",
            training_gpus=1,
            sampling_gpus=1,
            teacher_sampling_gpus=1,
            training_job_id=1,
        )


def test_local_launch_requires_disjoint_student_and_teacher_devices():
    with pytest.raises(ValueError, match="must be disjoint"):
        ArcticOPDClientConfig(
            student_model="student",
            teacher_model="teacher",
            training_gpus=1,
            sampling_gpus=1,
            teacher_sampling_gpus=1,
            teacher_server_cuda_visible_devices="0,1",
            backend=OnPremConfig(
                launch_local_server=True,
                server_cuda_visible_devices="0,1",
            ),
        )


def test_local_launch_isolates_student_and_teacher_ray_ports():
    cfg = ArcticOPDClientConfig(
        student_model="student",
        teacher_model="teacher",
        training_gpus=1,
        sampling_gpus=1,
        teacher_sampling_gpus=1,
        teacher_port=18101,
        teacher_server_cuda_visible_devices="1",
        backend=OnPremConfig(
            launch_local_server=True,
            port=18100,
            server_cuda_visible_devices="0",
        ),
    )
    student_env = cfg.student_transport_config().backend.server_extra_env
    teacher_env = cfg.teacher_transport_config().backend.server_extra_env
    for key in (
        "RAY_PORT",
        "RAY_DASHBOARD_PORT",
        "MASTER_PORT",
        "ARL_WEIGHT_SYNC_PORT",
        "ARL_RAY_MIN_WORKER_PORT",
        "ARL_RAY_MAX_WORKER_PORT",
    ):
        assert student_env[key] != teacher_env[key], key
        assert int(student_env["ARL_RAY_MAX_WORKER_PORT"]) < int(teacher_env["ARL_RAY_MIN_WORKER_PORT"]) or int(
            teacher_env["ARL_RAY_MAX_WORKER_PORT"]
        ) < int(student_env["ARL_RAY_MIN_WORKER_PORT"])
