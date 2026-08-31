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
"""The control plane: request shapes, status handling, and `attach`."""

from __future__ import annotations

import pytest

from arctic_platform.client.cortex.jobs import DEBUG_OPTIONS_ENV
from arctic_platform.client.transports.cortex import _is_connect_error


def _job(status="JOB_STATE_RUNNING", **extra):
    return {"job_id": "J1", "status": status, **extra}


class TestRequestShapes:
    def test_get_addresses_the_job(self, jobs, prefix):
        control = jobs([_job()])
        control.get("J1")
        assert control.session.last["method"] == "GET"
        assert control.session.last["url"] == f"{prefix}/J1"

    def test_list_unwraps_the_jobs_key(self, jobs):
        control = jobs([{"jobs": [_job()]}])
        assert [job["job_id"] for job in control.list()] == ["J1"]
        assert control.session.last["params"] is None

    def test_list_passes_a_status_filter(self, jobs):
        control = jobs([{"jobs": []}])
        control.list(status="RUNNING")
        assert control.session.last["params"] == {"status": "RUNNING"}

    def test_cancel_uses_the_colon_action(self, jobs, prefix):
        control = jobs([{}])
        control.cancel("J1")
        assert (control.session.last["method"], control.session.last["url"]) == ("POST", f"{prefix}/J1:cancel")

    def test_checkpoints_unwraps_the_checkpoints_key(self, jobs, prefix):
        control = jobs([{"checkpoints": [{"checkpoint_id": "ck"}]}])
        assert control.checkpoints("J1") == [{"checkpoint_id": "ck"}]
        assert control.session.last["url"] == f"{prefix}/J1/checkpoints"

    def test_load_posts_the_checkpoint_and_returns_a_request_id(self, jobs):
        control = jobs([{"request_id": "R1"}])
        assert control.load("J1", "ck", target_sub_job_id="J1:training:0") == "R1"
        assert control.session.last["json"] == {"checkpoint_id": "ck", "target_sub_job_id": "J1:training:0"}

    def test_load_omits_unset_routing(self, jobs):
        control = jobs([{"request_id": "R1"}])
        control.load("J1", "ck")
        assert control.session.last["json"] == {"checkpoint_id": "ck"}


class TestSubmit:
    def test_posts_the_body_to_the_collection(self, jobs, prefix):
        control = jobs([{"job_id": "J9"}])
        body = {"job_id": "J9", "sub_job_configs": [{"job_type": "training"}]}
        assert control.submit(body) == {"job_id": "J9"}
        assert (control.session.last["method"], control.session.last["url"]) == ("POST", prefix)
        assert control.session.last["json"] == body

    def test_only_retries_when_the_request_never_landed(self, jobs):
        """Retrying a create that may have landed would spawn a duplicate job."""
        control = jobs([{}])
        control.submit({"sub_job_configs": [{}]})
        assert control.session.last["retry_on"] is _is_connect_error

    def test_rejects_a_body_without_sub_jobs(self, jobs):
        with pytest.raises(ValueError, match="sub_job_configs"):
            jobs().submit({"job_id": "J9"})

    def test_rejects_debug_options_by_default(self, jobs, monkeypatch):
        monkeypatch.delenv(DEBUG_OPTIONS_ENV, raising=False)
        with pytest.raises(ValueError, match="internal-only"):
            jobs().submit({"sub_job_configs": [{}], "debug": {"image": "x"}})

    def test_allows_debug_options_when_enabled(self, jobs, monkeypatch):
        monkeypatch.setenv(DEBUG_OPTIONS_ENV, "1")
        control = jobs([{}])
        control.submit({"sub_job_configs": [{}], "debug": {"image": "x"}})
        assert control.session.last["json"]["debug"] == {"image": "x"}


class TestWait:
    def test_returns_once_running(self, jobs):
        control = jobs([_job("JOB_STATE_PENDING"), _job("JOB_STATE_RUNNING")])
        assert control.wait("J1")["status"] == "JOB_STATE_RUNNING"
        assert len(control.session.calls) == 2

    def test_accepts_an_already_short_status(self, jobs):
        """The server has answered both 'JOB_STATE_RUNNING' and 'running'."""
        control = jobs([_job("running")])
        assert control.wait("J1")["status"] == "running"

    def test_unwraps_a_nested_job(self, jobs):
        control = jobs([{"job": _job()}])
        assert control.wait("J1") == {"job": _job()}

    def test_fails_fast_on_a_terminal_state(self, jobs):
        control = jobs([_job("JOB_STATE_FAILED", reason="out of capacity")])
        with pytest.raises(RuntimeError, match="terminal state 'failed': out of capacity"):
            control.wait("J1")

    def test_times_out_rather_than_polling_forever(self, jobs):
        control = jobs([_job("JOB_STATE_PENDING")] * 50)
        control.poll_timeout = -1  # already expired
        with pytest.raises(TimeoutError, match="did not become running"):
            control.wait("J1")


class TestCapacity:
    def test_reads_the_reservation(self, jobs):
        reserved = jobs([{"has_reservation": True, "reserved_gpus": 8, "in_use_gpus": 2, "available_gpus": 6}])
        assert reserved.capacity().available_gpus == 6

    def test_defaults_fill_proto3_omissions(self, jobs):
        """proto3 JSON omits zero/false, so an unreserved account answers '{}'."""
        capacity = jobs([{}]).capacity()
        assert (capacity.has_reservation, capacity.reserved_gpus, capacity.available_gpus) == (False, 0, 0)

    def test_ignores_fields_the_client_does_not_know(self, jobs):
        assert jobs([{"reserved_gpus": 4, "future_field": "x"}]).capacity().reserved_gpus == 4


class TestAttach:
    def _job_with_sub_jobs(self):
        return {
            "job_id": "J1",
            "status": "JOB_STATE_RUNNING",
            "sub_jobs": [
                {
                    "sub_job_id": "J1:training:0",
                    "job_type": "JOB_TYPE_TRAINING",
                    "model_name": "Qwen/Qwen3-8B",
                    "training_config": {"n_gpus": 4, "max_seq_len": 4096},
                },
                {
                    "sub_job_id": "J1:sampling:0",
                    "job_type": "sampling",
                    "model_name": "Qwen/Qwen3-8B",
                    "inference_config": {"n_gpus": 2},
                },
            ],
        }

    def test_maps_sub_jobs_onto_client_roles(self, jobs):
        config = jobs([self._job_with_sub_jobs()]).attach("J1")
        assert config.training_job_id == "J1:training:0"
        assert config.sampling_job_id == "J1:sampling:0"
        assert (config.training_gpus, config.sampling_gpus) == (4, 2)

    def test_carries_the_model_and_seq_len(self, jobs):
        config = jobs([self._job_with_sub_jobs()]).attach("J1")
        assert (config.model_name, config.max_seq_len) == ("Qwen/Qwen3-8B", 4096)

    def test_keeps_the_same_backend_connection(self, jobs, config):
        assert jobs([self._job_with_sub_jobs()]).attach("J1").backend == config

    def test_maps_log_probability_to_the_log_prob_role(self, jobs):
        job = {"sub_jobs": [{"sub_job_id": "J1:log_probability:0", "job_type": "log_probability"}]}
        assert jobs([job]).attach("J1").log_prob_job_id == "J1:log_probability:0"

    def test_a_live_sub_job_without_n_gpus_still_counts_as_one(self, jobs):
        """n_gpus only sizes a role; 0 would read as 'role disabled'."""
        job = {"sub_jobs": [{"sub_job_id": "J1:training:0", "job_type": "training"}]}
        assert jobs([job]).attach("J1").training_gpus == 1

    def test_unwraps_a_nested_job(self, jobs):
        assert jobs([{"job": self._job_with_sub_jobs()}]).attach("J1").training_gpus == 4

    def test_rejects_a_job_with_no_sub_jobs(self, jobs):
        with pytest.raises(ValueError, match="no sub_jobs"):
            jobs([{"job_id": "J1"}]).attach("J1")

    def test_rejects_only_unknown_sub_job_types(self, jobs):
        job = {"sub_jobs": [{"sub_job_id": "J1:mystery:0", "job_type": "mystery"}]}
        with pytest.raises(ValueError, match="no sub-job of a known type"):
            jobs([job]).attach("J1")


class TestPollRequest:
    def test_returns_the_completed_result(self, jobs, prefix):
        control = jobs(
            [
                {"status": "REQUEST_STATE_RUNNING"},
                {"status": "REQUEST_STATE_COMPLETED", "result": {"loss": 1.5}},
            ]
        )
        assert control.poll_request("J1", "R1") == {"loss": 1.5}
        assert control.session.calls[0]["url"] == f"{prefix}/J1/requests/R1"

    def test_raises_on_a_failed_request(self, jobs):
        control = jobs([{"status": "REQUEST_STATE_FAILED", "error": "boom"}])
        with pytest.raises(RuntimeError, match="boom"):
            control.poll_request("J1", "R1")

    def test_times_out_rather_than_polling_forever(self, jobs):
        control = jobs([{"status": "REQUEST_STATE_RUNNING"}] * 20)
        control.poll_timeout = -1
        with pytest.raises(TimeoutError, match="did not complete"):
            control.poll_request("J1", "R1")
