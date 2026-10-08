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

"""The hosted transport's job creation, data-plane sequence and checkpoint resolution, without a server."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from arctic_platform.correctness.harness import hosted
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.hosted import HostedConnection
from arctic_platform.correctness.harness.hosted import HostedTrainingJob
from arctic_platform.correctness.harness.hosted import HostedTransport
from arctic_platform.correctness.harness.hosted import save_and_resolve_checkpoint_id
from arctic_platform.correctness.harness.hosted import training_sub_job
from arctic_platform.correctness.harness.transport import Checkpoint

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "qwen3-8b" / "h200" / "train-sft-full-4gpus-64k.config"
CONNECTION = HostedConnection(host="server", database="DB", schema="SCHEMA", pat="not-a-token")


def _config():
    return load_config(CONFIG)


def _wire(cfg, **kwargs) -> dict:
    """The create-job body for this config, validated the way ``create_job`` validates it."""
    sub_job = training_sub_job(cfg, **kwargs)
    sub_job.validate()
    return sub_job.to_wire()


class _Client:
    """Enough of the hosted client to record what the transport asked it to do."""

    def __init__(self, *, losses=(), save_result=None, catalog=None, added=()) -> None:
        self.losses = list(losses)
        self.save_result = {} if save_result is None else dict(save_result)
        self.catalog = [dict(entry) for entry in (catalog or [])]
        self.added = [dict(entry) for entry in added]
        self.calls: list = []
        self.created: list = []
        self.cancelled: list = []
        self.closed = False
        self._results: dict = {}

    def _request(self, result: dict) -> str:
        request_id = f"request-{len(self._results) + 1}"
        self._results[request_id] = result
        return request_id

    def create_job(self, sub_jobs, hardware=None):
        for sub_job in sub_jobs:
            sub_job.validate()
        self.created.append((sub_jobs[0].to_wire(), hardware))
        return f"job-{len(self.created)}"

    def wait_for_job(self, job_id):
        self.calls.append(("wait", job_id))

    def cancel_job(self, job_id):
        self.cancelled.append(job_id)

    def forward_backward(self, job_id, body):
        self.calls.append(("fwd-bwd", job_id, body))
        return self._request({"avg_loss": self.losses.pop(0) if self.losses else 0.0})

    def step(self, job_id, learning_rate=None):
        self.calls.append(("step", job_id, learning_rate))
        return self._request({"global_steps": len(self._results)})

    def save(self, job_id, checkpoint_type=None):
        self.calls.append(("save", job_id, checkpoint_type))
        self.catalog = self.catalog + self.added
        return self._request(self.save_result)

    def list_checkpoints(self, job_id):
        return [dict(entry) for entry in self.catalog]

    def poll_request(self, job_id, request_id):
        return self._results[request_id]

    def close(self):
        self.closed = True


def test_the_created_sub_job_is_the_config_file_s_training_sub_job() -> None:
    """Whatever the operator wrote reaches the server: typed fields as themselves, the rest as passthrough."""
    cfg = _config()
    wire = _wire(cfg)

    assert wire["job_type"] == "training"
    assert wire["model_name"] == cfg.model_name
    assert wire["dtype"] == cfg.sub_job["dtype"]
    assert wire["seed"] == cfg.sub_job["seed"]
    for key, value in cfg.training.items():
        if key == "optimizer":
            continue
        assert wire["training_config"][key] == value, key
    for key, value in cfg.training["optimizer"].items():
        assert wire["training_config"]["optimizer"][key] == value, key


def test_building_a_sub_job_leaves_the_loaded_config_alone() -> None:
    """The client canonicalizes the optimizer block it is handed, and the rest of the run reads that block."""
    cfg = _config()
    before = json.dumps(cfg.sub_job, sort_keys=True)

    _wire(cfg)

    assert json.dumps(cfg.sub_job, sort_keys=True) == before


def test_optimizer_state_loading_is_left_at_the_server_default() -> None:
    """A resume that restores the moments is what is wanted: both jobs read one config, so DP size is equal."""
    assert "load_optimizer_states" not in _wire(_config())["training_config"]


def test_an_ordinary_sub_job_carries_no_source_checkpoint() -> None:
    assert "source_checkpoint_info" not in _wire(_config())


def test_a_resumed_sub_job_names_the_checkpoint_and_the_job_that_saved_it() -> None:
    wire = _wire(_config(), source_checkpoint_info={"checkpoint_id": "cp_1", "source_job_id": "job-a"})

    source = wire["source_checkpoint_info"]
    assert source["checkpoint_id"] == "cp_1"
    assert source["source_job_id"] == "job-a"
    assert source.get("is_external", False) is False


def test_the_checkpoint_id_is_the_stage_path_segment() -> None:
    """The DeepSpeed tag beside it names a directory inside the saving job and resolves from nowhere else."""
    client = _Client(
        save_result={
            "stage_path": "snow://stage/run/checkpoints/cp_4c1f0a2b/global_step10/",
            "checkpoint_tag": "global_step10",
        }
    )

    assert save_and_resolve_checkpoint_id(client, "job-1", "resumable") == "cp_4c1f0a2b"
    assert ("save", "job-1", "resumable") in client.calls


def test_the_checkpoint_id_falls_back_to_what_the_save_added_to_the_catalog() -> None:
    """The catalog's order is not specified, so the id is the entry that appeared, not the last listed."""
    client = _Client(
        save_result={"checkpoint_tag": "global_step10"},
        catalog=[{"checkpoint_id": "cp_older"}],
        added=[{"checkpoint_id": "cp_newer"}],
    )

    assert save_and_resolve_checkpoint_id(client, "job-1", "resumable") == "cp_newer"


def test_a_save_that_cannot_be_identified_is_an_error() -> None:
    client = _Client(save_result={}, catalog=[{"checkpoint_id": "cp_older"}])

    with pytest.raises(RuntimeError, match="cannot be identified"):
        save_and_resolve_checkpoint_id(client, "job-1", "resumable")


def test_one_iteration_is_a_forward_backward_then_a_step_at_the_requested_rate() -> None:
    """The loss is the forward-backward's, so it is read before the step that follows changes a parameter."""
    client = _Client(losses=[2.5])

    loss = HostedTrainingJob(client, "job-1").fwd_bwd_step(b"frame", learning_rate=1e-3)

    assert loss == 2.5
    assert client.calls == [("fwd-bwd", "job-1", b"frame"), ("step", "job-1", 1e-3)]


def test_a_forward_backward_that_reports_no_loss_is_an_error() -> None:
    class Silent(_Client):
        def forward_backward(self, job_id, body):
            return self._request({"metrics": {}})

    with pytest.raises(RuntimeError, match="no avg_loss"):
        HostedTrainingJob(Silent(), "job-1").fwd_bwd_step(b"frame", learning_rate=1e-3)


def test_a_job_releases_its_gpus_when_its_block_ends(monkeypatch) -> None:
    client = _Client()
    monkeypatch.setattr(hosted, "hosted_client", lambda connection: client)

    with HostedTransport(CONNECTION, _config()) as transport:
        with transport.job() as job:
            assert job.job_id == "job-1"
            assert client.cancelled == []
        assert client.cancelled == ["job-1"]
    assert client.closed


def test_a_job_releases_its_gpus_when_the_block_raises(monkeypatch) -> None:
    client = _Client()
    monkeypatch.setattr(hosted, "hosted_client", lambda connection: client)

    with pytest.raises(ZeroDivisionError):
        with HostedTransport(CONNECTION, _config()) as transport:
            with transport.job():
                raise ZeroDivisionError

    assert client.cancelled == ["job-1"]


def test_a_resumed_job_is_created_from_the_checkpoint_it_was_given(monkeypatch) -> None:
    client = _Client()
    monkeypatch.setattr(hosted, "hosted_client", lambda connection: client)

    with HostedTransport(CONNECTION, _config()) as transport:
        with transport.job(resume_from=Checkpoint(reference="cp_1", source_job="job-a")):
            pass

    body, hardware = client.created[0]
    assert body["source_checkpoint_info"] == {"checkpoint_id": "cp_1", "source_job_id": "job-a"}
    # The GPU family the config's own location names, in the spelling create-job takes.
    assert hardware == "H200"


def test_creating_a_job_before_the_transport_is_entered_is_an_error() -> None:
    with pytest.raises(RuntimeError, match="entered first"):
        with HostedTransport(CONNECTION, _config()).job():
            pass


def test_the_connection_does_not_render_its_token() -> None:
    assert "not-a-token" not in repr(CONNECTION)
