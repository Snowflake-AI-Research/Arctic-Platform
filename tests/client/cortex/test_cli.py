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
"""Every ``cortex`` command, driven through typer's CliRunner against fakes.

The control plane and the reattached `ArcticClient` are both patched out, so
these assert routing and output shape -- which command calls what, with which
arguments -- rather than anything on the wire.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from typer.testing import CliRunner

from arctic_platform.client.cortex import cli
from arctic_platform.client.cortex.jobs import Capacity

runner = CliRunner()

_CONNECTION = ["--host", "acct.example.com", "--pat", "tok", "--database", "DB", "--schema", "PUBLIC"]


def invoke(*args, **kwargs):
    """Run the CLI with a connection already supplied via flags."""
    return runner.invoke(cli.app, [*_CONNECTION, *args], **kwargs)


def output_json(result):
    assert result.exit_code == 0, result.output
    return json.loads(result.output)


@pytest.fixture(autouse=True)
def isolate(monkeypatch, tmp_path):
    """No ambient login/env: the flags under test must be the only input."""
    for name in ("CORTEX_CONFIG", "CORTEX_JOB", "CORTEX_HOST", "CORTEX_PAT", "SNOWFLAKE_HOST", "SNOWFLAKE_PAT"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("CORTEX_LOGIN_FILE", str(tmp_path / "login.json"))


@pytest.fixture
def control(monkeypatch):
    """The fake `CortexJobs` every command resolves to."""
    fake = MagicMock()
    fake.list.return_value = []
    fake.capacity.return_value = Capacity(has_reservation=True, reserved_gpus=8, in_use_gpus=2, available_gpus=6)
    monkeypatch.setattr(cli, "_jobs", lambda: fake)
    return fake


@pytest.fixture
def client(monkeypatch):
    """The fake reattached `ArcticClient` the data-plane commands drive."""
    fake = MagicMock()
    monkeypatch.setattr(cli, "_client", lambda job_id: fake)
    return fake


def write_json(tmp_path, name, payload):
    path = tmp_path / name
    path.write_text(json.dumps(payload))
    return str(path)


class TestConnectionCommands:
    def test_login_remembers_the_file(self, tmp_path):
        config = write_json(tmp_path, "c.json", {"host": "h.example.com", "pat": "p", "database": "D", "schema": "S"})
        result = runner.invoke(cli.app, ["login", "--config", config])
        assert result.exit_code == 0
        assert "Logged in" in result.output

    def test_login_rejects_an_unusable_file(self, tmp_path):
        config = write_json(tmp_path, "c.json", {"database": "D"})
        assert runner.invoke(cli.app, ["login", "--config", config]).exit_code != 0

    def test_logout_reports_when_not_logged_in(self):
        assert "Not logged in" in runner.invoke(cli.app, ["logout"]).output

    def test_logout_after_login(self, tmp_path):
        config = write_json(tmp_path, "c.json", {"host": "h.example.com", "pat": "p", "database": "D", "schema": "S"})
        runner.invoke(cli.app, ["login", "--config", config])
        assert "Logged out" in runner.invoke(cli.app, ["logout"]).output

    def test_config_masks_the_pat(self):
        settings = output_json(invoke("config"))
        assert settings["pat"] == "***"
        assert settings["host"] == "acct.example.com"

    def test_a_missing_connection_names_the_fix(self):
        result = runner.invoke(cli.app, ["config"])
        assert result.exit_code != 0
        assert "cortex login" in result.output


class TestJobCommands:
    def test_list_renders_a_table(self, control):
        control.list.return_value = [
            {"job_id": "J1", "status": "JOB_STATE_RUNNING", "created_at": "2026-06-13T18:49:54Z"}
        ]
        result = invoke("list")
        assert result.exit_code == 0
        assert "J1" in result.output and "running" in result.output

    def test_list_says_so_when_empty(self, control):
        assert "No jobs found" in invoke("list").output

    def test_list_passes_the_status_filter(self, control):
        invoke("list", "--status", "RUNNING")
        control.list.assert_called_once_with(status="RUNNING")

    def test_list_json_skips_the_table(self, control):
        control.list.return_value = [{"job_id": "J1"}]
        assert output_json(invoke("--json", "list")) == {"jobs": [{"job_id": "J1"}]}

    def test_list_puts_the_newest_nearest_the_prompt(self, control):
        control.list.return_value = [
            {"job_id": "NEW", "created_at": "2026-06-14T00:00:00Z"},
            {"job_id": "OLD", "created_at": "2026-06-13T00:00:00Z"},
        ]
        rows = output_json(invoke("--json", "list"))["jobs"]
        assert [job["job_id"] for job in rows] == ["OLD", "NEW"]

    def test_get_prints_the_job(self, control):
        control.get.return_value = {"job_id": "J1", "status": "running"}
        assert output_json(invoke("get", "J1"))["job_id"] == "J1"
        control.get.assert_called_once_with("J1")

    def test_cancel_confirms(self, control):
        assert "Cancelled" in invoke("cancel", "J1").output
        control.cancel.assert_called_once_with("J1")

    def test_wait_prints_the_running_job(self, control):
        control.wait.return_value = {"job_id": "J1", "status": "running"}
        assert output_json(invoke("wait", "J1"))["status"] == "running"

    def test_capacity_renders_a_table(self, control):
        result = invoke("capacity")
        assert result.exit_code == 0
        assert "6" in result.output

    def test_capacity_json(self, control):
        assert output_json(invoke("--json", "capacity"))["available_gpus"] == 6

    def test_checkpoints_renders_a_table(self, control):
        control.checkpoints.return_value = [{"checkpoint_id": "ck-1", "step": 100}]
        assert "ck-1" in invoke("checkpoints", "J1").output

    def test_checkpoints_says_so_when_empty(self, control):
        control.checkpoints.return_value = []
        assert "No checkpoints found" in invoke("checkpoints", "J1").output

    def test_job_id_comes_from_the_environment(self, control, monkeypatch):
        monkeypatch.setenv("CORTEX_JOB", "J-ENV")
        control.get.return_value = {}
        invoke("get")
        control.get.assert_called_once_with("J-ENV")

    def test_a_missing_job_id_is_an_error(self, control):
        result = invoke("get")
        assert result.exit_code != 0
        assert "CORTEX_JOB" in result.output


class TestSubmit:
    BODY = {"job_id": "J9", "sub_job_configs": [{"job_type": "training"}]}

    def test_posts_the_file(self, control, tmp_path):
        control.submit.return_value = {"job_id": "J9"}
        invoke("submit", write_json(tmp_path, "job.json", self.BODY))
        control.submit.assert_called_once_with(self.BODY)

    def test_reads_stdin(self, control):
        control.submit.return_value = {"job_id": "J9"}
        invoke("submit", "-", input=json.dumps(self.BODY))
        control.submit.assert_called_once_with(self.BODY)

    def test_job_id_flag_overrides_the_file(self, control, tmp_path):
        control.submit.return_value = {}
        invoke("submit", write_json(tmp_path, "job.json", self.BODY), "--job-id", "OVERRIDE")
        assert control.submit.call_args[0][0]["job_id"] == "OVERRIDE"

    def test_dry_run_sends_nothing(self, control, tmp_path):
        body = output_json(invoke("submit", write_json(tmp_path, "job.json", self.BODY), "--dry-run"))
        assert body == self.BODY
        control.submit.assert_not_called()

    def test_wait_polls_the_created_job(self, control, tmp_path):
        control.submit.return_value = {"job_id": "J9"}
        control.wait.return_value = {"job_id": "J9", "status": "running"}
        assert (
            output_json(invoke("submit", write_json(tmp_path, "j.json", self.BODY), "--wait"))["status"] == "running"
        )
        control.wait.assert_called_once_with("J9")

    def test_reports_invalid_json(self, control, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{nope")
        result = invoke("submit", str(path))
        assert result.exit_code != 0


class TestDataPlaneCommands:
    def test_fwd_bwd_sends_the_built_batch(self, control, client, tmp_path):
        client.fwd_bwd.return_value = {"loss": 1.25}
        spec = write_json(tmp_path, "fb.json", {"payload": {"kwargs": {"input_ids": [[1, 2]]}}})
        assert output_json(invoke("fwd-bwd", "J1", spec))["result"] == {"loss": 1.25}
        assert "input_ids" in client.fwd_bwd.call_args[0][0]["kwargs"]

    def test_step_forwards_the_learning_rate(self, control, client):
        client.step.return_value = {"ok": True}
        invoke("step", "J1", "--lr", "0.0001")
        client.step.assert_called_once_with(learning_rate=0.0001)

    def test_step_omits_an_unset_learning_rate(self, control, client):
        client.step.return_value = {}
        invoke("step", "J1")
        client.step.assert_called_once_with(learning_rate=None)

    def test_generate_passes_the_validated_spec(self, control, client, tmp_path):
        client.generate.return_value = ["hello there"]
        spec = write_json(tmp_path, "g.json", {"payload": {"prompts": ["hi"], "sampling_params": {"max_tokens": 4}}})
        assert output_json(invoke("generate", "J1", spec))["results"] == ["hello there"]
        assert client.generate.call_args.kwargs["prompts"] == ["hi"]

    def test_generate_rejects_an_empty_prompt_list(self, control, client, tmp_path):
        spec = write_json(tmp_path, "g.json", {"payload": {"prompts": []}})
        assert invoke("generate", "J1", spec).exit_code != 0

    def test_load_polls_the_request(self, control):
        control.load.return_value = "R1"
        control.poll_request.return_value = {"global_step": 100}
        out = output_json(invoke("load", "J1", "ck-1", "--target-sub-job-id", "J1:training:0"))
        assert out["request_id"] == "R1" and out["result"] == {"global_step": 100}
        control.load.assert_called_once_with("J1", "ck-1", source_job_id=None, target_sub_job_id="J1:training:0")

    def test_load_no_poll_returns_the_request_id(self, control):
        control.load.return_value = "R1"
        assert "result" not in output_json(invoke("load", "J1", "ck-1", "--no-poll"))
        control.poll_request.assert_not_called()

    def test_sync_weights_uses_session_defaults(self, control, client):
        client.sync_weights.return_value = {}
        invoke("sync-weights", "J1")
        assert client.sync_weights.call_args.kwargs == {
            "weight_format": None,
            "source_sub_job_id": None,
            "target_sub_job_ids": None,
        }

    def test_sync_weights_forwards_explicit_routing(self, control, client):
        client.sync_weights.return_value = {}
        invoke(
            "sync-weights",
            "J1",
            "--source-sub-job-id",
            "J1:training:0",
            "--target-sub-job-id",
            "J1:sampling:0",
            "--target-sub-job-id",
            "J1:sampling:1",
            "--weight-format",
            "lora",
        )
        kwargs = client.sync_weights.call_args.kwargs
        assert kwargs["target_sub_job_ids"] == ["J1:sampling:0", "J1:sampling:1"]
        assert kwargs["weight_format"] == "lora"


class TestLogCommands:
    def test_logs_streams_plainly_when_asked(self, control, monkeypatch):
        reader = MagicMock()
        reader.stream_logs.return_value = iter([{"msg": "line one"}, {"msg": "line two"}])
        monkeypatch.setattr(cli, "CortexLogs", lambda jobs, poll_interval=1.0: reader)

        result = invoke("logs", "J1", "--plain", "--no-follow")
        assert result.exit_code == 0
        assert "line one" in result.output and "line two" in result.output
        assert reader.stream_logs.call_args.kwargs["follow"] is False

    def test_logs_passes_the_sub_job(self, control, monkeypatch):
        reader = MagicMock()
        reader.stream_logs.return_value = iter([])
        monkeypatch.setattr(cli, "CortexLogs", lambda jobs, poll_interval=1.0: reader)
        invoke("logs", "J1", "--plain", "--sub-job", "J1:training:0")
        assert reader.stream_logs.call_args.kwargs["sub_job_id"] == "J1:training:0"

    def test_download_logs_reports_saved_paths(self, control, monkeypatch, tmp_path):
        reader = MagicMock()
        reader.download_execution_logs.return_value = [{"saved_path": str(tmp_path / "a.log")}]
        monkeypatch.setattr(cli, "CortexLogs", lambda jobs, **kw: reader)

        result = invoke("download-logs", "J1", "--output-dir", str(tmp_path))
        assert "a.log" in result.output
        reader.download_execution_logs.assert_called_once_with("J1", str(tmp_path))

    def test_download_logs_says_so_when_empty(self, control, monkeypatch):
        reader = MagicMock()
        reader.download_execution_logs.return_value = []
        monkeypatch.setattr(cli, "CortexLogs", lambda jobs, **kw: reader)
        assert "No log files found" in invoke("download-logs", "J1").output
