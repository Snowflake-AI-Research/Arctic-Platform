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
"""Live tail paging and the archived-log walk through a mocked S3."""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from arctic_platform.client.cortex.logs import CortexLogs
from arctic_platform.client.cortex.logs import _parse_s3_stage_credentials


@pytest.fixture
def reader(jobs):
    def build(replies=None):
        return CortexLogs(jobs(replies), poll_interval=0.0)

    return build


class TestTailRequests:
    def test_tail_logs_posts_an_operation(self, reader, prefix):
        logs = reader([{"entries": []}])
        logs.tail_logs("J1", cursor="c1", max_lines=500, sub_job_id="J1:training:0")
        call = logs.session.last
        assert (call["method"], call["url"]) == ("POST", f"{prefix}/J1/operation")
        assert call["json"] == {
            "operation_type": "tail-logs",
            "payload": {"cursor": "c1", "max_lines": 500},
            "sub_job_id": "J1:training:0",
        }

    def test_tail_logs_omits_unset_fields(self, reader):
        logs = reader([{"entries": []}])
        logs.tail_logs("J1")
        assert logs.session.last["json"] == {"operation_type": "tail-logs", "payload": {}}

    def test_tail_events_uses_its_own_operation_type(self, reader):
        logs = reader([{"events": []}])
        logs.tail_events("J1", sub_job_type="training")
        body = logs.session.last["json"]
        assert body["operation_type"] == "tail-events"
        assert body["sub_job_type"] == "training"


class TestStreaming:
    def test_drains_pages_and_advances_the_cursor(self, reader):
        logs = reader(
            [
                {"entries": [{"msg": "a"}], "next_cursor": "c1"},
                {"entries": [{"msg": "b"}], "next_cursor": "c2"},
                {"entries": [], "eof": True},
            ]
        )
        assert list(logs.stream_logs("J1", follow=False)) == [{"msg": "a"}, {"msg": "b"}]
        assert [call["json"]["payload"].get("cursor") for call in logs.session.calls] == [None, "c1", "c2"]

    def test_stops_at_eof_when_not_following(self, reader):
        logs = reader([{"entries": [], "eof": True}])
        assert list(logs.stream_logs("J1", follow=False)) == []
        assert len(logs.session.calls) == 1

    def test_stream_events_reads_the_events_key(self, reader):
        logs = reader([{"events": [{"kind": "placed"}], "next_cursor": "c1"}, {"events": [], "eof": True}])
        assert list(logs.stream_events("J1", follow=False)) == [{"kind": "placed"}]


class TestStageCredentials:
    def _stage(self, **overrides):
        return {
            "locationType": "S3",
            "location": "s3://bucket/some/prefix/",
            "region": "us-west-2",
            "creds": {"AWS_KEY_ID": "k", "AWS_SECRET_KEY": "s", "AWS_TOKEN": "t"},
            **overrides,
        }

    def test_splits_bucket_from_prefix(self):
        creds = _parse_s3_stage_credentials(self._stage())
        assert (creds["bucket"], creds["prefix"]) == ("bucket", "some/prefix")

    def test_accepts_a_json_string(self):
        assert _parse_s3_stage_credentials(json.dumps(self._stage()))["access_key_id"] == "k"

    def test_rejects_a_non_s3_stage(self):
        with pytest.raises(NotImplementedError, match="only supports S3"):
            _parse_s3_stage_credentials(self._stage(locationType="AZURE"))

    def test_rejects_missing_aws_fields(self):
        with pytest.raises(ValueError, match="missing AWS field"):
            _parse_s3_stage_credentials(self._stage(creds={"AWS_KEY_ID": "k"}))

    def test_rejects_a_location_without_a_bucket(self):
        with pytest.raises(ValueError, match="missing bucket"):
            _parse_s3_stage_credentials(self._stage(location=""))


def _fake_boto3(objects: dict[str, bytes]) -> MagicMock:
    """A boto3 module whose S3 client serves ``objects`` (key -> body)."""
    client = MagicMock()
    client.get_paginator.return_value.paginate.return_value = [
        {"Contents": [{"Key": key} for key in objects]},
    ]
    client.get_object.side_effect = lambda Bucket, Key: {"Body": SimpleNamespace(read=lambda: objects[Key])}
    module = MagicMock()
    module.client.return_value = client
    return module


class TestFetchExecutionLogs:
    STAGE = {
        "locationType": "S3",
        "location": "s3://bucket/run/",
        "region": "us-west-2",
        "creds": {"AWS_KEY_ID": "k", "AWS_SECRET_KEY": "s", "AWS_TOKEN": "t"},
    }

    def _reader(self, reader, objects):
        return (
            reader(
                [
                    {"experiment_name": "exp", "experiment_run_name": "run1"},
                    {"data": [[json.dumps(self.STAGE)]]},
                ]
            ),
            objects,
        )

    def test_walks_the_stage_and_groups_by_sub_job(self, reader, monkeypatch):
        objects = {
            "run/_logs/J1:training:0/server.log": b"train output",
            "run/_logs/J1:sampling:0/server.log": b"sample output",
            "run/metrics.json": b"not a log",  # outside _logs/: skipped
        }
        logs, _ = self._reader(reader, objects)
        monkeypatch.setitem(sys.modules, "boto3", _fake_boto3(objects))

        found = logs.fetch_execution_logs("J1")
        assert [(f["sub_job_id"], f["filename"]) for f in found] == [
            ("J1:training:0", "server.log"),
            ("J1:sampling:0", "server.log"),
        ]
        assert found[0]["content"] == "train output"
        assert found[0]["s3_uri"] == "s3://bucket/run/_logs/J1:training:0/server.log"

    def test_asks_snowflake_for_stage_credentials(self, reader, monkeypatch, prefix):
        logs, objects = self._reader(reader, {})
        monkeypatch.setitem(sys.modules, "boto3", _fake_boto3(objects))
        logs.fetch_execution_logs("J1")

        experiment_run, sql = logs.session.calls
        assert experiment_run["url"] == f"{prefix}/J1/experiment-run"
        assert sql["url"] == "https://acct.example.com/api/v2/statements"
        assert "SYSTEM$GET_VSTAGE_WRITE_CREDS('snow://experiment/exp/versions/run1/')" in sql["json"]["statement"]

    def test_reports_a_broken_experiment_run(self, reader):
        logs = reader([{"experiment_name": "exp"}])
        with pytest.raises(ValueError, match="missing field: experiment_run_name"):
            logs.fetch_execution_logs("J1")

    def test_download_writes_files_under_their_sub_job(self, reader, monkeypatch, tmp_path):
        objects = {"run/_logs/J1:training:0/server.log": b"hello"}
        logs, _ = self._reader(reader, objects)
        monkeypatch.setitem(sys.modules, "boto3", _fake_boto3(objects))

        saved = logs.download_execution_logs("J1", tmp_path)
        written = tmp_path / "J1:training:0" / "server.log"
        assert written.read_text() == "hello"
        assert saved[0]["saved_path"] == str(written)
        assert "content" not in saved[0]  # the payload is on disk, not in the summary
