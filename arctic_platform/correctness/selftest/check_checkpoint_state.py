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

"""Static correctness selftests: checkpoint state."""

from __future__ import annotations

import contextlib
import copy
import hashlib
import importlib
import json
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import torch

from arctic_platform.client.requests import export_checkpoint_request
from arctic_platform.client.transports.cortex import _route
from arctic_platform.correctness.checks.fwd_bwd_step import _compare_arm
from arctic_platform.correctness.checks.optimizer_state import compare_optimizer_artifacts
from arctic_platform.correctness.checks.optimizer_state import max_pairwise_optimizer_delta
from arctic_platform.correctness.checks.optimizer_state import summarize_optimizer_artifacts
from arctic_platform.correctness.harness import inference_export as exporter
from arctic_platform.correctness.harness.config import TestTolerance as Tolerance
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.runner import OPTIMIZER_LEARNING_RATE

# Allocation-local native checkpoint bridge. No DSS process, API or GPU is started.


@pytest.mark.parametrize("fused", [None, False, True])
def test_native_checkpoint_bridge_training_config_round_trips_optional_optimizer_defaults(fused):
    from dss.ray_dss.models import JobConfig

    from arctic_platform.correctness.harness.gateway import _training_config

    optimizer = {"lr": 1e-5}
    if fused is not None:
        optimizer["fused"] = fused
    config = JobConfig.model_validate(
        {
            "job_id": "generated",
            "model_name": "control-model",
            "model_post_init": ["control-hook"],
            "training_config": {
                "optimizer": optimizer,
                "ds_config": {"checkpoint": {"tag_validation": "Fail"}},
            },
        }
    )
    original = config.model_dump(mode="json", by_alias=True)

    raw = _training_config(config)
    assert "job_id" not in raw
    assert raw["model_post_init"] == ["control-hook"]
    assert raw["training_config"]["ds_config"] == {"checkpoint": {"tag_validation": "Fail"}}
    if fused is None:
        assert "fused" not in raw["training_config"]["optimizer"]
    else:
        assert raw["training_config"]["optimizer"]["fused"] is fused
    restored = JobConfig.model_validate({**raw, "job_id": config.job_id})
    assert restored.model_dump(mode="json", by_alias=True) == original
    assert config.model_dump(mode="json", by_alias=True) == original


# Checkpoint resume.


# Cortex checkpoint export.


_cortex_checkpoint_export_JOB_ID = "0abf38e6-80f6-49cc-820d-883c8c451df4"

_cortex_checkpoint_export_CP_ID = "cp_dc1ee034-d625-4989-8164-092903a8d150"

_cortex_checkpoint_export_SECRET = "DO-NOT-LOG-PRESIGNED-SECRET"


class _cortex_checkpoint_export_Response:

    def __init__(self, payload=b"checkpoint", *, status=200, headers=None, chunks=None):
        self.status_code = status
        self.headers = {} if headers is None else headers
        self.closed = False
        self.raw_chunks = list(chunks) if chunks is not None else [payload[:3], payload[3:]]
        self.read_calls = []
        self.timeouts = []
        self.raw = SimpleNamespace(
            read1=self.read1,
            chunked=False,
            connection=SimpleNamespace(sock=SimpleNamespace(settimeout=self.timeouts.append)),
        )

    def read1(self, size, *, decode_content):
        assert 0 < size <= exporter._CHUNK_BYTES
        assert decode_content is False
        self.read_calls.append(size)
        if not self.raw_chunks:
            return b""
        chunk = self.raw_chunks.pop(0)
        if isinstance(chunk, BaseException):
            raise chunk
        if len(chunk) > size:
            self.raw_chunks.insert(0, chunk[size:])
        return chunk[:size]

    def close(self):
        self.closed = True


class _cortex_checkpoint_export_DownloadSession:

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.trust_env = True
        self.auth = "unexpected-auth"
        self.closed = False

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        assert self.trust_env is False
        assert self.auth is None
        assert kwargs["stream"] is True
        assert kwargs["allow_redirects"] is False
        assert kwargs["headers"] == {"Accept-Encoding": "identity"}
        assert 0 < kwargs["timeout"][0] <= 30
        assert 0 < kwargs["timeout"][1] <= 60
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True


def _cortex_checkpoint_export_fixture(monkeypatch, *, dp_size=2, prefix="bf16_", reply=None, responses=None):
    names = [
        *(f"global_step10/{prefix}zero_pp_rank_{rank}_mp_rank_00_optim_states.pt" for rank in range(dp_size)),
        "global_step10/mp_rank_00_model_states.pt",
        "metadata.json",
    ]
    if reply is None:
        reply = {
            "checkpoint_id": _cortex_checkpoint_export_CP_ID,
            "expires_at": "2026-10-14T08:00:00Z",
            "files": [
                {
                    "filename": name,
                    "size_bytes": "10",
                    "url": f"https://storage.example/file{index}?sig={_cortex_checkpoint_export_SECRET}",
                }
                for index, name in enumerate(names)
            ],
        }
    posted = []

    def call(request):
        assert request == export_checkpoint_request(_cortex_checkpoint_export_JOB_ID, _cortex_checkpoint_export_CP_ID)
        route = _route(request.op)
        url, sends = route.prepare("https://cortex.example/jobs", request)
        assert sends == [{}]
        posted.append(((route.method, url), sends[0]))
        return copy.deepcopy(reply)

    transport = SimpleNamespace(job_id=_cortex_checkpoint_export_JOB_ID, call=call)
    transport.session = SimpleNamespace(headers={"Authorization": "Bearer DO-NOT-LEAK-PAT"})
    client = SimpleNamespace(transport=transport)
    if responses is None:
        responses = [_cortex_checkpoint_export_Response() for _ in names]
    session = _cortex_checkpoint_export_DownloadSession(responses)
    monkeypatch.setattr(exporter.requests, "Session", lambda: session)
    return (client, reply, posted, session, responses)


def _cortex_checkpoint_export_export(client, tmp_path, **kwargs):
    return exporter.export_checkpoint(
        client,
        _cortex_checkpoint_export_JOB_ID,
        _cortex_checkpoint_export_CP_ID,
        tmp_path,
        expected_dp_world_size=2,
        **kwargs,
    )


@pytest.mark.parametrize("unsafe", ["/tmp", "/code", "/data-fast-shadow", "/data-fast", "symlink"])
def test_cortex_checkpoint_export_export_requires_data_fast_before_public_post(tmp_path, monkeypatch, unsafe):
    destination = Path(unsafe)
    if unsafe == "symlink":
        destination = tmp_path / "escape"
        destination.symlink_to("/tmp", target_is_directory=True)
        destination /= "output"
    client, _, posted, session, _ = _cortex_checkpoint_export_fixture(monkeypatch)
    monkeypatch.setattr(client.transport, "call", lambda request: pytest.fail("unsafe output reached public POST"))
    with pytest.raises(exporter.CheckpointExportError, match="/data-fast|symlink"):
        _cortex_checkpoint_export_export(client, destination)
    assert posted == []
    assert session.calls == []
    assert not list(tmp_path.glob("checkpoint-export-*"))


@pytest.mark.parametrize("prefix", ["", "bf16_"])
def test_cortex_checkpoint_export_export_preserves_public_post_and_streams_all_files_to_unique_receipted_root(
    tmp_path, monkeypatch, prefix
):
    client, reply, posted, session, responses = _cortex_checkpoint_export_fixture(monkeypatch, prefix=prefix)
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("existing unrelated output")
    root = _cortex_checkpoint_export_export(client, tmp_path)
    assert root.parent == tmp_path
    assert root != tmp_path
    assert sentinel.read_text() == "existing unrelated output"
    assert posted == [
        (
            (
                "POST",
                f"https://cortex.example/jobs/{_cortex_checkpoint_export_JOB_ID}/checkpoints/{_cortex_checkpoint_export_CP_ID}:export",
            ),
            {},
        )
    ]
    assert len(session.calls) == 4
    assert all((response.closed for response in responses))
    assert session.closed
    receipt_text = (root / "export_receipt.json").read_text()
    receipt = json.loads(receipt_text)
    assert receipt["complete"] is True
    assert receipt["source_job_id"] == _cortex_checkpoint_export_JOB_ID
    assert receipt["checkpoint_id"] == _cortex_checkpoint_export_CP_ID
    assert receipt["dp_world_size"] == 2
    assert receipt["tag"] == "global_step10"
    assert receipt["total_bytes"] == 40
    assert {entry["path"] for entry in receipt["files"]} == {entry["filename"] for entry in reply["files"]}
    for item in receipt["files"]:
        assert (root / item["path"]).read_bytes() == b"checkpoint"
        assert item["sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
        assert item["size_bytes"] == 10
        assert set(item) == {"path", "size_bytes", "sha256"}
    assert _cortex_checkpoint_export_SECRET not in receipt_text
    assert "url" not in receipt_text
    assert "DO-NOT-LEAK-PAT" not in receipt_text
    assert all(("Authorization" not in kwargs["headers"] for _, kwargs in session.calls))


def test_cortex_checkpoint_export_each_export_allocates_new_directory_without_reusing_completed_snapshot(
    tmp_path, monkeypatch
):
    client, *_ = _cortex_checkpoint_export_fixture(monkeypatch)
    first = _cortex_checkpoint_export_export(client, tmp_path)
    client, *_ = _cortex_checkpoint_export_fixture(monkeypatch)
    second = _cortex_checkpoint_export_export(client, tmp_path)
    assert first != second
    assert (first / "export_receipt.json").is_file()
    assert (second / "export_receipt.json").is_file()


def test_cortex_checkpoint_export_valid_export_creates_parent_only_after_manifest_validation(tmp_path, monkeypatch):
    destination = tmp_path / "nested" / "exports"
    client, reply, posted, _, _ = _cortex_checkpoint_export_fixture(monkeypatch)
    original_call = client.transport.call

    def call(request):
        assert not destination.exists()
        return original_call(request)

    monkeypatch.setattr(client.transport, "call", call)
    root = _cortex_checkpoint_export_export(client, destination)
    assert root.parent == destination
    assert (root / "export_receipt.json").is_file()
    assert len(posted) == 1
    assert len(reply["files"]) == 4


def test_cortex_checkpoint_export_observed_four_rank_six_file_export_layout(tmp_path, monkeypatch):
    client, reply, _, session, _ = _cortex_checkpoint_export_fixture(monkeypatch, dp_size=4)
    root = exporter.export_checkpoint(
        client, _cortex_checkpoint_export_JOB_ID, _cortex_checkpoint_export_CP_ID, tmp_path, expected_dp_world_size=4
    )
    receipt = json.loads((root / "export_receipt.json").read_text())
    assert receipt["dp_world_size"] == 4
    assert len(session.calls) == len(receipt["files"]) == len(reply["files"]) == 6
    assert len(list((root / "global_step10").glob("*_optim_states.pt"))) == 4


@pytest.mark.parametrize(
    "name",
    [
        "/metadata.json",
        "../metadata.json",
        "global_step10/../../metadata.json",
        "./metadata.json",
        "global_step10//mp_rank_00_model_states.pt",
        "global_step10\\mp_rank_00_model_states.pt",
        "C:/metadata.json",
        "global_step10/%2e%2e/metadata.json",
        "global_step10/\x00file",
        "global_step10/file\n",
    ],
)
def test_cortex_checkpoint_export_unsafe_filenames_rejected_before_directory_or_download(tmp_path, monkeypatch, name):
    _, reply, *_ = _cortex_checkpoint_export_fixture(monkeypatch)
    reply["files"][0]["filename"] = name
    client, _, _, session, _ = _cortex_checkpoint_export_fixture(monkeypatch, reply=reply)
    with pytest.raises(exporter.CheckpointExportError, match="unsafe export filename"):
        _cortex_checkpoint_export_export(client, tmp_path)
    assert not session.calls
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "duplicate",
        "rank_gap",
        "rank_alias",
        "rank_leading_zero",
        "mixed_prefix",
        "different_tag",
        "multiple_models",
        "no_metadata",
        "tensor_parallel",
        "nested_tag",
        "wrong_cp",
        "symlink",
    ],
)
def test_cortex_checkpoint_export_incomplete_or_unsupported_layout_rejected_before_download(
    tmp_path, monkeypatch, mutation
):
    _, reply, *_ = _cortex_checkpoint_export_fixture(monkeypatch)
    files = reply["files"]
    if mutation == "missing":
        files.pop()
    elif mutation == "extra":
        files.append({"filename": "extra", "size_bytes": 1, "url": "https://storage.example/file"})
    elif mutation == "duplicate":
        files[1] = copy.deepcopy(files[0])
    elif mutation == "rank_gap":
        files[1]["filename"] = files[1]["filename"].replace("rank_1_", "rank_2_")
    elif mutation == "rank_alias":
        files[1]["filename"] = files[0]["filename"].replace("bf16_", "")
    elif mutation == "rank_leading_zero":
        files[1]["filename"] = files[1]["filename"].replace("rank_1_", "rank_01_")
    elif mutation == "mixed_prefix":
        files[1]["filename"] = files[1]["filename"].replace("bf16_", "")
    elif mutation == "different_tag":
        files[1]["filename"] = files[1]["filename"].replace("step10", "step11")
    elif mutation == "multiple_models":
        files[1]["filename"] = "global_step11/mp_rank_00_model_states.pt"
    elif mutation == "no_metadata":
        files[-1]["filename"] = "other.json"
    elif mutation == "tensor_parallel":
        files[1]["filename"] = files[1]["filename"].replace("mp_rank_00", "mp_rank_01")
    elif mutation == "nested_tag":
        files[1]["filename"] = "nested/" + files[1]["filename"]
    elif mutation == "wrong_cp":
        reply["checkpoint_id"] = "cp-other"
    elif mutation == "symlink":
        files[0]["type"] = "symlink"
    client, _, _, session, _ = _cortex_checkpoint_export_fixture(monkeypatch, reply=reply)
    with pytest.raises(exporter.CheckpointExportError):
        _cortex_checkpoint_export_export(client, tmp_path)
    assert not session.calls
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("size", [0, -1, True, None, 1.5, "1.5", "-1", "+10", " 10", "10 ", ""])
def test_cortex_checkpoint_export_invalid_declared_sizes_rejected(tmp_path, monkeypatch, size):
    _, reply, *_ = _cortex_checkpoint_export_fixture(monkeypatch)
    reply["files"][0]["size_bytes"] = size
    client, _, _, session, _ = _cortex_checkpoint_export_fixture(monkeypatch, reply=reply)
    with pytest.raises(exporter.CheckpointExportError, match="declared file size"):
        _cortex_checkpoint_export_export(client, tmp_path)
    assert not session.calls
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "http://storage.example/file",
        "file:///etc/passwd",
        "https://user:password@storage.example/file",
        "https://storage.example/file#fragment",
        "https://storage.example:invalid/file",
        "https:///file",
        "https://storage.example/file\n",
    ],
)
def test_cortex_checkpoint_export_invalid_urls_rejected_without_logging_values(tmp_path, monkeypatch, url):
    _, reply, *_ = _cortex_checkpoint_export_fixture(monkeypatch)
    reply["files"][0]["url"] = url
    client, _, _, session, _ = _cortex_checkpoint_export_fixture(monkeypatch, reply=reply)
    with pytest.raises(exporter.CheckpointExportError, match="HTTPS URL") as error:
        _cortex_checkpoint_export_export(client, tmp_path)
    assert "password" not in str(error.value)
    assert not session.calls
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("alias", ["download_url", "presigned_url"])
def test_cortex_checkpoint_export_undocumented_url_alias_not_guessed(tmp_path, monkeypatch, alias):
    _, reply, *_ = _cortex_checkpoint_export_fixture(monkeypatch)
    reply["files"][0][alias] = reply["files"][0].pop("url")
    client, _, _, session, _ = _cortex_checkpoint_export_fixture(monkeypatch, reply=reply)
    with pytest.raises(exporter.CheckpointExportError, match="HTTPS URL"):
        _cortex_checkpoint_export_export(client, tmp_path)
    assert not session.calls


def test_cortex_checkpoint_export_byte_budget_checked_before_download_and_exact_limit_accepted(tmp_path, monkeypatch):
    client, _, _, session, _ = _cortex_checkpoint_export_fixture(monkeypatch)
    with pytest.raises(exporter.CheckpointExportError, match="byte budget"):
        _cortex_checkpoint_export_export(client, tmp_path, max_total_bytes=39)
    assert not session.calls
    assert not list(tmp_path.iterdir())
    root = _cortex_checkpoint_export_export(client, tmp_path, max_total_bytes=40)
    assert (root / "export_receipt.json").is_file()


@pytest.mark.parametrize(
    "response",
    [
        _cortex_checkpoint_export_Response(b"short"),
        _cortex_checkpoint_export_Response(b"checkpoint-extra"),
        _cortex_checkpoint_export_Response(status=206),
        _cortex_checkpoint_export_Response(status=403),
        _cortex_checkpoint_export_Response(headers={"Content-Length": "11"}),
        _cortex_checkpoint_export_Response(headers={"Content-Length": "invalid"}),
        _cortex_checkpoint_export_Response(headers={"Content-Encoding": "gzip"}),
        _cortex_checkpoint_export_Response(
            chunks=[b"part", requests.ConnectionError(f"broken signed URL ?sig={_cortex_checkpoint_export_SECRET}")]
        ),
        _cortex_checkpoint_export_Response(
            chunks=[requests.Timeout(f"request ?sig={_cortex_checkpoint_export_SECRET}")]
        ),
    ],
)
def test_cortex_checkpoint_export_failed_stream_never_leaves_receipt_or_partial_snapshot(
    tmp_path, monkeypatch, response
):
    responses = [_cortex_checkpoint_export_Response(), response]
    client, _, _, session, _ = _cortex_checkpoint_export_fixture(monkeypatch, responses=responses)
    sentinel = tmp_path / "caller-owned"
    sentinel.mkdir()
    with pytest.raises(exporter.CheckpointExportError) as error:
        _cortex_checkpoint_export_export(client, tmp_path)
    assert _cortex_checkpoint_export_SECRET not in "".join(traceback.format_exception(error.value))
    assert response.closed
    assert session.closed
    assert list(tmp_path.iterdir()) == [sentinel]


def test_cortex_checkpoint_export_content_length_matches_and_streamed_size_still_validated(tmp_path, monkeypatch):
    responses = [_cortex_checkpoint_export_Response(headers={"Content-Length": "10"}) for _ in range(4)]
    client, *_ = _cortex_checkpoint_export_fixture(monkeypatch, responses=responses)
    assert (_cortex_checkpoint_export_export(client, tmp_path) / "export_receipt.json").is_file()
    assert all((len(response.read_calls) == 2 for response in responses))


def test_cortex_checkpoint_export_presigned_https_cross_host_redirects_followed_without_auth(tmp_path, monkeypatch):
    redirect = _cortex_checkpoint_export_Response(
        status=307, headers={"Location": f"https://other.example/file?sig={_cortex_checkpoint_export_SECRET}"}
    )
    client, _, _, session, _ = _cortex_checkpoint_export_fixture(
        monkeypatch, responses=[redirect, *[_cortex_checkpoint_export_Response() for _ in range(4)]]
    )
    root = _cortex_checkpoint_export_export(client, tmp_path)
    assert redirect.closed
    assert len(session.calls) == 5
    assert session.calls[1][0] == f"https://other.example/file?sig={_cortex_checkpoint_export_SECRET}"
    assert (root / "export_receipt.json").is_file()


@pytest.mark.parametrize(
    "location", [None, "", f"http://other.example/file?sig={_cortex_checkpoint_export_SECRET}", "file:///etc/passwd"]
)
def test_cortex_checkpoint_export_invalid_redirect_rejected_before_following(tmp_path, monkeypatch, location):
    redirect = _cortex_checkpoint_export_Response(status=302, headers={"Location": location})
    client, _, _, session, _ = _cortex_checkpoint_export_fixture(monkeypatch, responses=[redirect])
    with pytest.raises(exporter.CheckpointExportError) as error:
        _cortex_checkpoint_export_export(client, tmp_path)
    assert _cortex_checkpoint_export_SECRET not in "".join(traceback.format_exception(error.value))
    assert redirect.closed
    assert len(session.calls) == 1
    assert not list(tmp_path.iterdir())


def test_cortex_checkpoint_export_redirect_loop_is_bounded(tmp_path, monkeypatch):
    redirects = [_cortex_checkpoint_export_Response(status=302, headers={"Location": "/same"}) for _ in range(6)]
    client, _, _, session, _ = _cortex_checkpoint_export_fixture(monkeypatch, responses=redirects)
    with pytest.raises(exporter.CheckpointExportError, match="redirect limit"):
        _cortex_checkpoint_export_export(client, tmp_path)
    assert len(session.calls) == 6
    assert all((response.closed for response in redirects))
    assert not list(tmp_path.iterdir())


def test_cortex_checkpoint_export_http_and_post_exception_messages_do_not_leak_secrets(tmp_path, monkeypatch):
    client, _, _, session, _ = _cortex_checkpoint_export_fixture(
        monkeypatch, responses=[requests.ConnectionError(_cortex_checkpoint_export_SECRET)]
    )
    with pytest.raises(exporter.CheckpointExportError) as error:
        _cortex_checkpoint_export_export(client, tmp_path)
    assert _cortex_checkpoint_export_SECRET not in "".join(traceback.format_exception(error.value))
    assert session.closed
    assert not list(tmp_path.iterdir())
    client.transport.call = lambda request: (_ for _ in ()).throw(RuntimeError(_cortex_checkpoint_export_SECRET))
    with pytest.raises(exporter.CheckpointExportError) as error:
        _cortex_checkpoint_export_export(client, tmp_path)
    assert _cortex_checkpoint_export_SECRET not in "".join(traceback.format_exception(error.value))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("timeout", [0, -1, True, None, float("inf"), float("nan")])
def test_cortex_checkpoint_export_invalid_timeouts_rejected_before_post(tmp_path, monkeypatch, timeout):
    client, _, posted, session, _ = _cortex_checkpoint_export_fixture(monkeypatch)
    with pytest.raises(exporter.CheckpointExportError, match="timeout"):
        _cortex_checkpoint_export_export(client, tmp_path, timeout_s=timeout)
    assert not posted
    assert not session.calls


def test_cortex_checkpoint_export_deadline_exceeded_mid_stream_cleans_partial_files(tmp_path, monkeypatch):
    client, _, _, session, responses = _cortex_checkpoint_export_fixture(monkeypatch)
    ticks = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(exporter.time, "monotonic", lambda: next(ticks))
    with pytest.raises(exporter.CheckpointExportError, match="deadline"):
        _cortex_checkpoint_export_export(client, tmp_path, timeout_s=1)
    assert responses[0].closed
    assert session.closed
    assert not list(tmp_path.iterdir())


def test_cortex_checkpoint_export_raw_reads_return_short_chunks_and_refresh_remaining_timeout(tmp_path, monkeypatch):
    clock = [0.0]
    responses = [
        _cortex_checkpoint_export_Response(chunks=[bytes([byte]) for byte in b"checkpoint"]) for _ in range(4)
    ]
    for response in responses:
        read1 = response.raw.read1

        def read(size, *, decode_content, read1=read1):
            clock[0] += 0.25
            return read1(size, decode_content=decode_content)

        response.raw.read1 = read
    monkeypatch.setattr(exporter.time, "monotonic", lambda: clock[0])
    client, _, _, _, _ = _cortex_checkpoint_export_fixture(monkeypatch, responses=responses)
    root = _cortex_checkpoint_export_export(client, tmp_path, timeout_s=20)
    assert (root / "export_receipt.json").is_file()
    timeouts = [timeout for response in responses for timeout in response.timeouts]
    assert len(timeouts) == 44
    assert timeouts == sorted(timeouts, reverse=True)
    assert timeouts[0] == 20
    assert timeouts[-1] == 9.25
    assert all((len(response.read_calls) == 11 for response in responses))


@pytest.mark.parametrize("connection", ["closed", "released"])
def test_cortex_checkpoint_export_connection_close_keeps_bounded_response_socket_available(
    tmp_path, monkeypatch, connection
):
    responses = [
        _cortex_checkpoint_export_Response(headers={"Connection": "close", "Content-Length": "10"}) for _ in range(4)
    ]
    for response in responses:
        sock = response.raw.connection.sock
        response.raw._fp = SimpleNamespace(fp=SimpleNamespace(raw=SimpleNamespace(_sock=sock)))
        if connection == "closed":
            response.raw.connection.sock = None
        else:
            response.raw.connection = None
    client, *_ = _cortex_checkpoint_export_fixture(monkeypatch, responses=responses)
    assert (_cortex_checkpoint_export_export(client, tmp_path) / "export_receipt.json").is_file()
    assert all((len(response.timeouts) == len(response.read_calls) == 2 for response in responses))


def test_cortex_checkpoint_export_raw_read_requests_are_capped_even_for_large_files(tmp_path, monkeypatch):
    payload = b"x" * (exporter._CHUNK_BYTES + 7)
    responses = [_cortex_checkpoint_export_Response(payload) for _ in range(4)]
    client, reply, *_ = _cortex_checkpoint_export_fixture(monkeypatch, responses=responses)
    for item in reply["files"]:
        item["size_bytes"] = len(payload)
    root = _cortex_checkpoint_export_export(client, tmp_path)
    for response in responses:
        assert max(response.read_calls) == exporter._CHUNK_BYTES
        assert response.read_calls[-1] == 1
    assert all(
        (
            item["size_bytes"] == len(payload)
            for item in json.loads((root / "export_receipt.json").read_text())["files"]
        )
    )


def test_cortex_checkpoint_export_slow_trickle_stops_at_deadline_without_waiting_for_full_chunk(tmp_path, monkeypatch):
    clock = [0.0]
    response = _cortex_checkpoint_export_Response(chunks=[b"x"] * 10)
    read1 = response.raw.read1

    def read(size, *, decode_content):
        clock[0] += 0.4
        return read1(size, decode_content=decode_content)

    response.raw.read1 = read
    monkeypatch.setattr(exporter.time, "monotonic", lambda: clock[0])
    client, _, _, session, _ = _cortex_checkpoint_export_fixture(monkeypatch, responses=[response])
    with pytest.raises(exporter.CheckpointExportError, match="deadline"):
        _cortex_checkpoint_export_export(client, tmp_path, timeout_s=1)
    assert len(response.read_calls) == 3
    assert response.timeouts == pytest.approx([1.0, 0.6, 0.2])
    assert response.closed and session.closed
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("missing", ["raw", "read1", "connection", "socket", "settimeout"])
def test_cortex_checkpoint_export_unbounded_raw_interfaces_fail_closed(tmp_path, monkeypatch, missing):
    response = _cortex_checkpoint_export_Response()
    if missing == "raw":
        del response.raw
    elif missing == "read1":
        response.raw.read1 = None
    elif missing == "connection":
        response.raw.connection = None
    elif missing == "socket":
        response.raw.connection.sock = None
    else:
        response.raw.connection.sock.settimeout = None
    client, _, _, session, _ = _cortex_checkpoint_export_fixture(monkeypatch, responses=[response])
    with pytest.raises(exporter.CheckpointExportError, match="bounded raw"):
        _cortex_checkpoint_export_export(client, tmp_path)
    assert response.closed and session.closed
    assert not response.read_calls
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("declaration", ["header", "raw"])
def test_cortex_checkpoint_export_chunked_framing_fails_closed_before_reads(tmp_path, monkeypatch, declaration):
    response = _cortex_checkpoint_export_Response()
    if declaration == "header":
        response.headers["Transfer-Encoding"] = "chunked"
    else:
        response.raw.chunked = True
    client, _, _, session, _ = _cortex_checkpoint_export_fixture(monkeypatch, responses=[response])
    with pytest.raises(exporter.CheckpointExportError, match="chunked"):
        _cortex_checkpoint_export_export(client, tmp_path)
    assert response.closed and session.closed
    assert not response.read_calls
    assert not list(tmp_path.iterdir())


def test_cortex_checkpoint_export_raw_empty_read_is_eof_not_an_interim_chunk(tmp_path, monkeypatch):
    response = _cortex_checkpoint_export_Response(chunks=[b"check", b"", b"point"])
    client, _, _, _, _ = _cortex_checkpoint_export_fixture(monkeypatch, responses=[response])
    with pytest.raises(exporter.CheckpointExportError, match="truncated"):
        _cortex_checkpoint_export_export(client, tmp_path)
    assert len(response.read_calls) == 2
    assert response.closed
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("value", [None, "checkpoint", b"x" * 12])
def test_cortex_checkpoint_export_invalid_raw_read_output_fails_closed(tmp_path, monkeypatch, value):
    response = _cortex_checkpoint_export_Response()
    response.raw.read1 = lambda *args, **kwargs: value
    client, _, _, _, _ = _cortex_checkpoint_export_fixture(monkeypatch, responses=[response])
    with pytest.raises(exporter.CheckpointExportError, match="bounded raw bytes"):
        _cortex_checkpoint_export_export(client, tmp_path)
    assert response.closed
    assert not list(tmp_path.iterdir())


def test_cortex_checkpoint_export_socket_timeout_failure_is_sanitized_and_cleans_snapshot(tmp_path, monkeypatch):
    response = _cortex_checkpoint_export_Response()

    def settimeout(timeout):
        raise OSError(_cortex_checkpoint_export_SECRET)

    response.raw.connection.sock.settimeout = settimeout
    client, _, _, session, _ = _cortex_checkpoint_export_fixture(monkeypatch, responses=[response])
    with pytest.raises(exporter.CheckpointExportError) as error:
        _cortex_checkpoint_export_export(client, tmp_path)
    assert _cortex_checkpoint_export_SECRET not in "".join(traceback.format_exception(error.value))
    assert response.closed and session.closed
    assert not list(tmp_path.iterdir())


def test_cortex_checkpoint_export_interrupted_download_cleans_its_owned_snapshot(tmp_path, monkeypatch):
    response = _cortex_checkpoint_export_Response(chunks=[b"part", KeyboardInterrupt()])
    client, _, _, session, _ = _cortex_checkpoint_export_fixture(monkeypatch, responses=[response])
    with pytest.raises(KeyboardInterrupt):
        _cortex_checkpoint_export_export(client, tmp_path)
    assert response.closed
    assert session.closed
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("value", [0, -1, True, None, 1.5, "2"])
@pytest.mark.parametrize("field", ["expected_dp_world_size", "max_total_bytes"])
def test_cortex_checkpoint_export_invalid_caller_budgets_rejected_before_post(tmp_path, monkeypatch, value, field):
    client, _, posted, session, _ = _cortex_checkpoint_export_fixture(monkeypatch)
    kwargs = {"expected_dp_world_size": 2, field: value}
    with pytest.raises(exporter.CheckpointExportError, match="positive integer"):
        exporter.export_checkpoint(
            client, _cortex_checkpoint_export_JOB_ID, _cortex_checkpoint_export_CP_ID, tmp_path, **kwargs
        )
    assert not posted
    assert not session.calls


@pytest.mark.parametrize(
    "reply",
    [
        None,
        [],
        {"checkpoint_id": _cortex_checkpoint_export_CP_ID},
        {"checkpoint_id": _cortex_checkpoint_export_CP_ID, "files": {}},
    ],
)
def test_cortex_checkpoint_export_malformed_export_envelope_rejected_without_download(tmp_path, monkeypatch, reply):
    client, _, _, session, _ = _cortex_checkpoint_export_fixture(monkeypatch)
    client.transport.call = lambda request: reply
    with pytest.raises(exporter.CheckpointExportError):
        _cortex_checkpoint_export_export(client, tmp_path / "nested" / "exports")
    assert not session.calls
    assert not list(tmp_path.iterdir())


def test_cortex_checkpoint_export_symlink_destination_rejected_without_touching_target(tmp_path, monkeypatch):
    client, _, posted, session, _ = _cortex_checkpoint_export_fixture(monkeypatch)
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(exporter.CheckpointExportError, match="symlink"):
        _cortex_checkpoint_export_export(client, link / "child")
    assert not posted
    assert not session.calls
    assert not list(actual.iterdir())


def test_cortex_checkpoint_export_injected_tag_symlink_cannot_redirect_writes(tmp_path, monkeypatch):
    client, _, _, _, _ = _cortex_checkpoint_export_fixture(monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    real_mkdtemp = exporter.tempfile.mkdtemp

    def mkdtemp(**kwargs):
        root = Path(real_mkdtemp(**kwargs))
        (root / "global_step10").symlink_to(outside, target_is_directory=True)
        return str(root)

    monkeypatch.setattr(exporter.tempfile, "mkdtemp", mkdtemp)
    with pytest.raises(exporter.CheckpointExportError):
        _cortex_checkpoint_export_export(client, tmp_path)
    assert not list(outside.iterdir())
    assert list(tmp_path.iterdir()) == [outside]


def test_cortex_checkpoint_export_existing_file_or_symlink_never_overwritten(tmp_path):
    existing = tmp_path / "metadata.json"
    existing.write_text("do not replace")
    with pytest.raises(FileExistsError):
        exporter._open_exclusive(tmp_path, "metadata.json")
    assert existing.read_text() == "do not replace"
    link = tmp_path / "link"
    link.symlink_to(existing)
    with pytest.raises(OSError):
        exporter._open_exclusive(tmp_path, "link")
    assert existing.read_text() == "do not replace"


def test_cortex_checkpoint_export_receipt_write_failure_cleans_every_downloaded_file(tmp_path, monkeypatch):
    client, _, _, session, _ = _cortex_checkpoint_export_fixture(monkeypatch)
    real_open = exporter._open_exclusive

    def open_exclusive(root, name):
        if name == "export_receipt.json":
            raise OSError(_cortex_checkpoint_export_SECRET)
        return real_open(root, name)

    monkeypatch.setattr(exporter, "_open_exclusive", open_exclusive)
    with pytest.raises(exporter.CheckpointExportError) as error:
        _cortex_checkpoint_export_export(client, tmp_path)
    assert _cortex_checkpoint_export_SECRET not in "".join(traceback.format_exception(error.value))
    assert session.closed
    assert not list(tmp_path.iterdir())


def test_cortex_checkpoint_export_job_identity_and_unsafe_endpoint_ids_rejected_before_post(tmp_path, monkeypatch):
    client, _, posted, session, _ = _cortex_checkpoint_export_fixture(monkeypatch)
    with pytest.raises(exporter.CheckpointExportError, match="parent job"):
        exporter.export_checkpoint(
            client, "other-job", _cortex_checkpoint_export_CP_ID, tmp_path, expected_dp_world_size=2
        )
    with pytest.raises(exporter.CheckpointExportError, match="checkpoint ID"):
        exporter.export_checkpoint(
            client, _cortex_checkpoint_export_JOB_ID, "../cp", tmp_path, expected_dp_world_size=2
        )
    assert not posted
    assert not session.calls


# Model state capture.


@pytest.fixture
def _model_state_capture_capture_module():
    return importlib.import_module("arctic_platform.correctness.harness.optimizer_capture")


def _model_state_capture_engine():
    model = torch.nn.Linear(4, 3, bias=False, dtype=torch.bfloat16)
    model.weight.requires_grad_(False)
    model.weight.allreduce = False
    model.weight.group_name = "ep_size_8"
    model.register_buffer("persistent", torch.arange(4))
    model.register_buffer("runtime", torch.ones(2), persistent=False)
    return SimpleNamespace(module=model, global_steps=10, has_moe_layers=False, num_experts=[])


def test_model_state_capture_manifest_includes_frozen_experts_and_all_buffers(_model_state_capture_capture_module):
    manifest = _model_state_capture_capture_module.model_state_manifest(_model_state_capture_engine(), rank=7)
    assert manifest["rank"] == 7
    assert manifest["engine"]["global_steps"] == 10
    assert manifest["engine"]["has_moe_layers"] is False
    entries = {entry["name"]: entry for entry in manifest["tensors"]}
    assert set(entries) == {"weight", "persistent", "runtime"}
    assert entries["weight"]["requires_grad"] is False
    assert entries["weight"]["group_name"] == "ep_size_8"
    assert entries["weight"]["allreduce"] is False
    assert entries["weight"]["shape"] == [3, 4]
    assert entries["weight"]["dtype"] == "torch.bfloat16"
    assert entries["weight"]["kind"] == "parameter"
    assert entries["persistent"]["persistent"] is True
    assert entries["runtime"]["persistent"] is False


def test_model_state_capture_hash_is_exact_layout_independent_and_handles_scalars(_model_state_capture_capture_module):
    tensor = torch.arange(80, dtype=torch.bfloat16).reshape(8, 10).t()
    digest = _model_state_capture_capture_module.tensor_digest(tensor, chunk_bytes=8)
    assert digest == _model_state_capture_capture_module.tensor_digest(tensor.contiguous(), chunk_bytes=8)
    changed = tensor.clone()
    changed[0, 0] += 1
    assert digest != _model_state_capture_capture_module.tensor_digest(changed, chunk_bytes=8)
    assert _model_state_capture_capture_module.tensor_digest(torch.tensor(2.0))
    assert _model_state_capture_capture_module.tensor_digest(torch.empty(0))


def test_model_state_capture_hash_cpu_transfers_are_bounded(_model_state_capture_capture_module, monkeypatch):
    original_cpu = torch.Tensor.cpu
    transfers = []

    def cpu(tensor, *args, **kwargs):
        transfers.append(tensor.numel() * tensor.element_size())
        return original_cpu(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", cpu)
    _model_state_capture_capture_module.tensor_digest(torch.arange(128).reshape(8, 16).t(), chunk_bytes=32)
    assert transfers and max(transfers) <= 32


def test_model_state_capture_save_load_hooks_capture_at_boundary_without_changing_results(
    _model_state_capture_capture_module, tmp_path, monkeypatch
):
    monkeypatch.setitem(sys.modules, "deepspeed", SimpleNamespace(__version__="test-installed"))
    events = []

    class Base:

        def save_checkpoint(self, path, export_hf=False):
            events.append(("save", path, export_hf))
            self.engine.module.weight.data.add_(1)
            return {"path": path}

        def load_checkpoint(self, path):
            events.append(("load", path))
            self.engine.module.weight.data.add_(2)
            return 10

    class Worker(_model_state_capture_capture_module.ModelStateCaptureMixin, Base):
        pass

    worker = Worker()
    worker.engine, worker.rank = (_model_state_capture_engine(), 3)
    worker._model_state_output_dir = str(tmp_path)
    before = _model_state_capture_capture_module.tensor_digest(worker.engine.module.weight)
    assert worker.save_checkpoint("/checkpoint", export_hf=True) == {"path": "/checkpoint"}
    assert worker.load_checkpoint("/checkpoint") == 10
    assert events == [("save", "/checkpoint", True), ("load", "/checkpoint")]
    saved = json.loads((tmp_path / "before-save" / "rank-00003.json").read_text())
    loaded = json.loads((tmp_path / "after-load" / "rank-00003.json").read_text())
    assert saved["tensors"][0]["sha256"] == before
    assert saved["deepspeed_version"] == "test-installed"
    assert loaded["tensors"][0]["sha256"] == _model_state_capture_capture_module.tensor_digest(
        worker.engine.module.weight
    )


def test_model_state_capture_hooks_are_inert_when_not_requested(_model_state_capture_capture_module, tmp_path):

    class Base:

        def save_checkpoint(self, path, export_hf=False):
            return {"path": path}

        def load_checkpoint(self, path):
            return 0

    class Worker(_model_state_capture_capture_module.ModelStateCaptureMixin, Base):
        _model_state_output_dir = None

    worker = Worker()
    assert worker.save_checkpoint("/checkpoint") == {"path": "/checkpoint"}
    assert worker.load_checkpoint("/checkpoint") == 0
    assert list(tmp_path.iterdir()) == []


def _model_state_capture_write_pair(capture_module, tmp_path, changed_rank=None):
    for rank in range(2):
        engine = _model_state_capture_engine()
        left = capture_module.model_state_manifest(engine, rank)
        if rank == changed_rank:
            engine.module.weight.data[0, 0] += 1
        right = capture_module.model_state_manifest(engine, rank)
        for phase, manifest in (("before-save", left), ("after-load", right)):
            directory = tmp_path / phase
            directory.mkdir(exist_ok=True)
            (directory / f"rank-{rank:05d}.json").write_text(json.dumps(manifest))


def test_model_state_capture_comparison_detects_frozen_rank_local_corruption(
    _model_state_capture_capture_module, tmp_path
):
    _model_state_capture_write_pair(_model_state_capture_capture_module, tmp_path, changed_rank=1)
    result = _model_state_capture_capture_module.compare_model_state(
        tmp_path / "before-save", tmp_path / "after-load", ranks=2
    )
    assert result["exact"] is False
    assert result["compared_tensors"] == 6
    assert result["differences"] == [{"rank": 1, "kind": "parameter", "name": "weight", "fields": ["sha256"]}]


def test_model_state_capture_comparison_rejects_incomplete_rank_coverage(
    _model_state_capture_capture_module, tmp_path
):
    _model_state_capture_write_pair(_model_state_capture_capture_module, tmp_path)
    (tmp_path / "after-load" / "rank-00001.json").unlink()
    with pytest.raises(RuntimeError, match="rank coverage"):
        _model_state_capture_capture_module.compare_model_state(
            tmp_path / "before-save", tmp_path / "after-load", ranks=2
        )


def test_model_state_capture_comparison_reports_exact_model_state(_model_state_capture_capture_module, tmp_path):
    _model_state_capture_write_pair(_model_state_capture_capture_module, tmp_path)
    result = _model_state_capture_capture_module.compare_model_state(
        tmp_path / "before-save", tmp_path / "after-load", ranks=2
    )
    assert result["exact"] is True
    assert result["differences"] == []
    assert result["compared_tensors"] == 6


# Optimizer state check.

_optimizer_state_check_COMPARE_KWARGS = {"optimizer_config": {}, "learning_rate": OPTIMIZER_LEARNING_RATE}


def _optimizer_state_check_write_artifact(root: Path, name: str, tensors: dict) -> Path:
    root.mkdir()
    torch.save(tensors, root / "000000.pt")
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"step": 1, "parameters": [{"name": name, "file": "000000.pt"}]}))
    return manifest


def _optimizer_state_check_write_artifacts(root: Path, tensors_by_name: dict[str, dict]) -> Path:
    root.mkdir()
    parameters = []
    for index, (name, tensors) in enumerate(tensors_by_name.items()):
        filename = f"{index:06d}.pt"
        torch.save(tensors, root / filename)
        parameters.append({"name": name, "file": filename})
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"step": 1, "parameters": parameters}))
    return manifest


def _optimizer_state_check_states(value: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "parameter_update": value.clone(),
        "exp_avg": torch.zeros_like(value),
        "exp_avg_sq": torch.zeros_like(value),
    }


def test_optimizer_state_check_compare_optimizer_artifacts_aligns_qwen_moe_tensor_layouts(tmp_path):
    w1 = torch.tensor([[[1.0, 2.0]]])
    w3 = torch.tensor([[[3.0, 4.0]]])
    w2 = torch.tensor([[[5.0], [6.0]]])
    router = torch.tensor([[7.0]])
    shared_gate = torch.tensor([[8.0, 9.0]])
    dss = _optimizer_state_check_write_artifacts(
        tmp_path / "dss-moe",
        {
            "module.layers.0.mlp.experts.w1": _optimizer_state_check_states(w1),
            "module.layers.0.mlp.experts.w2": _optimizer_state_check_states(w2),
            "module.layers.0.mlp.experts.w3": _optimizer_state_check_states(w3),
            "module.layers.0.mlp.router.gate.weight": _optimizer_state_check_states(router),
            "module.layers.0.mlp.shared_expert.w1.weight": _optimizer_state_check_states(shared_gate),
            "module.visual.unused.weight": _optimizer_state_check_states(torch.zeros(2)),
        },
    )
    reference = _optimizer_state_check_write_artifacts(
        tmp_path / "reference-moe",
        {
            "model.layers.0.mlp.experts.gate_up_proj": _optimizer_state_check_states(torch.cat([w1, w3], dim=1)),
            "model.layers.0.mlp.experts.down_proj": _optimizer_state_check_states(w2),
            "model.layers.0.mlp.gate.weight": _optimizer_state_check_states(router),
            "model.layers.0.mlp.shared_expert.gate_proj.weight": _optimizer_state_check_states(shared_gate),
        },
    )
    result = compare_optimizer_artifacts(dss, reference, **_optimizer_state_check_COMPARE_KWARGS)
    assert result.only_dss == []
    assert result.only_reference == []
    assert len(result.deltas) == 12
    assert result.worst_delta == 0.0


def test_optimizer_state_check_compare_optimizer_artifacts_uses_full_tensor_residual_delta_norm(tmp_path):
    reference = _optimizer_state_check_write_artifact(
        tmp_path / "reference",
        "model.layer.weight",
        {"parameter_update": torch.tensor([1.0, 2.0]), "exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2)},
    )
    dss = _optimizer_state_check_write_artifact(
        tmp_path / "dss",
        "module.layer.weight",
        {"parameter_update": torch.tensor([4.0, 6.0]), "exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2)},
    )
    result = compare_optimizer_artifacts(dss, reference, **_optimizer_state_check_COMPARE_KWARGS)
    assert result.only_dss == []
    assert result.only_reference == []
    assert [(item.state, item.delta_norm) for item in result.deltas] == [
        ("parameter_update_residual", 5.0),
        ("exp_avg", 0.0),
        ("exp_avg_sq", 0.0),
    ]
    assert result.worst_delta == 5.0


def test_optimizer_state_check_compare_optimizer_artifacts_rejects_shape_mismatch(tmp_path):
    reference = _optimizer_state_check_write_artifact(
        tmp_path / "reference",
        "layer.weight",
        {"parameter_update": torch.zeros(2), "exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2)},
    )
    dss = _optimizer_state_check_write_artifact(
        tmp_path / "dss",
        "layer.weight",
        {"parameter_update": torch.zeros(3), "exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2)},
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        compare_optimizer_artifacts(dss, reference, **_optimizer_state_check_COMPARE_KWARGS)


def test_optimizer_state_check_delta_update_arm_applies_reviewed_norm_gate_to_every_state(tmp_path):
    reference = _optimizer_state_check_write_artifact(
        tmp_path / "reference",
        "layer.weight",
        {"parameter_update": torch.zeros(2), "exp_avg": torch.zeros(2), "exp_avg_sq": torch.zeros(2)},
    )
    dss_manifest = _optimizer_state_check_write_artifact(
        tmp_path / "dss",
        "layer.weight",
        _optimizer_state_check_first_step_adam_state(torch.tensor([0.02, 0.0]), backend="dss"),
    )
    spec = SimpleNamespace(
        tolerance_for=lambda test_id: 1.0, test_tolerances={"single-step-optimizer": Tolerance(absolute=1.0)}
    )
    ctx = SimpleNamespace(config_id="config", spec=spec, cfg=SimpleNamespace(training={"optimizer": {}}))
    arm = SimpleNamespace(name="gas1")
    reference_result = {
        "optimizer_norms": summarize_optimizer_artifacts(reference, **_optimizer_state_check_COMPARE_KWARGS),
        "optimizer_norm_gate": {
            "criterion": "optimizer-state-l2-norm-v1",
            "absolute": 0.001,
            "status": "reviewed",
            "provenance": "selftest reviewed reference calibration",
        },
        "loss": 2.0,
        "microbatches": 1,
    }
    dss_result = SimpleNamespace(optimizer_state_manifest=str(dss_manifest), avg_loss=2.0, model_calls=1)
    result = _compare_arm(ctx, arm, reference_result, dss_result)
    assert result.outcome is Outcome.FAIL
    assert result.metrics["optimizer_values_compared"] == 3
    assert result.metrics["optimizer_values_over_criterion"] == 1
    assert result.metrics["stated_criterion_abs"] == 0.001
    assert result.worst_name == "layer.weight::exp_avg"
    assert [mismatch.name for mismatch in result.mismatches] == ["layer.weight::exp_avg"]


def test_optimizer_state_check_pairwise_optimizer_variation_is_not_limited_to_the_first_run(tmp_path: Path) -> None:
    manifests = []
    for index, value in enumerate((0.0, 1.0, -1.0)):
        manifests.append(
            _optimizer_state_check_write_artifact(
                tmp_path / f"run-{index}",
                "layer.weight",
                {"parameter_update": torch.tensor([value]), "exp_avg": torch.zeros(1), "exp_avg_sq": torch.zeros(1)},
            )
        )
    worst = max_pairwise_optimizer_delta(manifests, **_optimizer_state_check_COMPARE_KWARGS)
    assert worst is not None
    assert (worst.name, worst.state, worst.delta_norm) == ("layer.weight", "parameter_update_residual", 2.0)


def _optimizer_state_check_first_step_adam_state(
    gradient: torch.Tensor, *, backend: str = "reference"
) -> dict[str, torch.Tensor]:
    beta1, beta2 = (0.9, 0.999)
    epsilon = 1e-08
    learning_rate = OPTIMIZER_LEARNING_RATE
    if backend == "dss":
        beta1 = float(torch.tensor(beta1, dtype=torch.float32))
        beta2 = float(torch.tensor(beta2, dtype=torch.float32))
        epsilon = float(torch.tensor(epsilon, dtype=torch.float32))
        learning_rate = float(torch.tensor(learning_rate, dtype=torch.float32))
    elif backend != "reference":
        raise ValueError(f"unknown backend: {backend}")
    exp_avg = (1.0 - beta1) * gradient
    exp_avg_sq = (1.0 - beta2) * gradient.square()
    if backend == "dss":
        beta1_correction = float(torch.tensor(1.0 - beta1, dtype=torch.float32))
        beta2_correction = float(torch.tensor(1.0 - beta2, dtype=torch.float32))
        update = -(exp_avg / beta1_correction) / ((exp_avg_sq / beta2_correction).sqrt() + epsilon)
        update = update * learning_rate
    else:
        update = -learning_rate * gradient / (gradient.abs() + epsilon)
    return {"parameter_update": update, "exp_avg": exp_avg, "exp_avg_sq": exp_avg_sq}


def test_optimizer_state_check_moment_implied_raw_update_difference_has_zero_residual(tmp_path: Path) -> None:
    reference = _optimizer_state_check_write_artifact(
        tmp_path / "reference-sign",
        "layer.weight",
        _optimizer_state_check_first_step_adam_state(torch.tensor([1e-06])),
    )
    dss = _optimizer_state_check_write_artifact(
        tmp_path / "dss-sign",
        "layer.weight",
        _optimizer_state_check_first_step_adam_state(torch.tensor([-1e-06]), backend="dss"),
    )
    result = compare_optimizer_artifacts(dss, reference, **_optimizer_state_check_COMPARE_KWARGS)
    by_state = {item.state: item.delta_norm for item in result.deltas}
    assert by_state["parameter_update_residual"] == pytest.approx(0.0, abs=1e-09)
    assert by_state["exp_avg"] == pytest.approx(2e-07)


def test_optimizer_state_check_unexplained_update_corruption_remains_in_residual(tmp_path: Path) -> None:
    reference_states = _optimizer_state_check_first_step_adam_state(torch.tensor([1.0]))
    dss_states = _optimizer_state_check_first_step_adam_state(torch.tensor([1.0]), backend="dss")
    dss_states["parameter_update"] += 0.005
    reference = _optimizer_state_check_write_artifact(tmp_path / "reference-corrupt", "layer.weight", reference_states)
    dss = _optimizer_state_check_write_artifact(tmp_path / "dss-corrupt", "layer.weight", dss_states)
    result = compare_optimizer_artifacts(dss, reference, **_optimizer_state_check_COMPARE_KWARGS)
    by_state = {item.state: item.delta_norm for item in result.deltas}
    assert by_state["parameter_update_residual"] == pytest.approx(0.005)
    assert by_state["exp_avg"] < 0.001
    assert by_state["exp_avg_sq"] < 0.001


# Native retained checkpoint store.


@pytest.fixture
def _native_checkpoint_fresh_child(tmp_path, monkeypatch):
    import base64
    import os
    import shlex

    from arctic_platform.correctness.harness import checkpoint_store as module

    overlay = tmp_path / "private-overlay"
    package = overlay / "arctic_platform" / "correctness" / "harness"
    package.mkdir(parents=True)
    for directory in (package, package.parent, package.parent.parent):
        (directory / "__init__.py").write_text("")
    source = package / "checkpoint_store.py"
    source.write_text(
        "def _node_receipt(request):\n"
        "    binding = dict(request['binding'], native_path='/synthetic/native')\n"
        "    identity = {key: [1, 2, 3, 4] for key in ('root', 'marker', 'link', 'lock')}\n"
        "    if request['operation'] == 'close':\n"
        "        identity = {'closed': True, 'lock': [1, 2, 3, 4]}\n"
        "    return dict(host='synthetic-node', token=request['token'], operation=request['operation'],\n"
        "                binding=binding, ok=True, identity=identity, error=None)\n"
    )
    child_cwd = tmp_path / "login-cwd"
    child_cwd.mkdir()
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("node-0 slots=1\n")
    original_run = module.subprocess.run
    children = []
    before_child = []

    def run(command, **kwargs):
        assert command[1:3] == ["-f", str(hostfile)]
        tokens = shlex.split(command[-1])
        assert tokens[0] == "echo"
        script = base64.b64decode(tokens[1], validate=True).decode()
        for hook in before_child:
            hook()
        child_env = dict(os.environ)
        child_env.pop("PYTHONPATH", None)
        done = original_run(
            [sys.executable, "-B", "-"],
            input=script,
            cwd=child_cwd,
            env=child_env,
            **kwargs,
        )
        children.append(done)
        # Model ds_ssh reporting rc=0 even when its remote child fails.
        return SimpleNamespace(
            returncode=0,
            stdout="".join(f"node-0: {line}\n" for line in done.stdout.splitlines()),
            stderr=done.stderr,
        )

    monkeypatch.setattr(module, "__file__", str(source))
    monkeypatch.setattr(sys, "path", [str(overlay), *sys.path])
    monkeypatch.setattr(module.subprocess, "run", run)
    return SimpleNamespace(
        module=module,
        source=source,
        overlay=overlay,
        children=children,
        before_child=before_child,
        store=module.NativeCheckpointStore(hostfile),
    )


def test_native_checkpoint_fanout_fresh_child_uses_private_overlay(_native_checkpoint_fresh_child):
    env = _native_checkpoint_fresh_child
    binding = {"source_job_id": "synthetic-source", "path": "/synthetic/retained"}
    receipts = env.store._fanout("create", binding, {})
    assert receipts["node-0"]["host"] == "synthetic-node"
    env.store._fanout("verify", binding, receipts)
    closed = env.store._fanout("close", binding, receipts)
    assert closed["node-0"]["identity"]["closed"] is True
    assert len(env.children) == 3
    assert all(child.returncode == 0 and child.stderr == "" for child in env.children)


@pytest.mark.parametrize("import_root", ["absolute", "relative", "cwd", "non-string"])
def test_native_checkpoint_fanout_fresh_child_keeps_gateway_dependency_roots(
    _native_checkpoint_fresh_child, monkeypatch, import_root
):
    env = _native_checkpoint_fresh_child
    dependency = env.overlay.parent / "native-dependency"
    dependency.mkdir()
    (dependency / "synthetic_native_dependency.py").write_text("NATIVE_PATH = '/synthetic/native'\n")
    env.source.write_text(
        "from synthetic_native_dependency import NATIVE_PATH\n"
        + env.source.read_text().replace("native_path='/synthetic/native'", "native_path=NATIVE_PATH")
    )
    if import_root == "cwd":
        monkeypatch.chdir(dependency)
        root = ""
    else:
        monkeypatch.chdir(dependency.parent)
        root = str(dependency) if import_root in {"absolute", "non-string"} else dependency.name
    paths = [root, *sys.path]
    if import_root == "non-string":
        paths.insert(0, None)
    monkeypatch.setattr(sys, "path", paths)
    receipts = env.store._fanout("create", {"path": "/synthetic/retained"}, {})
    assert receipts["node-0"]["binding"]["native_path"] == "/synthetic/native"
    assert env.children[0].returncode == 0


@pytest.mark.parametrize("source_change", ["changed", "missing", "shadowed"])
def test_native_checkpoint_fanout_fresh_child_rejects_wrong_source(_native_checkpoint_fresh_child, source_change):
    env = _native_checkpoint_fresh_child
    if source_change == "changed":
        env.before_child.append(
            lambda: env.source.write_text("raise AssertionError('must not execute changed source')\n")
        )
        expected_error = "checkpoint store source SHA256 differs"
    elif source_change == "missing":
        env.before_child.append(env.source.unlink)
        expected_error = "FileNotFoundError"
    else:
        shadow = env.overlay.parent / "foreign-harness"
        shadow.mkdir()
        (shadow / "checkpoint_store.py").write_text(
            "def _node_receipt(request):\n    raise AssertionError('must not call foreign receipt')\n"
        )
        env.source.parent.joinpath("__init__.py").write_text(f"__path__ = [{str(shadow)!r}]\n")
        expected_error = "checkpoint store imported source path differs"
    with pytest.raises(env.module.CheckpointStoreError, match="missing hosts=.*node-0") as error:
        env.store._fanout("create", {"path": "/synthetic/retained"}, {})
    assert len(env.children) == 1 and env.children[0].returncode != 0
    assert error.value.receipts == {}
    assert error.value.transport_stderr == env.children[0].stderr
    assert expected_error in error.value.transport_stderr
    assert "must not" not in error.value.transport_stderr
    assert expected_error in "".join(traceback.format_exception(error.value))


def test_native_checkpoint_fanout_missing_gateway_source_is_structured(_native_checkpoint_fresh_child):
    env = _native_checkpoint_fresh_child
    env.source.unlink()
    with pytest.raises(env.module.CheckpointStoreError) as error:
        env.store._fanout("create", {"path": "/synthetic/retained"}, {})
    assert isinstance(error.value.__cause__, FileNotFoundError)
    assert env.children == []
    assert error.value.receipts == {}
    assert error.value.transport_stdout == error.value.transport_stderr == ""


@pytest.fixture
def _native_checkpoint_store_env(tmp_path, monkeypatch):
    import base64
    import io
    import shlex

    from arctic_platform.correctness.harness import checkpoint_store as module

    fast = tmp_path / "fast"
    native = tmp_path / "native"
    fast.mkdir()
    native.mkdir()
    root = fast / "retained"
    root.mkdir()
    hostfile = tmp_path / "hostfile"
    hostfile.write_text("node-0 slots=1\n")
    monkeypatch.setattr(module, "_FAST_ROOT", fast)
    monkeypatch.setattr(module, "_native_path", lambda source: native / f"job-{source}")
    monkeypatch.setattr(module.socket, "gethostname", lambda: "node-0")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        assert command[1:3] == ["-f", str(hostfile)]
        assert kwargs["timeout"] == 120
        tokens = shlex.split(command[-1])
        if tokens[0] == "echo":
            assert tokens[2:7] == ["|", "base64", "-d", "|", sys.executable]
            assert tokens[7:] == ["-"]
            script = base64.b64decode(tokens[1], validate=True).decode()
        else:
            script = tokens[-1]
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(script, {})
        return SimpleNamespace(returncode=0, stdout=f"node-0: {output.getvalue()}", stderr="")

    monkeypatch.setattr(module.subprocess, "run", run)
    return SimpleNamespace(
        module=module,
        root=root,
        fast=fast,
        native=native,
        hostfile=hostfile,
        run=run,
        calls=calls,
        store=module.NativeCheckpointStore(hostfile),
    )


def test_native_checkpoint_store_survives_ds_ssh_unquoted_argument_flattening(
    _native_checkpoint_store_env, monkeypatch
):
    env = _native_checkpoint_store_env

    def flattened(command, **kwargs):
        remote = " ".join(command[-1].split())
        return env.run([*command[:-1], remote], **kwargs)

    monkeypatch.setattr(env.module.subprocess, "run", flattened)
    binding = env.store.create(env.root)
    env.store.verify(binding)
    env.store.close()
    assert len(env.calls) == 3
    assert not env.root.exists()


def test_native_checkpoint_store_missing_peer_names_the_absent_host(_native_checkpoint_store_env):
    env = _native_checkpoint_store_env
    env.hostfile.write_text("node-0 slots=1\nnode-1 slots=1\n")
    store = env.module.NativeCheckpointStore(env.hostfile)
    with pytest.raises(env.module.CheckpointStoreError, match="missing hosts=.*node-1"):
        store.create(env.root)


def test_native_checkpoint_store_preserves_launcher_output(_native_checkpoint_store_env, monkeypatch):
    env = _native_checkpoint_store_env
    output = SimpleNamespace(
        returncode=0,
        stdout="hostfile=/job/hostfile\nnode-0: diagnostic-only output\n",
        stderr="node-0: ModuleNotFoundError: No module named 'arctic_platform'\n",
    )
    monkeypatch.setattr(env.module.subprocess, "run", lambda *args, **kwargs: output)
    with pytest.raises(env.module.CheckpointStoreError, match="missing hosts=.*node-0") as error:
        env.store.create(env.root)
    assert error.value.transport_stdout == output.stdout
    assert error.value.transport_stderr == output.stderr
    rendered = "".join(traceback.format_exception(error.value))
    assert output.stdout.strip() in rendered
    assert output.stderr.strip() in rendered
    assert error.value.receipts == {}
    assert env.root.exists()


def test_native_checkpoint_store_preserves_timeout_output(_native_checkpoint_store_env, monkeypatch):
    env = _native_checkpoint_store_env

    def timeout(command, **kwargs):
        raise env.module.subprocess.TimeoutExpired(
            command, kwargs["timeout"], output=b"node-0: started\n", stderr=b"node-0: waiting\n"
        )

    monkeypatch.setattr(env.module.subprocess, "run", timeout)
    with pytest.raises(env.module.CheckpointStoreError) as error:
        env.store.create(env.root)
    assert error.value.transport_stdout == "node-0: started\n"
    assert error.value.transport_stderr == "node-0: waiting\n"
    assert isinstance(error.value.__cause__, env.module.subprocess.TimeoutExpired)
    assert error.value.receipts == {}
    assert env.root.exists()


def test_native_checkpoint_store_create_verify_and_explicit_close(_native_checkpoint_store_env):
    env = _native_checkpoint_store_env
    binding = env.store.create(env.root)
    native = Path(binding["native_path"])
    lock = Path(f"{native}.lock")
    lock_inode = lock.stat().st_ino
    assert set(binding) == {"source_job_id", "path", "native_path"}
    assert native.is_symlink() and native.readlink() == env.root
    (env.root / "global_step1").mkdir()
    (env.root / "global_step1/model.pt").write_bytes(b"synthetic checkpoint")
    env.store.verify(binding)
    assert env.store.bindings == (binding,)
    env.store.bindings[0]["path"] = "not the store's binding"
    assert env.store.bindings[0] == binding
    env.store.close()
    assert not native.exists() and not native.is_symlink() and not env.root.exists()
    assert lock.stat().st_ino == lock_inode
    assert env.store.bindings == ()
    assert len(env.calls) == 3
    env.store.close()


def test_native_checkpoint_store_root_is_retained_without_close(_native_checkpoint_store_env):
    env = _native_checkpoint_store_env
    binding = env.store.create(env.root)
    assert env.root.exists() and Path(binding["native_path"]).is_symlink()
    assert len(env.calls) == 1


@pytest.mark.parametrize("condition", ["nonempty", "symlink", "outside"])
def test_native_checkpoint_store_rejects_unsafe_roots(_native_checkpoint_store_env, condition):
    env = _native_checkpoint_store_env
    root = env.root
    if condition == "nonempty":
        (root / "foreign").write_bytes(b"preserve")
    elif condition == "symlink":
        root = env.fast / "alias"
        root.symlink_to(env.root, target_is_directory=True)
    else:
        root = env.native
    with pytest.raises((ValueError, env.module.CheckpointStoreError)):
        env.store.create(root)
    assert env.root.exists()
    assert all(
        path.name.endswith(".lock") and path.is_file() and not path.is_symlink()
        for path in env.native.glob("job-correctness-*")
    )
    if condition == "nonempty":
        assert (root / "foreign").read_bytes() == b"preserve"


def test_native_checkpoint_store_never_overwrites_existing_native_namespace(_native_checkpoint_store_env, monkeypatch):
    env = _native_checkpoint_store_env
    native = env.native / "existing"
    native.write_bytes(b"foreign")
    monkeypatch.setattr(env.module, "_native_path", lambda source: native)
    with pytest.raises(env.module.CheckpointStoreError, match="already exists"):
        env.store.create(env.root)
    assert native.read_bytes() == b"foreign"
    assert list(env.root.iterdir()) == []
    assert env.store.bindings[0]["path"] == str(env.root)


@pytest.mark.parametrize("changed", ["link", "root", "marker", "lock"])
def test_native_checkpoint_store_preserves_replaced_owned_objects(_native_checkpoint_store_env, changed):
    env = _native_checkpoint_store_env
    binding = env.store.create(env.root)
    native = Path(binding["native_path"])
    if changed == "link":
        foreign = env.fast / "foreign"
        foreign.mkdir()
        native.unlink()
        native.symlink_to(foreign)
    elif changed == "root":
        env.root.rename(env.fast / "original")
        env.root.mkdir()
        marker = env.fast / "original" / env.module._MARKER
        (env.root / env.module._MARKER).write_bytes(marker.read_bytes())
    else:
        target = env.root / env.module._MARKER if changed == "marker" else Path(f"{native}.lock")
        replacement = target.with_name("replacement")
        replacement.write_bytes(target.read_bytes())
        replacement.replace(target)
    with pytest.raises(env.module.CheckpointStoreError, match="changed|differs"):
        env.store.verify(binding)
    with pytest.raises(env.module.CheckpointStoreError):
        env.store.close()
    assert env.root.exists() and native.is_symlink()
    assert env.store.bindings


def test_native_checkpoint_store_rejects_payload_symlinks(_native_checkpoint_store_env):
    env = _native_checkpoint_store_env
    binding = env.store.create(env.root)
    (env.root / "payload-link").symlink_to(env.hostfile)
    with pytest.raises(env.module.CheckpointStoreError, match="payload"):
        env.store.verify(binding)
    with pytest.raises(env.module.CheckpointStoreError, match="payload"):
        env.store.close()
    assert env.hostfile.exists() and Path(binding["native_path"]).is_symlink()


def test_native_checkpoint_store_exclusive_close_refuses_active_native_reader(_native_checkpoint_store_env):
    import fcntl

    env = _native_checkpoint_store_env
    binding = env.store.create(env.root)
    native = Path(binding["native_path"])
    with Path(f"{native}.lock").open("r+") as reader:
        fcntl.flock(reader, fcntl.LOCK_SH)
        with pytest.raises(env.module.CheckpointStoreError, match="BlockingIOError"):
            env.store.close()
        assert native.is_symlink() and env.root.exists()
        env.store.verify(binding)
        fcntl.flock(reader, fcntl.LOCK_UN)
    env.store.close()
    assert not env.root.exists()


def test_native_checkpoint_store_cleanup_failure_is_retained_and_can_be_retried(
    _native_checkpoint_store_env, monkeypatch
):
    env = _native_checkpoint_store_env
    binding = env.store.create(env.root)
    remove = env.module.shutil.rmtree

    def fail(path):
        raise PermissionError("synthetic tree cleanup failed")

    fail.avoids_symlink_attacks = True
    monkeypatch.setattr(env.module.shutil, "rmtree", fail)
    with pytest.raises(env.module.CheckpointStoreError, match="synthetic tree cleanup failed"):
        env.store.close()
    assert env.root.exists() and not Path(binding["native_path"]).is_symlink()
    assert env.store.bindings == (binding,)
    monkeypatch.setattr(env.module.shutil, "rmtree", remove)
    env.store.close()
    assert not env.root.exists()


def test_native_checkpoint_store_hostfile_drift_rejects_before_fanout(_native_checkpoint_store_env):
    env = _native_checkpoint_store_env
    expected = env.store._host_identity
    env.hostfile.write_text("node-1 slots=1\n")
    with pytest.raises(env.module.CheckpointStoreError, match="hostfile changed") as error:
        env.store.create(env.root)
    assert env.calls == []
    message = str(error.value)
    assert f"path={env.hostfile}" in message
    changes = json.loads(message.split("differences=", 1)[1])
    observed = env.store._hostfile_identity()
    assert expected["sha256"] != observed["sha256"]
    assert changes == {
        key: {"expected": expected[key], "observed": observed[key]}
        for key in expected
        if expected[key] != observed[key]
    }


def test_native_checkpoint_store_same_content_replacement_reports_identity_drift(_native_checkpoint_store_env):
    env = _native_checkpoint_store_env
    replacement = env.hostfile.with_name("replacement-hostfile")
    replacement.write_bytes(env.hostfile.read_bytes())
    replacement.replace(env.hostfile)
    with pytest.raises(env.module.CheckpointStoreError, match="hostfile changed") as error:
        env.store.create(env.root)
    changes = json.loads(str(error.value).split("differences=", 1)[1])
    assert "sha256" not in changes
    assert changes["path_identity"]["expected"] != changes["path_identity"]["observed"]
    assert changes["target_identity"]["expected"] != changes["target_identity"]["observed"]
    assert env.calls == []


def test_native_checkpoint_store_hostfile_drift_after_fanout_retains_partial_root(
    _native_checkpoint_store_env, monkeypatch
):
    env = _native_checkpoint_store_env

    def changed(command, **kwargs):
        result = env.run(command, **kwargs)
        env.hostfile.write_text("node-0 slots=2\n")
        return result

    monkeypatch.setattr(env.module.subprocess, "run", changed)
    with pytest.raises(env.module.CheckpointStoreError, match="hostfile changed"):
        env.store.create(env.root)
    assert env.store.bindings and (env.root / env.module._MARKER).exists()
    with pytest.raises(env.module.CheckpointStoreError, match="hostfile changed"):
        env.store.close()
    assert env.root.exists()


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "foreign", "token", "identity", "nonzero"])
def test_native_checkpoint_store_requires_exact_structured_node_receipts(
    _native_checkpoint_store_env, monkeypatch, corruption
):
    env = _native_checkpoint_store_env

    def corrupted(command, **kwargs):
        result = env.run(command, **kwargs)
        prefix, _, raw = result.stdout.partition(env.module._RECEIPT)
        receipt = json.loads(raw)
        if corruption == "missing":
            result.stdout = ""
        elif corruption == "duplicate":
            result.stdout *= 2
        elif corruption == "foreign":
            result.stdout = f"foreign: {env.module._RECEIPT}{raw}"
        elif corruption == "nonzero":
            result.returncode = 1
        else:
            receipt[corruption] = None
            result.stdout = prefix + env.module._RECEIPT + json.dumps(receipt)
        return result

    monkeypatch.setattr(env.module.subprocess, "run", corrupted)
    with pytest.raises(env.module.CheckpointStoreError):
        env.store.create(env.root)
    assert len(env.calls) == 1
    assert env.root.exists()
    assert env.store.bindings[0]["path"] == str(env.root)


def test_native_checkpoint_store_rejects_missing_peer_without_local_fallback(_native_checkpoint_store_env):
    env = _native_checkpoint_store_env
    env.hostfile.write_text("node-0 slots=1\nnode-1 slots=1\n")
    store = env.module.NativeCheckpointStore(env.hostfile)
    with pytest.raises(env.module.CheckpointStoreError):
        store.create(env.root)
    assert len(env.calls) == 1


def test_native_checkpoint_store_rejects_same_executing_node_twice(_native_checkpoint_store_env, monkeypatch):
    env = _native_checkpoint_store_env
    env.hostfile.write_text("node-0 slots=1\nnode-1 slots=1\n")
    store = env.module.NativeCheckpointStore(env.hostfile)

    def duplicate_node(command, **kwargs):
        result = env.run(command, **kwargs)
        result.stdout += result.stdout.replace("node-0:", "node-1:")
        return result

    monkeypatch.setattr(env.module.subprocess, "run", duplicate_node)
    with pytest.raises(env.module.CheckpointStoreError, match="duplicate executing"):
        store.create(env.root)


def test_native_checkpoint_store_verify_rejects_changed_executing_host(_native_checkpoint_store_env, monkeypatch):
    env = _native_checkpoint_store_env
    binding = env.store.create(env.root)

    def changed_host(command, **kwargs):
        result = env.run(command, **kwargs)
        prefix, _, raw = result.stdout.partition(env.module._RECEIPT)
        receipt = json.loads(raw)
        receipt["host"] = "replacement-node"
        result.stdout = prefix + env.module._RECEIPT + json.dumps(receipt)
        return result

    monkeypatch.setattr(env.module.subprocess, "run", changed_host)
    with pytest.raises(env.module.CheckpointStoreError, match="node identity changed"):
        env.store.verify(binding)


def test_native_checkpoint_store_rejects_foreign_binding_and_closed_create(_native_checkpoint_store_env):
    env = _native_checkpoint_store_env
    binding = env.store.create(env.root)
    with pytest.raises(env.module.CheckpointStoreError, match="not owned"):
        env.store.verify({**binding, "path": str(env.fast / "foreign")})
    env.store.close()
    with pytest.raises(env.module.CheckpointStoreError, match="closed"):
        env.store.create(env.root)


def test_native_checkpoint_store_close_attempts_all_owned_bindings(_native_checkpoint_store_env, monkeypatch):
    env = _native_checkpoint_store_env
    first = env.store.create(env.root)
    second_root = env.fast / "second"
    second_root.mkdir()
    second = env.store.create(second_root)
    marker = env.root / env.module._MARKER
    marker.write_text("{}")
    with pytest.raises(env.module.CheckpointStoreError):
        env.store.close()
    assert Path(first["native_path"]).is_symlink() and env.root.exists()
    assert not Path(second["native_path"]).is_symlink() and not second_root.exists()
    assert env.store.bindings == (first,)


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf")])
def test_native_checkpoint_store_requires_bounded_timeout(_native_checkpoint_store_env, timeout):
    env = _native_checkpoint_store_env
    with pytest.raises(ValueError, match="finite and positive"):
        env.module.NativeCheckpointStore(env.hostfile, timeout=timeout)
