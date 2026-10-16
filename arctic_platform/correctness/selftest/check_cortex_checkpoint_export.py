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

"""Public checkpoint exports are bounded, complete and credential-free on disk."""

import copy
import hashlib
import json
import traceback
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests

from arctic_platform.correctness.harness import cortex_checkpoint_export as exporter

JOB_ID = "0abf38e6-80f6-49cc-820d-883c8c451df4"
CP_ID = "cp_dc1ee034-d625-4989-8164-092903a8d150"
SECRET = "DO-NOT-LOG-PRESIGNED-SECRET"


class Response:
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


class DownloadSession:
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


def _fixture(monkeypatch, *, dp_size=2, prefix="bf16_", reply=None, responses=None):
    names = [
        *(f"global_step10/{prefix}zero_pp_rank_{rank}_mp_rank_00_optim_states.pt" for rank in range(dp_size)),
        "global_step10/mp_rank_00_model_states.pt",
        "metadata.json",
    ]
    if reply is None:
        reply = {
            "checkpoint_id": CP_ID,
            "expires_at": "2026-10-14T08:00:00Z",
            "files": [
                {"filename": name, "size_bytes": "10", "url": f"https://storage.example/file{index}?sig={SECRET}"}
                for index, name in enumerate(names)
            ],
        }
    posted = []

    def send(*args, **kwargs):
        posted.append((args, kwargs))
        return copy.deepcopy(reply)

    transport = SimpleNamespace(job_id=JOB_ID, _prefix="https://cortex.example/jobs", _send=send)
    # Sentinel credentials live ONLY in the public transport, never the download session.
    transport.session = SimpleNamespace(headers={"Authorization": "Bearer DO-NOT-LEAK-PAT"})
    client = SimpleNamespace(transport=transport)
    if responses is None:
        responses = [Response() for _ in names]
    session = DownloadSession(responses)
    monkeypatch.setattr(exporter.requests, "Session", lambda: session)
    return client, reply, posted, session, responses


def _export(client, tmp_path, **kwargs):
    return exporter.export_checkpoint(client, JOB_ID, CP_ID, tmp_path, expected_dp_world_size=2, **kwargs)


@pytest.mark.parametrize("unsafe", ["/tmp", "/code", "/data-fast-shadow", "/data-fast", "symlink"])
def test_export_requires_data_fast_before_public_post(tmp_path, monkeypatch, unsafe):
    destination = Path(unsafe)
    if unsafe == "symlink":
        destination = tmp_path / "escape"
        destination.symlink_to("/tmp", target_is_directory=True)
        destination /= "output"
    client, _, posted, session, _ = _fixture(monkeypatch)
    monkeypatch.setattr(
        client.transport, "_send", lambda *args, **kwargs: pytest.fail("unsafe output reached public POST")
    )
    with pytest.raises(exporter.CheckpointExportError, match="/data-fast|symlink"):
        _export(client, destination)
    assert posted == []
    assert session.calls == []
    assert not list(tmp_path.glob("checkpoint-export-*"))


@pytest.mark.parametrize("prefix", ["", "bf16_"])
def test_export_preserves_public_post_and_streams_all_files_to_unique_receipted_root(tmp_path, monkeypatch, prefix):
    client, reply, posted, session, responses = _fixture(monkeypatch, prefix=prefix)
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("existing unrelated output")
    root = _export(client, tmp_path)
    assert root.parent == tmp_path
    assert root != tmp_path
    assert sentinel.read_text() == "existing unrelated output"
    assert posted == [(("POST", f"https://cortex.example/jobs/{JOB_ID}/checkpoints/{CP_ID}:export"), {})]
    assert len(session.calls) == 4
    assert all(response.closed for response in responses)
    assert session.closed
    receipt_text = (root / "export_receipt.json").read_text()
    receipt = json.loads(receipt_text)
    assert receipt["complete"] is True
    assert receipt["source_job_id"] == JOB_ID
    assert receipt["checkpoint_id"] == CP_ID
    assert receipt["dp_world_size"] == 2
    assert receipt["tag"] == "global_step10"
    assert receipt["total_bytes"] == 40
    assert {entry["path"] for entry in receipt["files"]} == {entry["filename"] for entry in reply["files"]}
    for item in receipt["files"]:
        assert (root / item["path"]).read_bytes() == b"checkpoint"
        assert item["sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
        assert item["size_bytes"] == 10
        assert set(item) == {"path", "size_bytes", "sha256"}
    assert SECRET not in receipt_text
    assert "url" not in receipt_text
    assert "DO-NOT-LEAK-PAT" not in receipt_text
    assert all("Authorization" not in kwargs["headers"] for _, kwargs in session.calls)


def test_each_export_allocates_new_directory_without_reusing_completed_snapshot(tmp_path, monkeypatch):
    client, *_ = _fixture(monkeypatch)
    first = _export(client, tmp_path)
    client, *_ = _fixture(monkeypatch)
    second = _export(client, tmp_path)
    assert first != second
    assert (first / "export_receipt.json").is_file()
    assert (second / "export_receipt.json").is_file()


def test_valid_export_creates_parent_only_after_manifest_validation(tmp_path, monkeypatch):
    destination = tmp_path / "nested" / "exports"
    client, reply, posted, _, _ = _fixture(monkeypatch)
    original_send = client.transport._send

    def send(*args, **kwargs):
        assert not destination.exists()
        return original_send(*args, **kwargs)

    monkeypatch.setattr(client.transport, "_send", send)
    root = _export(client, destination)
    assert root.parent == destination
    assert (root / "export_receipt.json").is_file()
    assert len(posted) == 1
    assert len(reply["files"]) == 4


def test_observed_four_rank_six_file_export_layout(tmp_path, monkeypatch):
    client, reply, _, session, _ = _fixture(monkeypatch, dp_size=4)
    root = exporter.export_checkpoint(client, JOB_ID, CP_ID, tmp_path, expected_dp_world_size=4)
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
def test_unsafe_filenames_rejected_before_directory_or_download(tmp_path, monkeypatch, name):
    _, reply, *_ = _fixture(monkeypatch)
    reply["files"][0]["filename"] = name
    client, _, _, session, _ = _fixture(monkeypatch, reply=reply)
    with pytest.raises(exporter.CheckpointExportError, match="unsafe export filename"):
        _export(client, tmp_path)
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
def test_incomplete_or_unsupported_layout_rejected_before_download(tmp_path, monkeypatch, mutation):
    _, reply, *_ = _fixture(monkeypatch)
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
    client, _, _, session, _ = _fixture(monkeypatch, reply=reply)
    with pytest.raises(exporter.CheckpointExportError):
        _export(client, tmp_path)
    assert not session.calls
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("size", [0, -1, True, None, 1.5, "1.5", "-1", "+10", " 10", "10 ", ""])
def test_invalid_declared_sizes_rejected(tmp_path, monkeypatch, size):
    _, reply, *_ = _fixture(monkeypatch)
    reply["files"][0]["size_bytes"] = size
    client, _, _, session, _ = _fixture(monkeypatch, reply=reply)
    with pytest.raises(exporter.CheckpointExportError, match="declared file size"):
        _export(client, tmp_path)
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
def test_invalid_urls_rejected_without_logging_values(tmp_path, monkeypatch, url):
    _, reply, *_ = _fixture(monkeypatch)
    reply["files"][0]["url"] = url
    client, _, _, session, _ = _fixture(monkeypatch, reply=reply)
    with pytest.raises(exporter.CheckpointExportError, match="HTTPS URL") as error:
        _export(client, tmp_path)
    assert "password" not in str(error.value)
    assert not session.calls
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("alias", ["download_url", "presigned_url"])
def test_undocumented_url_alias_not_guessed(tmp_path, monkeypatch, alias):
    _, reply, *_ = _fixture(monkeypatch)
    reply["files"][0][alias] = reply["files"][0].pop("url")
    client, _, _, session, _ = _fixture(monkeypatch, reply=reply)
    with pytest.raises(exporter.CheckpointExportError, match="HTTPS URL"):
        _export(client, tmp_path)
    assert not session.calls


def test_byte_budget_checked_before_download_and_exact_limit_accepted(tmp_path, monkeypatch):
    client, _, _, session, _ = _fixture(monkeypatch)
    with pytest.raises(exporter.CheckpointExportError, match="byte budget"):
        _export(client, tmp_path, max_total_bytes=39)
    assert not session.calls
    assert not list(tmp_path.iterdir())
    root = _export(client, tmp_path, max_total_bytes=40)
    assert (root / "export_receipt.json").is_file()


@pytest.mark.parametrize(
    "response",
    [
        Response(b"short"),
        Response(b"checkpoint-extra"),
        Response(status=206),
        Response(status=403),
        Response(headers={"Content-Length": "11"}),
        Response(headers={"Content-Length": "invalid"}),
        Response(headers={"Content-Encoding": "gzip"}),
        Response(chunks=[b"part", requests.ConnectionError(f"broken signed URL ?sig={SECRET}")]),
        Response(chunks=[requests.Timeout(f"request ?sig={SECRET}")]),
    ],
)
def test_failed_stream_never_leaves_receipt_or_partial_snapshot(tmp_path, monkeypatch, response):
    # Fail after one complete file, ensuring cleanup is not just first-file cleanup.
    responses = [Response(), response]
    client, _, _, session, _ = _fixture(monkeypatch, responses=responses)
    sentinel = tmp_path / "caller-owned"
    sentinel.mkdir()
    with pytest.raises(exporter.CheckpointExportError) as error:
        _export(client, tmp_path)
    assert SECRET not in "".join(traceback.format_exception(error.value))
    assert response.closed
    assert session.closed
    assert list(tmp_path.iterdir()) == [sentinel]


def test_content_length_matches_and_streamed_size_still_validated(tmp_path, monkeypatch):
    responses = [Response(headers={"Content-Length": "10"}) for _ in range(4)]
    client, *_ = _fixture(monkeypatch, responses=responses)
    assert (_export(client, tmp_path) / "export_receipt.json").is_file()
    assert all(len(response.read_calls) == 2 for response in responses)


def test_presigned_https_cross_host_redirects_followed_without_auth(tmp_path, monkeypatch):
    redirect = Response(status=307, headers={"Location": f"https://other.example/file?sig={SECRET}"})
    client, _, _, session, _ = _fixture(monkeypatch, responses=[redirect, *[Response() for _ in range(4)]])
    root = _export(client, tmp_path)
    assert redirect.closed
    assert len(session.calls) == 5
    assert session.calls[1][0] == f"https://other.example/file?sig={SECRET}"
    assert (root / "export_receipt.json").is_file()


@pytest.mark.parametrize("location", [None, "", f"http://other.example/file?sig={SECRET}", "file:///etc/passwd"])
def test_invalid_redirect_rejected_before_following(tmp_path, monkeypatch, location):
    redirect = Response(status=302, headers={"Location": location})
    client, _, _, session, _ = _fixture(monkeypatch, responses=[redirect])
    with pytest.raises(exporter.CheckpointExportError) as error:
        _export(client, tmp_path)
    assert SECRET not in "".join(traceback.format_exception(error.value))
    assert redirect.closed
    assert len(session.calls) == 1
    assert not list(tmp_path.iterdir())


def test_redirect_loop_is_bounded(tmp_path, monkeypatch):
    redirects = [Response(status=302, headers={"Location": "/same"}) for _ in range(6)]
    client, _, _, session, _ = _fixture(monkeypatch, responses=redirects)
    with pytest.raises(exporter.CheckpointExportError, match="redirect limit"):
        _export(client, tmp_path)
    assert len(session.calls) == 6
    assert all(response.closed for response in redirects)
    assert not list(tmp_path.iterdir())


def test_http_and_post_exception_messages_do_not_leak_secrets(tmp_path, monkeypatch):
    client, _, _, session, _ = _fixture(monkeypatch, responses=[requests.ConnectionError(SECRET)])
    with pytest.raises(exporter.CheckpointExportError) as error:
        _export(client, tmp_path)
    assert SECRET not in "".join(traceback.format_exception(error.value))
    assert session.closed
    assert not list(tmp_path.iterdir())
    client.transport._send = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError(SECRET))
    with pytest.raises(exporter.CheckpointExportError) as error:
        _export(client, tmp_path)
    assert SECRET not in "".join(traceback.format_exception(error.value))
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("timeout", [0, -1, True, None, float("inf"), float("nan")])
def test_invalid_timeouts_rejected_before_post(tmp_path, monkeypatch, timeout):
    client, _, posted, session, _ = _fixture(monkeypatch)
    with pytest.raises(exporter.CheckpointExportError, match="timeout"):
        _export(client, tmp_path, timeout_s=timeout)
    assert not posted
    assert not session.calls


def test_deadline_exceeded_mid_stream_cleans_partial_files(tmp_path, monkeypatch):
    client, _, _, session, responses = _fixture(monkeypatch)
    ticks = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(exporter.time, "monotonic", lambda: next(ticks))
    with pytest.raises(exporter.CheckpointExportError, match="deadline"):
        _export(client, tmp_path, timeout_s=1)
    assert responses[0].closed
    assert session.closed
    assert not list(tmp_path.iterdir())


def test_raw_reads_return_short_chunks_and_refresh_remaining_timeout(tmp_path, monkeypatch):
    clock = [0.0]
    responses = [Response(chunks=[bytes([byte]) for byte in b"checkpoint"]) for _ in range(4)]
    for response in responses:
        read1 = response.raw.read1

        def read(size, *, decode_content, read1=read1):
            clock[0] += 0.25
            return read1(size, decode_content=decode_content)

        response.raw.read1 = read
    monkeypatch.setattr(exporter.time, "monotonic", lambda: clock[0])
    client, _, _, _, _ = _fixture(monkeypatch, responses=responses)
    root = _export(client, tmp_path, timeout_s=20)
    assert (root / "export_receipt.json").is_file()
    timeouts = [timeout for response in responses for timeout in response.timeouts]
    assert len(timeouts) == 44
    assert timeouts == sorted(timeouts, reverse=True)
    assert timeouts[0] == 20
    assert timeouts[-1] == 9.25
    assert all(len(response.read_calls) == 11 for response in responses)


@pytest.mark.parametrize("connection", ["closed", "released"])
def test_connection_close_keeps_bounded_response_socket_available(tmp_path, monkeypatch, connection):
    responses = [Response(headers={"Connection": "close", "Content-Length": "10"}) for _ in range(4)]
    for response in responses:
        sock = response.raw.connection.sock
        response.raw._fp = SimpleNamespace(fp=SimpleNamespace(raw=SimpleNamespace(_sock=sock)))
        if connection == "closed":
            response.raw.connection.sock = None
        else:
            response.raw.connection = None
    client, *_ = _fixture(monkeypatch, responses=responses)
    assert (_export(client, tmp_path) / "export_receipt.json").is_file()
    assert all(len(response.timeouts) == len(response.read_calls) == 2 for response in responses)


def test_raw_read_requests_are_capped_even_for_large_files(tmp_path, monkeypatch):
    payload = b"x" * (exporter._CHUNK_BYTES + 7)
    responses = [Response(payload) for _ in range(4)]
    client, reply, *_ = _fixture(monkeypatch, responses=responses)
    for item in reply["files"]:
        item["size_bytes"] = len(payload)
    root = _export(client, tmp_path)
    for response in responses:
        assert max(response.read_calls) == exporter._CHUNK_BYTES
        assert response.read_calls[-1] == 1
    assert all(
        item["size_bytes"] == len(payload) for item in json.loads((root / "export_receipt.json").read_text())["files"]
    )


def test_slow_trickle_stops_at_deadline_without_waiting_for_full_chunk(tmp_path, monkeypatch):
    clock = [0.0]
    response = Response(chunks=[b"x"] * 10)
    read1 = response.raw.read1

    def read(size, *, decode_content):
        clock[0] += 0.4
        return read1(size, decode_content=decode_content)

    response.raw.read1 = read
    monkeypatch.setattr(exporter.time, "monotonic", lambda: clock[0])
    client, _, _, session, _ = _fixture(monkeypatch, responses=[response])
    with pytest.raises(exporter.CheckpointExportError, match="deadline"):
        _export(client, tmp_path, timeout_s=1)
    assert len(response.read_calls) == 3
    assert response.timeouts == pytest.approx([1.0, 0.6, 0.2])
    assert response.closed and session.closed
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("missing", ["raw", "read1", "connection", "socket", "settimeout"])
def test_unbounded_raw_interfaces_fail_closed(tmp_path, monkeypatch, missing):
    response = Response()
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
    client, _, _, session, _ = _fixture(monkeypatch, responses=[response])
    with pytest.raises(exporter.CheckpointExportError, match="bounded raw"):
        _export(client, tmp_path)
    assert response.closed and session.closed
    assert not response.read_calls
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("declaration", ["header", "raw"])
def test_chunked_framing_fails_closed_before_reads(tmp_path, monkeypatch, declaration):
    response = Response()
    if declaration == "header":
        response.headers["Transfer-Encoding"] = "chunked"
    else:
        response.raw.chunked = True
    client, _, _, session, _ = _fixture(monkeypatch, responses=[response])
    with pytest.raises(exporter.CheckpointExportError, match="chunked"):
        _export(client, tmp_path)
    assert response.closed and session.closed
    assert not response.read_calls
    assert not list(tmp_path.iterdir())


def test_raw_empty_read_is_eof_not_an_interim_chunk(tmp_path, monkeypatch):
    response = Response(chunks=[b"check", b"", b"point"])
    client, _, _, _, _ = _fixture(monkeypatch, responses=[response])
    with pytest.raises(exporter.CheckpointExportError, match="truncated"):
        _export(client, tmp_path)
    assert len(response.read_calls) == 2
    assert response.closed
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("value", [None, "checkpoint", b"x" * 12])
def test_invalid_raw_read_output_fails_closed(tmp_path, monkeypatch, value):
    response = Response()
    response.raw.read1 = lambda *args, **kwargs: value
    client, _, _, _, _ = _fixture(monkeypatch, responses=[response])
    with pytest.raises(exporter.CheckpointExportError, match="bounded raw bytes"):
        _export(client, tmp_path)
    assert response.closed
    assert not list(tmp_path.iterdir())


def test_socket_timeout_failure_is_sanitized_and_cleans_snapshot(tmp_path, monkeypatch):
    response = Response()

    def settimeout(timeout):
        raise OSError(SECRET)

    response.raw.connection.sock.settimeout = settimeout
    client, _, _, session, _ = _fixture(monkeypatch, responses=[response])
    with pytest.raises(exporter.CheckpointExportError) as error:
        _export(client, tmp_path)
    assert SECRET not in "".join(traceback.format_exception(error.value))
    assert response.closed and session.closed
    assert not list(tmp_path.iterdir())


def test_interrupted_download_cleans_its_owned_snapshot(tmp_path, monkeypatch):
    response = Response(chunks=[b"part", KeyboardInterrupt()])
    client, _, _, session, _ = _fixture(monkeypatch, responses=[response])
    with pytest.raises(KeyboardInterrupt):
        _export(client, tmp_path)
    assert response.closed
    assert session.closed
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("value", [0, -1, True, None, 1.5, "2"])
@pytest.mark.parametrize("field", ["expected_dp_world_size", "max_total_bytes"])
def test_invalid_caller_budgets_rejected_before_post(tmp_path, monkeypatch, value, field):
    client, _, posted, session, _ = _fixture(monkeypatch)
    kwargs = {"expected_dp_world_size": 2, field: value}
    with pytest.raises(exporter.CheckpointExportError, match="positive integer"):
        exporter.export_checkpoint(client, JOB_ID, CP_ID, tmp_path, **kwargs)
    assert not posted
    assert not session.calls


@pytest.mark.parametrize("reply", [None, [], {"checkpoint_id": CP_ID}, {"checkpoint_id": CP_ID, "files": {}}])
def test_malformed_export_envelope_rejected_without_download(tmp_path, monkeypatch, reply):
    client, _, _, session, _ = _fixture(monkeypatch)
    client.transport._send = lambda *args, **kwargs: reply
    with pytest.raises(exporter.CheckpointExportError):
        _export(client, tmp_path / "nested" / "exports")
    assert not session.calls
    assert not list(tmp_path.iterdir())


def test_symlink_destination_rejected_without_touching_target(tmp_path, monkeypatch):
    client, _, posted, session, _ = _fixture(monkeypatch)
    actual = tmp_path / "actual"
    actual.mkdir()
    link = tmp_path / "link"
    link.symlink_to(actual, target_is_directory=True)
    with pytest.raises(exporter.CheckpointExportError, match="symlink"):
        _export(client, link / "child")
    assert not posted
    assert not session.calls
    assert not list(actual.iterdir())


def test_injected_tag_symlink_cannot_redirect_writes(tmp_path, monkeypatch):
    client, _, _, _, _ = _fixture(monkeypatch)
    outside = tmp_path / "outside"
    outside.mkdir()
    real_mkdtemp = exporter.tempfile.mkdtemp

    def mkdtemp(**kwargs):
        root = Path(real_mkdtemp(**kwargs))
        (root / "global_step10").symlink_to(outside, target_is_directory=True)
        return str(root)

    monkeypatch.setattr(exporter.tempfile, "mkdtemp", mkdtemp)
    with pytest.raises(exporter.CheckpointExportError):
        _export(client, tmp_path)
    assert not list(outside.iterdir())
    assert list(tmp_path.iterdir()) == [outside]


def test_existing_file_or_symlink_never_overwritten(tmp_path):
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


def test_receipt_write_failure_cleans_every_downloaded_file(tmp_path, monkeypatch):
    client, _, _, session, _ = _fixture(monkeypatch)
    real_open = exporter._open_exclusive

    def open_exclusive(root, name):
        if name == "export_receipt.json":
            raise OSError(SECRET)
        return real_open(root, name)

    monkeypatch.setattr(exporter, "_open_exclusive", open_exclusive)
    with pytest.raises(exporter.CheckpointExportError) as error:
        _export(client, tmp_path)
    assert SECRET not in "".join(traceback.format_exception(error.value))
    assert session.closed
    assert not list(tmp_path.iterdir())


def test_job_identity_and_unsafe_endpoint_ids_rejected_before_post(tmp_path, monkeypatch):
    client, _, posted, session, _ = _fixture(monkeypatch)
    with pytest.raises(exporter.CheckpointExportError, match="parent job"):
        exporter.export_checkpoint(client, "other-job", CP_ID, tmp_path, expected_dp_world_size=2)
    with pytest.raises(exporter.CheckpointExportError, match="checkpoint ID"):
        exporter.export_checkpoint(client, JOB_ID, "../cp", tmp_path, expected_dp_world_size=2)
    assert not posted
    assert not session.calls
