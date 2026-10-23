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
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import requests
import torch

from arctic_platform.client.requests import export_checkpoint_request
from arctic_platform.client.transports.cortex import _route
from arctic_platform.correctness.checks import checkpoint_resume
from arctic_platform.correctness.checks.fwd_bwd_step import _compare_arm
from arctic_platform.correctness.checks.optimizer_state import compare_optimizer_artifacts
from arctic_platform.correctness.checks.optimizer_state import max_pairwise_optimizer_delta
from arctic_platform.correctness.checks.optimizer_state import summarize_optimizer_artifacts
from arctic_platform.correctness.harness import dss_driver
from arctic_platform.correctness.harness import inference_export as exporter
from arctic_platform.correctness.harness.config import TestTolerance as Tolerance
from arctic_platform.correctness.harness.registry import TestOutcome as Outcome
from arctic_platform.correctness.harness.runner import OPTIMIZER_LEARNING_RATE
from arctic_platform.correctness.harness.transport import Checkpoint

# Checkpoint resume.


_checkpoint_resume_CONTROL = checkpoint_resume.CHECKPOINT_ITERATION

_checkpoint_resume_FINAL = checkpoint_resume.ITERATIONS


class _checkpoint_resume_Gate:

    def tolerance_for(self, test_id: str) -> float:
        assert test_id == checkpoint_resume.TEST_ID
        return 0.001


def _checkpoint_resume_verdict(reference: dict, target: dict):
    return checkpoint_resume._verdict(
        SimpleNamespace(config_id="example", spec=_checkpoint_resume_Gate()),
        SimpleNamespace(name="gas1"),
        reference,
        target,
    )


def test_checkpoint_resume_identical_losses_pass() -> None:
    result = _checkpoint_resume_verdict(
        {_checkpoint_resume_CONTROL: 2.5, _checkpoint_resume_FINAL: 1.75},
        {_checkpoint_resume_CONTROL: 2.5, _checkpoint_resume_FINAL: 1.75},
    )
    assert result.outcome is Outcome.PASS
    assert result.mismatches == []
    assert result.metrics["max_abs_diff"] == 0.0
    assert result.metrics["stated_criterion_abs"] == 0.001


def test_checkpoint_resume_a_final_loss_inside_the_tolerance_passes() -> None:
    result = _checkpoint_resume_verdict(
        {_checkpoint_resume_CONTROL: 2.5, _checkpoint_resume_FINAL: 1.75},
        {_checkpoint_resume_CONTROL: 2.5, _checkpoint_resume_FINAL: 1.7505},
    )
    assert result.outcome is Outcome.PASS
    assert result.metrics[f"target_loss_iteration_{_checkpoint_resume_FINAL}"] == 1.7505


def test_checkpoint_resume_a_final_loss_outside_the_tolerance_fails() -> None:
    """The resume verdict: ten steps run from restored weights, optimizer moments and scheduler state."""
    result = _checkpoint_resume_verdict(
        {_checkpoint_resume_CONTROL: 2.5, _checkpoint_resume_FINAL: 1.75},
        {_checkpoint_resume_CONTROL: 2.5, _checkpoint_resume_FINAL: 1.753},
    )
    assert result.outcome is Outcome.FAIL
    assert [mismatch.name for mismatch in result.mismatches] == [f"loss at iteration {_checkpoint_resume_FINAL}"]
    assert result.worst_name == f"loss at iteration {_checkpoint_resume_FINAL}"
    assert result.metrics[f"reference_loss_iteration_{_checkpoint_resume_FINAL}"] == 1.75


def test_checkpoint_resume_a_control_disagreement_is_explained_rather_than_blamed_on_the_resume() -> None:
    """Before the checkpoint the two runs are one computation, so a difference there does not measure the resume."""
    result = _checkpoint_resume_verdict(
        {_checkpoint_resume_CONTROL: 2.5, _checkpoint_resume_FINAL: 1.75},
        {_checkpoint_resume_CONTROL: 2.6, _checkpoint_resume_FINAL: 1.75},
    )
    assert result.outcome is Outcome.FAIL
    assert [mismatch.name for mismatch in result.mismatches] == [f"loss at iteration {_checkpoint_resume_CONTROL}"]
    assert str(_checkpoint_resume_CONTROL) in result.reason
    assert (
        _checkpoint_resume_verdict(
            {_checkpoint_resume_CONTROL: 2.5, _checkpoint_resume_FINAL: 1.75},
            {_checkpoint_resume_CONTROL: 2.5, _checkpoint_resume_FINAL: 1.9},
        ).reason
        == ""
    )


def test_checkpoint_resume_a_resumed_arctic_job_loads_the_saved_path_after_creation(tmp_path, monkeypatch) -> None:
    calls = []

    class Client:

        def load_checkpoint(self, *, path):
            calls.append(("load", path))

    @contextlib.contextmanager
    def running_job(session, payload):
        calls.append(("create", payload["model_name"]))
        yield SimpleNamespace(client=Client())

    monkeypatch.setattr(dss_driver, "running_job", running_job)
    transport = dss_driver.GatewayTransport(tmp_path, {"n_gpus": 1}, "/models/reduced", 1234, slots=1)
    checkpoint = Checkpoint(reference="/checkpoints/global_step10", source_job="/checkpoints/global_step10")
    with transport.job(resume_from=checkpoint):
        pass
    assert calls == [("create", "/models/reduced"), ("load", "/checkpoints/global_step10")]


def test_checkpoint_resume_arctic_checkpoint_reference_is_the_saved_path(tmp_path) -> None:

    class Client:

        def save_checkpoint(self):
            return {"path": "/checkpoints/global_step10", "global_step": 10}

    wrapped = dss_driver.GatewayTrainingJob(dss_driver.GatewaySession(tmp_path, 1), SimpleNamespace(client=Client()))
    checkpoint = wrapped.save_resumable()
    assert checkpoint.reference == "/checkpoints/global_step10"
    assert checkpoint.source_job == "/checkpoints/global_step10"


def test_checkpoint_resume_every_iteration_gets_its_own_batch() -> None:
    """A run that replayed one batch instead of advancing would reach iteration 20 on the wrong data."""
    batches = checkpoint_resume.iteration_batches(
        SimpleNamespace(vocab_size=128), SimpleNamespace(name="gas1", global_batch_size=2, max_seq_len=64)
    )
    assert len(batches) == _checkpoint_resume_FINAL
    rows = {tuple(batch.input_ids.flatten().tolist()) for batch in batches}
    assert len(rows) == _checkpoint_resume_FINAL


class _checkpoint_resume_Job:
    """A job that answers every request with a loss decided by the request's own bytes.

    Two jobs sent the same body therefore report the same loss, so a run whose halves consumed the
    intended batches agrees at both recorded iterations and one that replayed or skipped a batch does not.
    """

    def __init__(self, resume_from) -> None:
        self.resume_from = resume_from
        self.bodies: list = []
        self.rates: list = []
        self.saved = None

    def fwd_bwd_step(self, body: bytes, *, learning_rate: float) -> float:
        self.bodies.append(body)
        self.rates.append(learning_rate)
        batches = body["batch"] if isinstance(body["batch"], list) else [body["batch"]]
        tokens = torch.cat([batch["input_ids"] for batch in batches]).detach().cpu().numpy().tobytes()
        return float(zlib.crc32(tokens))

    def save_resumable(self) -> Checkpoint:
        self.saved = Checkpoint(reference="cp_saved", source_job="job-1")
        return self.saved


class _checkpoint_resume_Transport:
    """Records every job a check creates, in the order it created them."""

    def __init__(self) -> None:
        self.jobs: list = []

    def __enter__(self) -> "_checkpoint_resume_Transport":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        return None

    @contextlib.contextmanager
    def job(self, *, resume_from=None):
        created = _checkpoint_resume_Job(resume_from)
        self.jobs.append(created)
        yield created


def _checkpoint_resume_context(transport, tmp_path):
    return SimpleNamespace(
        config_id="example",
        workdir=tmp_path,
        vocab_size=128,
        cfg=SimpleNamespace(training={}),
        spec=_checkpoint_resume_Gate(),
        arms=[SimpleNamespace(name="gas1", global_batch_size=2, max_seq_len=64, dss_microbatches=1)],
        transport_for=lambda workdir: transport,
    )


def test_checkpoint_resume_the_resumed_job_continues_the_data_order_at_batch_11(tmp_path) -> None:
    """Three jobs: twenty iterations uninterrupted, then ten, then iterations 11 to 20 from the checkpoint."""
    transport = _checkpoint_resume_Transport()
    ctx = _checkpoint_resume_context(transport, tmp_path)
    results = checkpoint_resume.run(ctx)
    bodies = checkpoint_resume._iteration_bodies(ctx, ctx.arms[0])
    uninterrupted, first_half, resumed = transport.jobs

    def body_tokens(values):
        return [body["batch"]["input_ids"].tolist() for body in values]

    assert body_tokens(uninterrupted.bodies) == body_tokens(bodies)
    assert uninterrupted.resume_from is None
    assert body_tokens(first_half.bodies) == body_tokens(bodies[:_checkpoint_resume_CONTROL])
    assert body_tokens(resumed.bodies) == body_tokens(bodies[_checkpoint_resume_CONTROL:])
    assert len(resumed.bodies) == _checkpoint_resume_FINAL - _checkpoint_resume_CONTROL
    assert resumed.resume_from == first_half.saved
    assert [result.outcome for result in results] == [Outcome.PASS]


def test_checkpoint_resume_every_iteration_steps_at_the_check_s_own_learning_rate(tmp_path) -> None:
    transport = _checkpoint_resume_Transport()
    checkpoint_resume.run(_checkpoint_resume_context(transport, tmp_path))
    rates = {rate for job in transport.jobs for rate in job.rates}
    assert rates == {checkpoint_resume.LEARNING_RATE}


def test_checkpoint_resume_gas4_uses_identical_plain_requests_for_uninterrupted_and_resumed_jobs(tmp_path) -> None:
    transport = _checkpoint_resume_Transport()
    ctx = _checkpoint_resume_context(transport, tmp_path)
    ctx.config_id = "qwen3-8b-h200-train-sft-full-4gpus-64k"
    ctx.arms = [SimpleNamespace(name="gas4", global_batch_size=8, max_seq_len=64, dss_microbatches=4)]
    calls = []

    def transport_for(workdir):
        calls.append(workdir)
        return transport

    ctx.transport_for = transport_for
    results = checkpoint_resume.run(ctx)
    assert calls == [tmp_path / checkpoint_resume.TEST_ID / "gas4"]
    assert [result.outcome for result in results] == [Outcome.PASS]
    bodies = checkpoint_resume._iteration_bodies(ctx, ctx.arms[0])
    expected = checkpoint_resume.iteration_batches(ctx, ctx.arms[0])
    for body, batch in zip(bodies, expected):
        assert isinstance(body["batch"], dict)
        assert body["batch"]["input_ids"].shape == (8, 64)
        assert torch.equal(body["batch"]["input_ids"], batch.input_ids)
        assert torch.equal(body["batch"]["labels"], batch.labels)
        assert body["processing"] == {"loss_fn": "sft"}
    uninterrupted, first_half, resumed = transport.jobs
    for full_body, split_body in zip(uninterrupted.bodies, first_half.bodies + resumed.bodies):
        assert torch.equal(full_body["batch"]["input_ids"], split_body["batch"]["input_ids"])
    assert resumed.resume_from == first_half.saved


def test_checkpoint_resume_gateway_transport_preserves_config_accumulation_for_all_three_jobs(
    tmp_path, monkeypatch
) -> None:
    payloads = []

    @contextlib.contextmanager
    def running_job(session, payload):
        payloads.append(payload)
        yield SimpleNamespace(client=SimpleNamespace(load_checkpoint=lambda **kwargs: None))

    monkeypatch.setattr(dss_driver, "running_job", running_job)
    transport = dss_driver.GatewayTransport(
        tmp_path,
        {
            "n_gpus": 4,
            "ds_config": {
                "train_micro_batch_size_per_gpu": 1,
                "train_batch_size": 4,
                "gradient_accumulation_steps": 1,
            },
        },
        "/models/reduced",
        1234,
        slots=4,
    )
    for checkpoint in (None, None, Checkpoint(reference="/saved/step10", source_job="/saved/step10")):
        with transport.job(resume_from=checkpoint):
            pass
    assert len(payloads) == 3
    assert all((payload["training_config"]["ds_config"]["gradient_accumulation_steps"] == 1 for payload in payloads))
    assert all((payload["training_config"]["ds_config"]["train_batch_size"] == 4 for payload in payloads))


def test_checkpoint_resume_a_case_that_raises_fails_that_case_alone(tmp_path) -> None:

    class Refusing(_checkpoint_resume_Transport):

        @contextlib.contextmanager
        def job(self, *, resume_from=None):
            raise RuntimeError("no capacity")
            yield

    results = checkpoint_resume.run(_checkpoint_resume_context(Refusing(), tmp_path))
    assert [result.outcome for result in results] == [Outcome.FAIL]
    assert results[0].summary == "RuntimeError: no capacity"


def test_checkpoint_resume_loss_range_gate_is_twice_the_range_rounded_up() -> None:
    gate = checkpoint_resume.loss_range_gate(0.1001, "gas1 loss at iteration 10")
    assert gate.calibration_runs == 3
    assert gate.multiplier == 2.0
    assert gate.computed_gate == 0.2002
    assert gate.absolute == 0.201
    assert gate.worst_tensor == "gas1 loss at iteration 10"


def test_checkpoint_resume_a_zero_loss_range_still_meets_the_minimum_gate() -> None:
    gate = checkpoint_resume.loss_range_gate(0.0, "gas1 loss at iteration 1")
    assert gate.absolute == 0.001


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
