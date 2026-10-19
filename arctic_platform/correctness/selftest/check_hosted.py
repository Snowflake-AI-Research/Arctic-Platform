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
from copy import deepcopy
from pathlib import Path

import pytest

from arctic_platform.client import CortexConfig
from arctic_platform.client.transport import JobHandles
from arctic_platform.correctness.harness import hosted
from arctic_platform.correctness.harness.config import load_config
from arctic_platform.correctness.harness.hosted import HostedConnection
from arctic_platform.correctness.harness.hosted import HostedTrainingJob
from arctic_platform.correctness.harness.hosted import HostedTransport
from arctic_platform.correctness.harness.hosted import is_local_cortex_gateway
from arctic_platform.correctness.harness.hosted import require_local_cortex_gateway
from arctic_platform.correctness.harness.hosted import save_and_resolve_checkpoint_id
from arctic_platform.correctness.harness.hosted import save_local_checkpoint
from arctic_platform.correctness.harness.hosted import training_sub_job
from arctic_platform.correctness.harness.transport import Checkpoint

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "qwen3-8b" / "h200" / "train-sft-full-4gpus-64k.config"
CONNECTION = HostedConnection(host="server", database="DB", schema="SCHEMA", pat="not-a-token")
LOCAL_CHECKPOINT_ID = "cp_00000000-0000-0000-0000-000000000001"
LOCAL_STAGE = {
    "location_type": "S3",
    "bucket": "synthetic-control-bucket",
    "prefix": f"jobs/saving-parent/checkpoints/{LOCAL_CHECKPOINT_ID}/",
}


def _config():
    return load_config(CONFIG)


def _wire(cfg, **kwargs) -> dict:
    return training_sub_job(cfg, **kwargs)


class _Client:
    """Enough of the hosted client to record what the transport asked it to do."""

    def __init__(self, *, losses=(), save_result=None, catalog=None, added=()) -> None:
        self.losses = list(losses)
        self.save_result = {} if save_result is None else dict(save_result)
        self.catalog = [dict(entry) for entry in catalog or []]
        self.added = [dict(entry) for entry in added]
        self.calls: list = []
        self.created: list = []
        self.cancelled: list = []
        self.closed = False
        self.transport = self
        self.jobs = JobHandles(training="job-1:training:0")

    def create(self, config):
        self.config = config
        self.created.append((config.to_cortex()[0], config.hardware))
        self.jobs = JobHandles(training=f"job-{len(self.created)}:training:0")
        return self

    def fwd_bwd(self, body):
        self.calls.append(("fwd-bwd", body))
        return {"avg_loss": self.losses.pop(0) if self.losses else 0.0}

    def step(self, learning_rate=None):
        self.calls.append(("step", learning_rate))
        return {"global_steps": len(self.calls)}

    def save_checkpoint(self, checkpoint_type=None, *, path=None):
        self.calls.append(("save", checkpoint_type) if path is None else ("save", checkpoint_type, path))
        self.catalog = self.catalog + self.added
        return self.save_result

    def load_checkpoint(self, *, path):
        self.calls.append(("load", path))
        return {"checkpoint_path": path, "checkpoint_tag": "global_step10", "version": 10}

    def list_checkpoints(self):
        return [dict(entry) for entry in self.catalog]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cancelled.append(str(self.jobs.training).split(":", 1)[0])
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
        if key in {"optimizer", "debug"}:
            continue
        assert wire["training_config"][key] == value, key
    for key, value in cfg.training["optimizer"].items():
        assert wire["training_config"]["optimizer"]["name" if key == "type" else key] == value, key


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
    assert ("save", "resumable") in client.calls


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

    body = {"args": (), "kwargs": {}}
    loss = HostedTrainingJob(client, "job-1").fwd_bwd_step(body, learning_rate=1e-3)

    assert loss == 2.5
    assert client.calls == [("fwd-bwd", body), ("step", 1e-3)]


def test_a_forward_backward_that_reports_no_loss_is_an_error() -> None:
    class Silent(_Client):
        def fwd_bwd(self, body):
            return {"metrics": {}}

    with pytest.raises(RuntimeError, match="no avg_loss"):
        HostedTrainingJob(Silent(), "job-1").fwd_bwd_step({}, learning_rate=1e-3)


def test_a_job_releases_its_gpus_when_its_block_ends(monkeypatch) -> None:
    client = _Client()
    monkeypatch.setattr(hosted, "ArcticClient", client.create)

    with HostedTransport(CONNECTION, _config()) as transport:
        with transport.job() as job:
            assert job.job_id == "job-1"
            assert client.cancelled == []
        assert client.cancelled == ["job-1"]
    assert client.closed


def test_a_job_releases_its_gpus_when_the_block_raises(monkeypatch) -> None:
    client = _Client()
    monkeypatch.setattr(hosted, "ArcticClient", client.create)

    with pytest.raises(ZeroDivisionError):
        with HostedTransport(CONNECTION, _config()) as transport:
            with transport.job():
                raise ZeroDivisionError

    assert client.cancelled == ["job-1"]


def test_a_resumed_job_is_created_from_the_checkpoint_it_was_given(monkeypatch) -> None:
    client = _Client()
    monkeypatch.setattr(hosted, "ArcticClient", client.create)

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


def test_explicit_checkpoint_id_does_not_use_a_deepspeed_tag() -> None:
    client = _Client(save_result={"checkpoint_id": "cp_new", "checkpoint_tag": "global_step10"})
    assert save_and_resolve_checkpoint_id(client, "job-1", "resumable") == "cp_new"


def test_an_ambiguous_catalog_delta_is_refused() -> None:
    client = _Client(added=[{"checkpoint_id": "cp_one"}, {"checkpoint_id": "cp_two"}])
    with pytest.raises(RuntimeError, match="cannot be identified"):
        save_and_resolve_checkpoint_id(client, "job-1", "resumable")


def test_checkpoint_jobs_preserve_dss_engine_gas_and_width(monkeypatch) -> None:
    client = _Client()
    monkeypatch.setattr(hosted, "ArcticClient", client.create)
    cfg = _config().at_gpu_width(8)
    with HostedTransport(CONNECTION, cfg) as transport:
        with transport.job():
            pass
        with transport.job(resume_from=Checkpoint("cp_new", "job-1")):
            pass
    assert len(client.created) == 2
    for wire, hardware in client.created:
        assert hardware == "H200"
        assert wire["training_config"]["n_gpus"] == 8
        assert wire["training_config"]["ds_config"]["gradient_accumulation_steps"] == 1
    assert client.cancelled == ["job-1", "job-2"]


def _local_backend(url="http://127.0.0.1:8003", **kwargs):
    return CortexConfig(base_url=url, host=None, pat=None, **kwargs)


@pytest.mark.parametrize("url", ["http://127.0.0.1:8003", "http://localhost:8003/", "http://[::1]:8003"])
def test_local_gateway_accepts_only_explicit_http_loopback(url) -> None:
    backend = _local_backend(url)
    assert is_local_cortex_gateway(backend)
    require_local_cortex_gateway(backend)


@pytest.mark.parametrize(
    "url",
    [
        "https://localhost:8003",
        "http://localhost.example:8003",
        "http://127.0.0.2:8003",
        "http://0.0.0.0:8003",
        "http://10.0.0.1:8003",
        "http://user:secret@localhost:8003",
        "http://localhost:8003/api",
        "http://localhost:8003/?value=secret",
        "http://localhost:8003/#secret",
    ],
)
def test_local_gateway_rejects_nonloopback_and_decorated_urls(url) -> None:
    with pytest.raises(ValueError, match="unauthenticated HTTP loopback") as exc:
        require_local_cortex_gateway(_local_backend(url))
    assert "secret" not in str(exc.value)


@pytest.mark.parametrize("options", [{"host": "server"}, {"pat": "not-a-token"}])
def test_local_gateway_refuses_hosted_credentials(options) -> None:
    backend = CortexConfig(base_url="http://127.0.0.1:8003", **options)
    with pytest.raises(ValueError, match="unauthenticated HTTP loopback") as exc:
        require_local_cortex_gateway(backend)
    assert "not-a-token" not in str(exc.value)


class _LocalClient(_Client):
    def list_checkpoints(self):
        raise AssertionError("the supplied local gateway has no checkpoint catalog")

    def load_checkpoint(self, checkpoint_id, *, stage_info=None):
        self.calls.append(("load", checkpoint_id, deepcopy(stage_info)))
        return hosted._local_load_metadata(
            {"checkpoint_id": checkpoint_id, "checkpoint_tag": "global_step10", "version": 10},
            checkpoint_id,
        )


def _local_checkpoint(root):
    return hosted.LocalCortexCheckpoint(
        LOCAL_CHECKPOINT_ID,
        "owned-source",
        str(root),
        "global_step10",
        10,
        stage_info=deepcopy(LOCAL_STAGE),
    )


@pytest.mark.parametrize("checkpoint_type", ["resumable", "weights-only"])
def test_local_save_uses_a_fresh_fast_path_without_catalog(monkeypatch, tmp_path, checkpoint_type) -> None:
    root = tmp_path / "retained-checkpoint"
    root.mkdir()
    marker = root / "retained-evidence"
    marker.write_text("preserve", encoding="ascii")
    monkeypatch.setattr(hosted, "correctness_workdir", lambda prefix: root)
    client = _LocalClient(save_result={"checkpoint_path": str(root), "checkpoint_tag": "global_step10", "version": 10})

    assert save_local_checkpoint(client, checkpoint_type) == root.resolve()
    assert client.calls == [("save", checkpoint_type, str(root))]
    assert marker.read_text(encoding="ascii") == "preserve"


def test_local_resumable_checkpoint_keeps_the_saving_job_identity(monkeypatch, tmp_path) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    monkeypatch.setattr(hosted, "correctness_workdir", lambda prefix: root)
    client = _LocalClient(
        save_result={
            "checkpoint_path": str(root),
            "checkpoint_tag": "global_step10",
            "version": 10,
            "checkpoint_id": LOCAL_CHECKPOINT_ID,
            "stage_info": deepcopy(LOCAL_STAGE),
        }
    )
    checkpoint = HostedTrainingJob(client, "owned-parent", local_gateway=True).save_resumable()
    assert checkpoint == hosted.LocalCortexCheckpoint(
        LOCAL_CHECKPOINT_ID,
        "owned-parent",
        str(root),
        "global_step10",
        10,
        stage_info=LOCAL_STAGE,
    )
    assert checkpoint.stage_info is not client.save_result["stage_info"]
    assert client.calls == [("save", "resumable", str(root))]


def test_local_resumable_save_without_durable_identity_retains_output_and_releases_job(monkeypatch, tmp_path) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    marker = root / "retained-evidence"
    marker.write_text("preserve", encoding="ascii")
    monkeypatch.setattr(hosted, "correctness_workdir", lambda prefix: root)
    client = _LocalClient(
        save_result={
            "checkpoint_path": str(root),
            "checkpoint_tag": "global_step10",
            "version": 10,
        }
    )
    monkeypatch.setattr(hosted, "ArcticClient", client.create)
    with pytest.raises(RuntimeError, match="only a FAST path/tag"):
        with HostedTransport(_local_backend(), _config()) as transport:
            with transport.job() as job:
                job.save_resumable()
    assert client.calls == [("save", "resumable", str(root))]
    assert client.cancelled == ["job-1"] and client.closed
    assert marker.read_text(encoding="ascii") == "preserve"


@pytest.mark.parametrize(
    "changes",
    [
        {"checkpoint_path": None},
        {"checkpoint_path": "relative/checkpoint"},
        {"checkpoint_path": "/tmp/checkpoint"},
        {"checkpoint_path": "/data-fast"},
        {"checkpoint_path": "/data-fast/tmp/not-this-save"},
        {"checkpoint_tag": None},
        {"checkpoint_tag": "cp_123"},
        {"checkpoint_tag": "global_step-1"},
        {"checkpoint_tag": "global_step010"},
        {"checkpoint_tag": "../global_step10"},
        {"version": None},
        {"version": "10"},
        {"version": 10.0},
        {"version": True},
        {"version": 11},
    ],
)
def test_local_save_rejects_inconsistent_metadata_and_retains_output(monkeypatch, tmp_path, changes) -> None:
    root = tmp_path / "checkpoint"
    root.mkdir()
    marker = root / "retained-evidence"
    marker.write_text("preserve", encoding="ascii")
    monkeypatch.setattr(hosted, "correctness_workdir", lambda prefix: root)
    result = {"checkpoint_path": str(root), "checkpoint_tag": "global_step10", "version": 10, **changes}
    with pytest.raises((ValueError, RuntimeError)):
        save_local_checkpoint(_LocalClient(save_result=result), "resumable")
    assert marker.read_text(encoding="ascii") == "preserve"


@pytest.mark.parametrize("result", [None, [], "global_step10"])
def test_local_save_rejects_nonmapping_response(monkeypatch, tmp_path, result) -> None:
    monkeypatch.setattr(hosted, "correctness_workdir", lambda prefix: tmp_path)
    client = _LocalClient()
    client.save_result = result
    with pytest.raises(RuntimeError, match="no checkpoint metadata"):
        save_local_checkpoint(client, "resumable")


def test_local_save_rejects_invalid_checkpoint_type_before_creating_storage(monkeypatch) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid type must not allocate storage")

    monkeypatch.setattr(hosted, "correctness_workdir", unexpected)
    with pytest.raises(ValueError, match="checkpoint_type"):
        save_local_checkpoint(_LocalClient(), "unknown")


def test_local_resume_creates_a_fresh_full_model_job_then_loads_before_training(monkeypatch, tmp_path) -> None:
    cfg = _config()
    before = json.dumps(cfg.sub_job, sort_keys=True)
    root = tmp_path / "checkpoint"
    root.mkdir()
    (root / "latest").write_text("global_step10\n", encoding="ascii")
    client = _LocalClient(
        losses=[2.5],
        save_result={
            "checkpoint_path": str(root),
            "checkpoint_tag": "global_step10",
            "version": 10,
            "checkpoint_id": LOCAL_CHECKPOINT_ID,
            "stage_info": deepcopy(LOCAL_STAGE),
        },
    )
    monkeypatch.setattr(hosted, "correctness_workdir", lambda prefix: root)
    monkeypatch.setattr(hosted, "ArcticClient", client.create)

    with HostedTransport(_local_backend(), cfg) as transport:
        with transport.job() as first:
            checkpoint = first.save_resumable()
        assert client.cancelled == ["job-1"]
        with transport.job(resume_from=checkpoint) as resumed:
            assert resumed.job_id == "job-2"
            assert client.calls[-1] == ("load", LOCAL_CHECKPOINT_ID, LOCAL_STAGE)
            assert resumed.fwd_bwd_step({"args": (), "kwargs": {}}, learning_rate=1e-3) == 2.5

    assert client.cancelled == ["job-1", "job-2"]
    assert client.calls == [
        ("save", "resumable", str(root)),
        ("load", LOCAL_CHECKPOINT_ID, LOCAL_STAGE),
        ("fwd-bwd", {"args": (), "kwargs": {}}),
        ("step", 1e-3),
    ]
    for wire, hardware in client.created:
        assert wire["model_name"] == cfg.model_name == "Qwen/Qwen3-8B"
        assert "source_checkpoint_info" not in wire
        assert hardware == "H200"
        assert wire["training_config"]["n_gpus"] == cfg.n_gpus == 4
        assert wire["training_config"]["ds_config"]["gradient_accumulation_steps"] == 1
    assert json.dumps(cfg.sub_job, sort_keys=True) == before
    assert root.is_dir()
    assert (root / "latest").read_text(encoding="ascii") == "global_step10\n"
    assert client.closed


def test_local_resume_load_failure_releases_fresh_job_and_retains_checkpoint(monkeypatch, tmp_path) -> None:
    class FailingLoad(_LocalClient):
        def load_checkpoint(self, checkpoint_id, *, stage_info=None):
            super().load_checkpoint(checkpoint_id, stage_info=stage_info)
            raise RuntimeError("runtime load unavailable")

    client = FailingLoad()
    (tmp_path / "latest").write_text("global_step10\n", encoding="ascii")
    monkeypatch.setattr(hosted, "ArcticClient", client.create)
    with pytest.raises(RuntimeError, match="runtime load unavailable"):
        with HostedTransport(_local_backend(), _config()) as transport:
            with transport.job(resume_from=_local_checkpoint(tmp_path)):
                raise AssertionError("must not train before a successful load")
    assert client.cancelled == ["job-1"]
    assert client.calls == [("load", LOCAL_CHECKPOINT_ID, LOCAL_STAGE)]
    assert tmp_path.is_dir()
    assert (tmp_path / "latest").read_text(encoding="ascii") == "global_step10\n"
    assert client.closed


@pytest.mark.parametrize(
    "result",
    [
        {},
        {"checkpoint_tag": "global_step0", "version": 0},
        {"checkpoint_tag": "global_step10", "version": True},
        {"checkpoint_tag": "global_step10", "version": 9},
    ],
)
def test_local_resume_rejects_unconfirmed_or_wrong_step_and_releases_fresh_job(monkeypatch, tmp_path, result) -> None:
    class IncorrectLoad(_LocalClient):
        def load_checkpoint(self, checkpoint_id, *, stage_info=None):
            super().load_checkpoint(checkpoint_id, stage_info=stage_info)
            return hosted._local_load_metadata({"checkpoint_id": checkpoint_id, **result}, checkpoint_id)

    (tmp_path / "latest").write_text("global_step10\n", encoding="ascii")
    client = IncorrectLoad()
    monkeypatch.setattr(hosted, "ArcticClient", client.create)
    with pytest.raises(RuntimeError, match="local checkpoint load"):
        with HostedTransport(_local_backend(), _config()) as transport:
            with transport.job(resume_from=_local_checkpoint(tmp_path)):
                raise AssertionError("must not train before confirmed restored step")
    assert client.cancelled == ["job-1"]
    assert client.calls == [("load", LOCAL_CHECKPOINT_ID, LOCAL_STAGE)]
    assert tmp_path.is_dir()
    assert client.closed


@pytest.mark.parametrize("marker", [None, b"", b"cp_123", b"global_step-1", b"\xff", b"global_step10" + b" " * 129])
def test_local_resume_rejects_missing_or_invalid_marker_before_allocating_job(monkeypatch, tmp_path, marker) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("a FAST marker must not authorize cross-job resume")

    if marker is not None:
        (tmp_path / "latest").write_bytes(marker)
    monkeypatch.setattr(hosted, "ArcticClient", unexpected)
    with pytest.raises(ValueError, match="latest marker"):
        hosted._local_resume_tag(tmp_path)
    with pytest.raises(ValueError, match="durable checkpoint identity"):
        with HostedTransport(_local_backend(), _config()) as transport:
            with transport.job(resume_from=Checkpoint(str(tmp_path), "owned-source")):
                pass


def test_local_resume_refuses_a_symlink_latest_marker(monkeypatch, tmp_path) -> None:
    target = tmp_path / "real-marker"
    target.write_text("global_step10\n", encoding="ascii")
    (tmp_path / "latest").symlink_to(target)
    with pytest.raises(ValueError, match="regular latest marker"):
        hosted._local_resume_tag(tmp_path)
    client = _LocalClient()
    monkeypatch.setattr(hosted, "ArcticClient", client.create)
    with pytest.raises(ValueError, match="durable checkpoint identity"):
        with HostedTransport(_local_backend(), _config()) as transport:
            with transport.job(resume_from=Checkpoint(str(tmp_path), "owned-source")):
                pass
    assert client.created == client.calls == client.cancelled == []
    assert target.read_text(encoding="ascii") == "global_step10\n"


def test_local_resume_refuses_a_nonregular_latest_marker(monkeypatch, tmp_path) -> None:
    (tmp_path / "latest").mkdir()
    with pytest.raises(ValueError, match="regular latest marker"):
        hosted._local_resume_tag(tmp_path)
    client = _LocalClient()
    monkeypatch.setattr(hosted, "ArcticClient", client.create)
    with pytest.raises(ValueError, match="durable checkpoint identity"):
        with HostedTransport(_local_backend(), _config()) as transport:
            with transport.job(resume_from=Checkpoint(str(tmp_path), "owned-source")):
                pass
    assert client.created == client.calls == client.cancelled == []
    assert (tmp_path / "latest").is_dir()


@pytest.mark.parametrize("reference", ["cp_hosted", "relative/checkpoint", "/tmp/checkpoint", "/data-fast"])
def test_local_resume_rejects_nonfast_or_nonabsolute_reference_before_job_creation(monkeypatch, reference) -> None:
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid reference must not create a training job")

    monkeypatch.setattr(hosted, "ArcticClient", unexpected)
    with pytest.raises(ValueError, match="durable checkpoint identity"):
        with HostedTransport(_local_backend(), _config()) as transport:
            with transport.job(resume_from=Checkpoint(reference, "owned-source")):
                pass


def test_local_resume_refuses_a_valid_fast_marker_before_allocating_job(monkeypatch, tmp_path) -> None:
    (tmp_path / "latest").write_text("global_step10\n", encoding="ascii")
    assert hosted._local_resume_tag(tmp_path) == "global_step10"
    client = _LocalClient()
    monkeypatch.setattr(hosted, "ArcticClient", client.create)
    with pytest.raises(ValueError, match="durable checkpoint identity"):
        with HostedTransport(_local_backend(), _config()) as transport:
            with transport.job(resume_from=Checkpoint(str(tmp_path), "owned-source")):
                raise AssertionError("a valid FAST marker is not a durable identity")
    assert client.created == client.calls == client.cancelled == []
    assert (tmp_path / "latest").read_text(encoding="ascii") == "global_step10\n"


def test_local_resume_requires_explicit_stage_before_allocating_job(monkeypatch, tmp_path) -> None:
    checkpoint = hosted.LocalCortexCheckpoint(
        LOCAL_CHECKPOINT_ID,
        "owned-source",
        str(tmp_path),
        "global_step10",
        10,
    )
    client = _LocalClient()
    monkeypatch.setattr(hosted, "ArcticClient", client.create)
    with pytest.raises(RuntimeError, match="explicit stage_info"):
        with HostedTransport(_local_backend(), _config()) as transport:
            with transport.job(resume_from=checkpoint):
                raise AssertionError("a checkpoint id alone cannot resolve another parent's store")
    assert client.created == client.calls == client.cancelled == []
