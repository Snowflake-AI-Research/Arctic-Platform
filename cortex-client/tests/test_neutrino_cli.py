"""Offline tests for the dss-neutrino CLI."""

from __future__ import annotations

import io
import json

import dss_neutrino_cli as cli


class FakeClient:
    def __init__(self):
        self.submitted_body = None
        self.status_filter = None
        self.cancelled_job_id = None
        self.waited_job_id = None
        self.forward_backward_job_id = None
        self.forward_backward_payload = None
        self.polled_request = None
        self.stepped_job_id = None
        self.step_learning_rate = None
        self.loaded_job_id = None
        self.load_checkpoint_id = None
        self.load_source_job_id = None
        self.generated_job_id = None
        self.generate_prompts = None
        self.generate_sampling_params = None
        self.generate_routing_key = None
        self.generate_strict = None
        self.weight_sync_job_id = None
        self.weight_sync_source_sub_job_id = None
        self.weight_sync_target_sub_job_ids = None
        self.weight_sync_sub_job_id = None
        self.weight_sync_sub_job_type = None
        self.capacity_requested = False
        self.checkpoints_job_id = None
        self.jobs = None

    def create_job_from_body(self, body):
        self.submitted_body = body
        return {"job_id": body.get("job_id", "server-job")}

    def wait_for_job(self, job_id):
        self.waited_job_id = job_id
        return {"job_id": job_id, "status": "running"}

    def get_job(self, job_id):
        return {"job_id": job_id, "status": "running"}

    def list_jobs(self, status=None):
        self.status_filter = status
        if self.jobs is not None:
            return self.jobs
        return [{"job_id": "j1", "status": status or "running"}]

    def list_checkpoints(self, job_id):
        self.checkpoints_job_id = job_id
        return [
            {"checkpoint_id": "cp-1"},
            {"checkpoint_id": "cp-2"},
        ]

    def cancel_job(self, job_id):
        self.cancelled_job_id = job_id

    def get_capacity(self):
        self.capacity_requested = True
        return {
            "has_reservation": True,
            "reserved_gpus": 64,
            "in_use_gpus": 8,
            "available_gpus": 56,
        }

    def forward_backward(self, job_id, payload):
        self.forward_backward_job_id = job_id
        self.forward_backward_payload = payload
        return "req-fb-1"

    def poll_request(self, job_id, request_id):
        self.polled_request = (job_id, request_id)
        if request_id == "req-generate-1":
            return {"responses": [{"text": "MoE routing sends tokens to experts."}]}
        if request_id == "req-weight-sync-1":
            return {"synced": True}
        if request_id == "req-load-1":
            return {"checkpoint_id": "cp-7"}
        return {"avg_loss": 0.125}

    def step(self, job_id, learning_rate=None):
        self.stepped_job_id = job_id
        self.step_learning_rate = learning_rate
        return "req-step-1"

    def load(
        self,
        job_id,
        checkpoint_id,
        source_job_id=None,
    ):
        self.loaded_job_id = job_id
        self.load_checkpoint_id = checkpoint_id
        self.load_source_job_id = source_job_id
        return "req-load-1"

    def generate(
        self,
        job_id,
        prompts,
        sampling_params=None,
        routing_key=None,
        strict=None,
    ):
        self.generated_job_id = job_id
        self.generate_prompts = prompts
        self.generate_sampling_params = sampling_params
        self.generate_routing_key = routing_key
        self.generate_strict = strict
        return "req-generate-1"

    def weight_sync(
        self,
        job_id,
        source_sub_job_id,
        target_sub_job_ids,
        sub_job_id=None,
        sub_job_type=None,
    ):
        self.weight_sync_job_id = job_id
        self.weight_sync_source_sub_job_id = source_sub_job_id
        self.weight_sync_target_sub_job_ids = target_sub_job_ids
        self.weight_sync_sub_job_id = sub_job_id
        self.weight_sync_sub_job_type = sub_job_type
        return "req-weight-sync-1"

    def fetch_execution_logs(self, job_id):
        return [
            {
                "sub_job_id": f"{job_id}:training:0",
                "filename": "execution.jsonl",
                "s3_uri": (
                    f"s3://bucket/stage/versions/v1/checkpoints/_logs/"
                    f"{job_id}:training:0/execution.jsonl"
                ),
                "content": '{"a":1}\n',
            },
            {
                "sub_job_id": f"{job_id}:training:0",
                "filename": "server.log",
                "s3_uri": (
                    f"s3://bucket/stage/versions/v1/checkpoints/_logs/"
                    f"{job_id}:training:0/server.log"
                ),
                "content": "server line\n",
            },
            {
                "sub_job_id": f"{job_id}:sampling:0",
                "filename": "execution.jsonl",
                "s3_uri": (
                    f"s3://bucket/stage/versions/v1/checkpoints/_logs/"
                    f"{job_id}:sampling:0/execution.jsonl"
                ),
                "content": '{"b":2}\n',
            },
        ]


class FakeHTTPError(OSError):
    def __init__(self):
        super().__init__("500 Server Error: Internal Server Error for url: http://test")
        self.response = type(
            "Response",
            (),
            {
                "headers": {"x-snowflake-request-id": "sf-req-1"},
                "text": '{"message":"backend detail"}',
            },
        )()


def _factory(instances):
    def make_client(_args):
        client = FakeClient()
        instances.append(client)
        return client

    return make_client


def _write_job(tmp_path):
    path = tmp_path / "job.json"
    path.write_text(
        json.dumps({
            "sub_job_configs": [{
                "job_type": "sampling",
                "model_name": "gpt2",
                "inference_config": {"max_seq_len": 128, "n_gpus": 1},
            }],
        }),
        encoding="utf-8",
    )
    return path


def _write_config(tmp_path, data):
    path = tmp_path / "neutrino.config.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _wire_load(data: bytes):
    from dss_client import wire

    return wire.loads(data)


def _base_args():
    return ["--base-url", "http://test.local", "--database", "DB"]


def test_submit_reads_create_job_json_and_overrides_job_id(tmp_path):
    instances = []
    stdout = io.StringIO()
    path = _write_job(tmp_path)

    rc = cli.main(
        _base_args() + ["submit", str(path), "--job-id", "client-job"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].submitted_body["job_id"] == "client-job"
    assert json.loads(stdout.getvalue()) == {"job_id": "client-job"}


def test_submit_wait_prints_waited_job(tmp_path):
    instances = []
    stdout = io.StringIO()
    path = _write_job(tmp_path)

    rc = cli.main(
        _base_args() + ["submit", str(path), "--wait"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].waited_job_id == "server-job"
    assert json.loads(stdout.getvalue()) == {"job_id": "server-job", "status": "running"}


def test_submit_dry_run_does_not_require_connection(tmp_path, monkeypatch):
    stdout = io.StringIO()
    path = _write_job(tmp_path)
    monkeypatch.setattr(
        cli,
        "_load_neutrino_client_class",
        lambda: (_ for _ in ()).throw(AssertionError("client loaded")),
    )

    rc = cli.main(["submit", str(path), "--dry-run", "--job-id", "dry"], stdout=stdout)

    assert rc == 0
    assert json.loads(stdout.getvalue())["job_id"] == "dry"


def test_list_prints_jobs_with_status_filter():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["list", "--status", "running"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].status_filter == "running"
    assert json.loads(stdout.getvalue()) == {
        "jobs": [{"job_id": "j1", "status": "running"}],
    }


def test_list_prints_latest_jobs_last():
    instances = []
    stdout = io.StringIO()
    jobs = [
        {
            "job_id": "latest",
            "status": "running",
            "created_at": "2026-06-13T00:00:00Z",
        },
        {
            "job_id": "mid",
            "status": "done",
            "created_at": "2026-06-12T00:00:00Z",
        },
        {
            "job_id": "oldest",
            "status": "done",
            "created_at": "2026-06-10T00:00:00Z",
        },
    ]

    def make_client(_args):
        client = FakeClient()
        client.jobs = jobs
        instances.append(client)
        return client

    rc = cli.main(_base_args() + ["list"], client_factory=make_client, stdout=stdout)

    assert rc == 0
    assert json.loads(stdout.getvalue()) == {
        "jobs": [
            {
                "job_id": "oldest",
                "status": "done",
                "created_at": "2026-06-10T00:00:00Z",
            },
            {
                "job_id": "mid",
                "status": "done",
                "created_at": "2026-06-12T00:00:00Z",
            },
            {
                "job_id": "latest",
                "status": "running",
                "created_at": "2026-06-13T00:00:00Z",
            },
        ],
    }


def test_list_sorts_missing_created_at_after_dated_jobs():
    stdout = io.StringIO()

    def make_client(_args):
        client = FakeClient()
        client.jobs = [
            {"job_id": "missing", "status": "running"},
            {
                "job_id": "latest",
                "status": "running",
                "created_at": "2026-06-13T00:00:00Z",
            },
            {
                "job_id": "oldest",
                "status": "done",
                "created_at": "2026-06-10T00:00:00Z",
            },
        ]
        return client

    rc = cli.main(_base_args() + ["list"], client_factory=make_client, stdout=stdout)

    assert rc == 0
    assert [job["job_id"] for job in json.loads(stdout.getvalue())["jobs"]] == [
        "oldest",
        "latest",
        "missing",
    ]


def test_get_prints_job():
    stdout = io.StringIO()

    rc = cli.main(_base_args() + ["get", "j1"], client_factory=_factory([]), stdout=stdout)

    assert rc == 0
    assert json.loads(stdout.getvalue()) == {"job_id": "j1", "status": "running"}


def test_checkpoints_lists_checkpoints_for_job():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["checkpoints", "j1"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].checkpoints_job_id == "j1"
    assert json.loads(stdout.getvalue()) == {
        "checkpoints": [
            {"checkpoint_id": "cp-1"},
            {"checkpoint_id": "cp-2"},
        ],
    }


def test_capacity_prints_account_gpu_usage():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["capacity"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].capacity_requested is True
    assert json.loads(stdout.getvalue()) == {
        "has_reservation": True,
        "reserved_gpus": 64,
        "in_use_gpus": 8,
        "available_gpus": 56,
    }


def test_cancel_prints_confirmation():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["cancel", "j1"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].cancelled_job_id == "j1"
    assert json.loads(stdout.getvalue()) == {"cancelled": True, "job_id": "j1"}


def test_wait_prints_running_job():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["wait", "j1"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].waited_job_id == "j1"
    assert json.loads(stdout.getvalue()) == {"job_id": "j1", "status": "running"}


def test_fwd_bwd_serializes_payload_and_polls(tmp_path):
    instances = []
    stdout = io.StringIO()
    path = tmp_path / "fwd-bwd.json"
    path.write_text(
        json.dumps({
            "payload": {
                "kwargs": {
                    "input_ids": {"data": [[1, 2, 3]], "dtype": "long"},
                    "labels": {"data": [[2, 3, -100]], "dtype": "long"},
                },
            },
        }),
        encoding="utf-8",
    )

    rc = cli.main(
        _base_args() + ["--job", "job-1", "fwd-bwd", str(path)],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    client = instances[0]
    assert client.forward_backward_job_id == "job-1"
    assert client.polled_request == ("job-1", "req-fb-1")
    loaded = _wire_load(client.forward_backward_payload)
    assert loaded["args"] == ()
    assert loaded["kwargs"]["input_ids"].tolist() == [[1, 2, 3]]
    assert loaded["kwargs"]["labels"].tolist() == [[2, 3, -100]]
    assert json.loads(stdout.getvalue()) == {
        "job_id": "job-1",
        "payload_size_bytes": len(client.forward_backward_payload),
        "request_id": "req-fb-1",
        "result": {"avg_loss": 0.125},
    }


def test_fwd_bwd_can_skip_poll(tmp_path):
    instances = []
    stdout = io.StringIO()
    path = tmp_path / "fwd-bwd.json"
    path.write_text(
        json.dumps({
            "poll": False,
            "payload": {"kwargs": {"input_ids": [[1]]}},
        }),
        encoding="utf-8",
    )

    rc = cli.main(
        _base_args() + ["--job", "job-1", "fwd-bwd", str(path)],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].polled_request is None
    assert json.loads(stdout.getvalue()) == {
        "job_id": "job-1",
        "payload_size_bytes": len(instances[0].forward_backward_payload),
        "request_id": "req-fb-1",
    }


def test_fwd_bwd_accepts_job_id_alias(tmp_path):
    instances = []
    path = tmp_path / "fwd-bwd.json"
    path.write_text(
        json.dumps({
            "poll": False,
            "payload": {"kwargs": {"input_ids": [[1]]}},
        }),
        encoding="utf-8",
    )

    rc = cli.main(
        _base_args() + ["--job-id", "job-alias", "fwd-bwd", str(path)],
        client_factory=_factory(instances),
        stdout=io.StringIO(),
    )

    assert rc == 0
    assert instances[0].forward_backward_job_id == "job-alias"


def test_step_uses_default_lr_and_polls():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["--job-id", "job-1", "step"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].stepped_job_id == "job-1"
    assert instances[0].step_learning_rate == 1e-4
    assert instances[0].polled_request == ("job-1", "req-step-1")
    assert json.loads(stdout.getvalue()) == {
        "job_id": "job-1",
        "learning_rate": 1e-4,
        "request_id": "req-step-1",
        "result": {"avg_loss": 0.125},
    }


def test_step_accepts_lr_override():
    instances = []

    rc = cli.main(
        _base_args() + ["--job-id", "job-1", "step", "--lr", "2e-5"],
        client_factory=_factory(instances),
        stdout=io.StringIO(),
    )

    assert rc == 0
    assert instances[0].step_learning_rate == 2e-5


def test_load_uses_checkpoint_id_and_polls():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["--job-id", "job-1", "load", "cp-7"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    client = instances[0]
    assert client.loaded_job_id == "job-1"
    assert client.load_checkpoint_id == "cp-7"
    assert client.load_source_job_id is None
    assert client.polled_request == ("job-1", "req-load-1")
    assert json.loads(stdout.getvalue()) == {
        "checkpoint_id": "cp-7",
        "job_id": "job-1",
        "request_id": "req-load-1",
        "result": {"checkpoint_id": "cp-7"},
    }


def test_load_accepts_source_job_and_can_skip_poll():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args()
        + [
            "--job-id",
            "job-1",
            "load",
            "cp-7",
            "--source-job-id",
            "source-job",
            "--no-poll",
        ],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    client = instances[0]
    assert client.load_checkpoint_id == "cp-7"
    assert client.load_source_job_id == "source-job"
    assert client.polled_request is None
    assert json.loads(stdout.getvalue()) == {
        "checkpoint_id": "cp-7",
        "job_id": "job-1",
        "request_id": "req-load-1",
        "source_job_id": "source-job",
    }


def test_generate_reads_json_and_polls(tmp_path):
    instances = []
    stdout = io.StringIO()
    path = tmp_path / "generate.json"
    path.write_text(
        json.dumps({
            "prompts": [
                "Explain mixture-of-experts routing in one sentence.",
                "Name one benefit of sparse expert models.",
            ],
            "sampling_params": {
                "max_tokens": 32,
                "temperature": 0.2,
                "top_p": 0.9,
            },
            "routing_key": "sample-route",
            "strict": False,
        }),
        encoding="utf-8",
    )

    rc = cli.main(
        _base_args() + ["--job-id", "job-1", "generate", str(path)],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    client = instances[0]
    assert client.generated_job_id == "job-1"
    assert client.generate_prompts == [
        "Explain mixture-of-experts routing in one sentence.",
        "Name one benefit of sparse expert models.",
    ]
    assert client.generate_sampling_params == {
        "max_tokens": 32,
        "temperature": 0.2,
        "top_p": 0.9,
    }
    assert client.generate_routing_key == "sample-route"
    assert client.generate_strict is False
    assert client.polled_request == ("job-1", "req-generate-1")
    assert json.loads(stdout.getvalue()) == {
        "job_id": "job-1",
        "prompt_count": 2,
        "request_id": "req-generate-1",
        "result": {"responses": [{"text": "MoE routing sends tokens to experts."}]},
    }


def test_generate_accepts_per_prompt_sampling_params(tmp_path):
    instances = []
    path = tmp_path / "generate.json"
    path.write_text(
        json.dumps({
            "prompts": ["a", "b"],
            "sampling_params": [
                {"max_tokens": 32, "temperature": 0.2},
                None,
            ],
        }),
        encoding="utf-8",
    )

    rc = cli.main(
        _base_args() + ["--job-id", "job-1", "generate", str(path)],
        client_factory=_factory(instances),
        stdout=io.StringIO(),
    )

    assert rc == 0
    assert instances[0].generate_sampling_params == [
        {"max_tokens": 32, "temperature": 0.2},
        None,
    ]


def test_generate_can_skip_poll(tmp_path):
    instances = []
    stdout = io.StringIO()
    path = tmp_path / "generate.json"
    path.write_text(
        json.dumps({
            "poll": False,
            "payload": {
                "prompts": ["hello"],
                "sampling_params": {"max_tokens": 4},
            },
        }),
        encoding="utf-8",
    )

    rc = cli.main(
        _base_args() + ["--job-id", "job-1", "generate", str(path)],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    assert instances[0].polled_request is None
    assert json.loads(stdout.getvalue()) == {
        "job_id": "job-1",
        "prompt_count": 1,
        "request_id": "req-generate-1",
    }


def test_weight_sync_defaults_to_training_and_sampling_subjobs():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["--job-id", "job-1", "weight-sync"],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    client = instances[0]
    assert client.weight_sync_job_id == "job-1"
    assert client.weight_sync_source_sub_job_id == "job-1:training:0"
    assert client.weight_sync_target_sub_job_ids == ["job-1:sampling:0"]
    assert client.weight_sync_sub_job_id is None
    assert client.weight_sync_sub_job_type is None
    assert client.polled_request == ("job-1", "req-weight-sync-1")
    assert json.loads(stdout.getvalue()) == {
        "job_id": "job-1",
        "request_id": "req-weight-sync-1",
        "result": {"synced": True},
        "source_sub_job_id": "job-1:training:0",
        "target_sub_job_ids": ["job-1:sampling:0"],
    }


def test_weight_sync_accepts_subjob_overrides_and_can_skip_poll():
    instances = []
    stdout = io.StringIO()

    rc = cli.main(
        _base_args()
        + [
            "--job-id",
            "job-1",
            "weight-sync",
            "--source-sub-job-id",
            "job-1:training:1",
            "--target-sub-job-id",
            "job-1:sampling:0",
            "--target-sub-job-id",
            "job-1:sampling:1",
            "--operation-sub-job-id",
            "job-1:training:route",
            "--operation-sub-job-type",
            "training",
            "--no-poll",
        ],
        client_factory=_factory(instances),
        stdout=stdout,
    )

    assert rc == 0
    client = instances[0]
    assert client.weight_sync_source_sub_job_id == "job-1:training:1"
    assert client.weight_sync_target_sub_job_ids == [
        "job-1:sampling:0",
        "job-1:sampling:1",
    ]
    assert client.weight_sync_sub_job_id == "job-1:training:route"
    assert client.weight_sync_sub_job_type == "training"
    assert client.polled_request is None
    assert json.loads(stdout.getvalue()) == {
        "job_id": "job-1",
        "request_id": "req-weight-sync-1",
        "source_sub_job_id": "job-1:training:1",
        "target_sub_job_ids": ["job-1:sampling:0", "job-1:sampling:1"],
    }


def test_download_log_defaults_to_cwd_when_output_dir_omitted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    stdout = io.StringIO()

    rc = cli.main(
        _base_args() + ["download-log", "job-1"],
        client_factory=lambda _: FakeClient(),
        stdout=stdout,
    )

    assert rc == 0
    training_dir = tmp_path / "job-1:training:0"
    sampling_dir = tmp_path / "job-1:sampling:0"
    assert (training_dir / "execution.jsonl").read_text(encoding="utf-8") == '{"a":1}\n'
    assert (training_dir / "server.log").read_text(encoding="utf-8") == "server line\n"
    assert (sampling_dir / "execution.jsonl").read_text(encoding="utf-8") == '{"b":2}\n'

    payload = json.loads(stdout.getvalue())
    assert payload["job_id"] == "job-1"
    assert [(log["sub_job_id"], log["filename"]) for log in payload["logs"]] == [
        ("job-1:training:0", "execution.jsonl"),
        ("job-1:training:0", "server.log"),
        ("job-1:sampling:0", "execution.jsonl"),
    ]
    assert payload["logs"][0]["saved_path"] == str(training_dir / "execution.jsonl")


def test_download_log_writes_each_log_under_sub_job_dir(tmp_path):
    stdout = io.StringIO()

    rc = cli.main(
        _base_args()
        + [
            "download-log",
            "job-1",
            "--output-dir",
            str(tmp_path),
        ],
        client_factory=lambda _: FakeClient(),
        stdout=stdout,
    )

    assert rc == 0
    training_dir = tmp_path / "job-1:training:0"
    sampling_dir = tmp_path / "job-1:sampling:0"
    assert (training_dir / "execution.jsonl").read_text(encoding="utf-8") == '{"a":1}\n'
    assert (training_dir / "server.log").read_text(encoding="utf-8") == "server line\n"
    assert (sampling_dir / "execution.jsonl").read_text(encoding="utf-8") == '{"b":2}\n'

    payload = json.loads(stdout.getvalue())
    assert payload["job_id"] == "job-1"
    assert payload["logs"] == [
        {
            "sub_job_id": "job-1:training:0",
            "filename": "execution.jsonl",
            "s3_uri": (
                "s3://bucket/stage/versions/v1/checkpoints/_logs/"
                "job-1:training:0/execution.jsonl"
            ),
            "saved_path": str(training_dir / "execution.jsonl"),
        },
        {
            "sub_job_id": "job-1:training:0",
            "filename": "server.log",
            "s3_uri": (
                "s3://bucket/stage/versions/v1/checkpoints/_logs/"
                "job-1:training:0/server.log"
            ),
            "saved_path": str(training_dir / "server.log"),
        },
        {
            "sub_job_id": "job-1:sampling:0",
            "filename": "execution.jsonl",
            "s3_uri": (
                "s3://bucket/stage/versions/v1/checkpoints/_logs/"
                "job-1:sampling:0/execution.jsonl"
            ),
            "saved_path": str(sampling_dir / "execution.jsonl"),
        },
    ]


def test_http_errors_include_response_details():
    stderr = io.StringIO()

    def make_client(_args):
        raise FakeHTTPError()

    rc = cli.main(_base_args() + ["list"], client_factory=make_client, stderr=stderr)

    assert rc == 1
    err = stderr.getvalue()
    assert "500 Server Error" in err
    assert "snowflake request id: sf-req-1" in err
    assert 'response body: {"message":"backend detail"}' in err


def test_invalid_job_json_returns_error(tmp_path):
    path = tmp_path / "job.json"
    path.write_text("[]", encoding="utf-8")
    stderr = io.StringIO()

    rc = cli.main(["submit", str(path), "--dry-run"], stderr=stderr)

    assert rc == 1
    assert "job JSON must be an object" in stderr.getvalue()


def test_config_file_supplies_connection_values(tmp_path):
    config = _write_config(
        tmp_path,
        {
            "base_url": "http://config.local",
            "database": "CONFIG_DB",
            "schema": "CONFIG_SCHEMA",
            "endpoint": "config-endpoint",
            "poll_interval": 2.0,
            "poll_timeout": 30.0,
        },
    )
    seen = {}
    stdout = io.StringIO()

    def make_client(args):
        seen.update(vars(args))
        return FakeClient()

    rc = cli.main(["--config", str(config), "list"], client_factory=make_client, stdout=stdout)

    assert rc == 0
    assert seen["base_url"] == "http://config.local"
    assert seen["database"] == "CONFIG_DB"
    assert seen["schema"] == "CONFIG_SCHEMA"
    assert seen["endpoint"] == "config-endpoint"
    assert seen["poll_interval"] == 2.0
    assert seen["poll_timeout"] == 30.0


def test_cli_flags_override_config_file(tmp_path):
    config = _write_config(
        tmp_path,
        {
            "base_url": "http://config.local",
            "database": "CONFIG_DB",
            "schema": "CONFIG_SCHEMA",
        },
    )
    seen = {}

    def make_client(args):
        seen.update(vars(args))
        return FakeClient()

    rc = cli.main(
        [
            "--config",
            str(config),
            "--base-url",
            "http://cli.local",
            "--database",
            "CLI_DB",
            "list",
        ],
        client_factory=make_client,
        stdout=io.StringIO(),
    )

    assert rc == 0
    assert seen["base_url"] == "http://cli.local"
    assert seen["database"] == "CLI_DB"
    assert seen["schema"] == "CONFIG_SCHEMA"


def test_config_can_come_from_env(tmp_path, monkeypatch):
    config = _write_config(
        tmp_path,
        {
            "base_url": "http://env-config.local",
            "database": "ENV_CONFIG_DB",
        },
    )
    monkeypatch.setenv("NEUTRINO_CONFIG", str(config))
    seen = {}

    def make_client(args):
        seen.update(vars(args))
        return FakeClient()

    rc = cli.main(["list"], client_factory=make_client, stdout=io.StringIO())

    assert rc == 0
    assert seen["base_url"] == "http://env-config.local"
    assert seen["database"] == "ENV_CONFIG_DB"


def test_config_accepts_common_aliases(tmp_path):
    config = _write_config(tmp_path, {"url": "http://alias.local", "db": "ALIAS_DB"})
    seen = {}

    def make_client(args):
        seen.update(vars(args))
        return FakeClient()

    rc = cli.main(["--config", str(config), "list"], client_factory=make_client, stdout=io.StringIO())

    assert rc == 0
    assert seen["base_url"] == "http://alias.local"
    assert seen["database"] == "ALIAS_DB"


def test_bare_base_url_with_pat_is_treated_as_host(tmp_path):
    config = _write_config(
        tmp_path,
        {
            "base_url": "dsa-test.qa6.us-west-2.aws.snowflakecomputing.com",
            "pat": "test-pat",
            "database": "NEUTRINO_DB",
        },
    )
    seen = {}

    def make_client(args):
        seen.update(vars(args))
        return FakeClient()

    rc = cli.main(
        ["--config", str(config), "list"],
        client_factory=make_client,
        stdout=io.StringIO(),
    )

    assert rc == 0
    assert seen["base_url"] is None
    assert seen["host"] == "dsa-test.qa6.us-west-2.aws.snowflakecomputing.com"
    assert seen["pat"] == "test-pat"


def test_bare_base_url_without_pat_returns_clear_error(tmp_path, monkeypatch):
    monkeypatch.delenv("NEUTRINO_PAT", raising=False)
    monkeypatch.delenv("SNOWFLAKE_PAT", raising=False)
    config = _write_config(
        tmp_path,
        {
            "base_url": "dsa-test.qa6.us-west-2.aws.snowflakecomputing.com",
            "database": "NEUTRINO_DB",
        },
    )
    stderr = io.StringIO()

    rc = cli.main(["--config", str(config), "list"], stderr=stderr)

    assert rc == 1
    assert "base_url must start with http:// or https://" in stderr.getvalue()


def test_invalid_config_returns_error(tmp_path):
    config = _write_config(tmp_path, {"base_url": "http://test.local", "typo": "value"})
    stderr = io.StringIO()

    rc = cli.main(["--config", str(config), "submit", "missing.json", "--dry-run"], stderr=stderr)

    assert rc == 1
    assert "unknown config key" in stderr.getvalue()


def test_invalid_config_value_type_returns_error(tmp_path):
    config = _write_config(tmp_path, {"base_url": 123, "database": "DB"})
    stderr = io.StringIO()

    rc = cli.main(["--config", str(config), "list"], stderr=stderr)

    assert rc == 1
    assert "config base_url must be a string" in stderr.getvalue()


def test_login_persists_config_path(tmp_path, monkeypatch):
    login_state = tmp_path / "login.json"
    config = _write_config(
        tmp_path,
        {"base_url": "http://login.local", "database": "LOGIN_DB"},
    )
    monkeypatch.setenv("DSS_NEUTRINO_LOGIN_FILE", str(login_state))
    stdout = io.StringIO()

    rc = cli.main(["login", "--config", str(config)], stdout=stdout)

    assert rc == 0
    saved = json.loads(login_state.read_text(encoding="utf-8"))
    assert saved == {"config_path": str(config.resolve())}
    assert json.loads(stdout.getvalue()) == {
        "config_path": str(config.resolve()),
        "logged_in": True,
    }


def test_logged_in_config_is_used_without_config_arg(tmp_path, monkeypatch):
    login_state = tmp_path / "login.json"
    config = _write_config(
        tmp_path,
        {"base_url": "http://login.local", "database": "LOGIN_DB"},
    )
    login_state.write_text(
        json.dumps({"config_path": str(config)}),
        encoding="utf-8",
    )
    monkeypatch.setenv("DSS_NEUTRINO_LOGIN_FILE", str(login_state))
    seen = {}

    def make_client(args):
        seen.update(vars(args))
        return FakeClient()

    rc = cli.main(["list"], client_factory=make_client, stdout=io.StringIO())

    assert rc == 0
    assert seen["base_url"] == "http://login.local"
    assert seen["database"] == "LOGIN_DB"


def test_cli_flags_override_logged_in_config(tmp_path, monkeypatch):
    login_state = tmp_path / "login.json"
    config = _write_config(
        tmp_path,
        {
            "base_url": "http://login.local",
            "database": "LOGIN_DB",
            "schema": "LOGIN_SCHEMA",
        },
    )
    login_state.write_text(
        json.dumps({"config_path": str(config)}),
        encoding="utf-8",
    )
    monkeypatch.setenv("DSS_NEUTRINO_LOGIN_FILE", str(login_state))
    seen = {}

    def make_client(args):
        seen.update(vars(args))
        return FakeClient()

    rc = cli.main(
        ["--database", "CLI_DB", "list"],
        client_factory=make_client,
        stdout=io.StringIO(),
    )

    assert rc == 0
    assert seen["base_url"] == "http://login.local"
    assert seen["database"] == "CLI_DB"
    assert seen["schema"] == "LOGIN_SCHEMA"


def test_direct_connection_flags_do_not_read_login_state(tmp_path, monkeypatch):
    login_state = tmp_path / "login.json"
    login_state.write_text("not json", encoding="utf-8")
    monkeypatch.setenv("DSS_NEUTRINO_LOGIN_FILE", str(login_state))

    rc = cli.main(
        _base_args() + ["list"],
        client_factory=_factory([]),
        stdout=io.StringIO(),
    )

    assert rc == 0


def test_login_rejects_invalid_config(tmp_path, monkeypatch):
    login_state = tmp_path / "login.json"
    config = _write_config(tmp_path, {"typo": "value"})
    monkeypatch.setenv("DSS_NEUTRINO_LOGIN_FILE", str(login_state))
    stderr = io.StringIO()

    rc = cli.main(["login", "--config", str(config)], stderr=stderr)

    assert rc == 1
    assert not login_state.exists()
    assert "unknown config key" in stderr.getvalue()
