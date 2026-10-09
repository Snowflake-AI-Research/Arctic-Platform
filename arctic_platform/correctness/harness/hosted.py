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

"""Drive a Arctic Platform training job through the hosted control plane instead of a gateway this process starts.

Cross-job checkpoint initialization is the reason this transport exists. A create-time
``source_checkpoint_info`` of ``{"checkpoint_id": ..., "source_job_id": ...}`` is the documented client
shape, but a zone rejects it as ``config_invalid`` when it arrives without stage credentials, and a client
posting a job config straight to a zone has none to attach. The hosted server mints the stage and stamps
the scoped credentials onto the sub-job config before any zone sees it, so the resumed job initializes
from the saving job's checkpoint with no credential handling on this side.

A job runs the training sub-job of the ``LoadedConfig`` it is given. At the declared width that object
is the file. A wider placement hands it the in-memory copy ``at_gpu_width`` builds, so ``n_gpus`` and
``train_batch_size`` are the width's, and the file is not written. The fields the client validates -- the
optimizer block, ``max_seq_len``, ``train_batch_size``, ``n_gpus``, ``gradient_clipping``,
``multiplex_job_id`` and ``load_optimizer_states`` -- are passed as themselves, and every other key rides
through ``extra_training`` unchanged. Only the training sub-job is created: a correctness check measures a
training step, and a sampling sub-job beside it would claim GPUs nothing sends a request to.

``load_optimizer_states`` is left at whatever the config says, which for a config that does not mention it
is the server default of true. Optimizer state is sharded over the data-parallel ranks and cannot be
resized, so the resumed job declares the same ``n_gpus`` as the job that saved the checkpoint -- here they
read the same config, so they always do.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Iterator
from typing import Optional

from .config import LoadedConfig
from .transport import Checkpoint

# The fields ``SubJobConfig.training_job`` takes as named arguments. Everything else in the operator's
# training config is passthrough, and is forwarded without being read.
TYPED_TRAINING_FIELDS = (
    "optimizer",
    "max_seq_len",
    "train_batch_size",
    "n_gpus",
    "gradient_clipping",
    "multiplex_job_id",
    "load_optimizer_states",
)

# The durable checkpoint id is the path segment the save result's stage path carries. The DeepSpeed tag
# beside it in that result identifies a directory inside the saving job's containers, not a checkpoint
# resource, so it cannot be handed to another job.
_STAGE_CHECKPOINT_ID = re.compile(r"/checkpoints/(cp_[0-9a-fA-F-]+)/")


def read_pat(path: str | Path) -> str:
    """Read a programmatic access token from a file.

    The token is a credential: it is read at the moment it is needed, passed to the client, and never
    logged, echoed, or written anywhere.
    """
    token = Path(path).read_text().strip()
    if not token:
        raise ValueError(f"{path} holds no programmatic access token")
    return token


@dataclass(frozen=True)
class HostedConnection:
    """Which hosted deployment to call, and with whose credential.

    ``pat`` is excluded from the representation so a connection can be logged or included in an exception
    without exposing the token.
    """

    host: str
    database: str
    schema: str
    pat: str = field(repr=False)
    endpoint: Optional[str] = None


def hosted_client(connection: HostedConnection):
    """A PAT-authenticated client for one hosted deployment.

    Client-side telemetry is switched off before construction, because the client decides once, in
    ``from_pat``, whether to stand up a metric exporter, and a correctness run has no use for one.
    ``endpoint`` is left to the client's own default when the connection does not name one.
    """
    from dss_client.neutrino_client import NeutrinoClient

    os.environ["DSS_NEUTRINO_DISABLE_TELEMETRY"] = "1"
    options = {} if connection.endpoint is None else {"endpoint": connection.endpoint}
    return NeutrinoClient.from_pat(
        host=connection.host,
        pat=connection.pat,
        database=connection.database,
        schema=connection.schema,
        **options,
    )


def training_sub_job(cfg: LoadedConfig, *, source_checkpoint_info: Optional[dict] = None):
    """The config file's training sub-job as a create-job request, optionally resuming from a checkpoint.

    The training config is copied before it is handed over, because the client canonicalizes the optimizer
    block it is given and the loaded config is read by everything else in the run.
    """
    from dss_client.neutrino_client import SubJobConfig

    from arctic_platform.correctness.harness.dss_driver import pin_suite_determinism

    training = json.loads(json.dumps(cfg.training))
    # Same set as the gateway path. A config that asks for the other values is overwritten here, in the
    # copy, because a checkpoint comparison of two unpinned jobs disagrees before the checkpoint exists.
    debug = dict(training.get("debug") or {})
    pin_suite_determinism(debug)
    training["debug"] = debug
    passthrough = {key: value for key, value in training.items() if key not in TYPED_TRAINING_FIELDS}
    return SubJobConfig.training_job(
        cfg.model_name,
        optimizer=training["optimizer"],
        max_seq_len=training["max_seq_len"],
        train_batch_size=training["train_batch_size"],
        n_gpus=training["n_gpus"],
        gradient_clipping=training.get("gradient_clipping"),
        multiplex_job_id=training.get("multiplex_job_id"),
        load_optimizer_states=training.get("load_optimizer_states"),
        extra_training=passthrough,
        global_batch_size=cfg.sub_job.get("global_batch_size"),
        dtype=cfg.sub_job.get("dtype"),
        seed=cfg.sub_job.get("seed"),
        model_post_init=cfg.sub_job.get("model_post_init"),
        source_checkpoint_info=source_checkpoint_info,
    )


def save_and_resolve_checkpoint_id(client, job_id: str, checkpoint_type: str) -> str:
    """Save a checkpoint and return the durable id another job can initialize from.

    The save result names the stage path the checkpoint landed in, and the id is a segment of it. When the
    result carries no stage path, the job's checkpoint catalog is diffed against the ids that existed
    before the save, which does not depend on the catalog's order -- the order is not specified.
    """
    known = {entry["checkpoint_id"] for entry in client.list_checkpoints(job_id)}
    result = client.poll_request(job_id, client.save(job_id, checkpoint_type=checkpoint_type))
    match = _STAGE_CHECKPOINT_ID.search(str(result.get("stage_path", "")))
    if match:
        return match.group(1)
    added = [entry for entry in client.list_checkpoints(job_id) if entry["checkpoint_id"] not in known]
    if len(added) != 1:
        raise RuntimeError(
            f"{checkpoint_type} save on job {job_id} reported stage path "
            f"{result.get('stage_path')!r} and added {len(added)} checkpoint(s) to the catalog, so the "
            "saved checkpoint cannot be identified"
        )
    return added[0]["checkpoint_id"]


@dataclass
class HostedTrainingJob:
    """One hosted job, driven through the asynchronous data plane: submit, then poll to completion."""

    client: object
    job_id: str

    def fwd_bwd_step(self, body: bytes, *, learning_rate: float) -> float:
        result = self.client.poll_request(self.job_id, self.client.forward_backward(self.job_id, body))
        if "avg_loss" not in result:
            raise RuntimeError(
                f"forward-backward on job {self.job_id} returned no avg_loss; keys are {sorted(result)}"
            )
        loss = float(result["avg_loss"])
        self.client.poll_request(self.job_id, self.client.step(self.job_id, learning_rate=learning_rate))
        return loss

    def save_resumable(self) -> Checkpoint:
        return Checkpoint(
            reference=save_and_resolve_checkpoint_id(self.client, self.job_id, "resumable"),
            source_job=self.job_id,
        )


class HostedTransport:
    """Create hosted jobs for one config, one at a time, for as long as the block runs.

    Each job is cancelled when its block exits, so the GPUs it holds are released before the next job asks
    for them. A checkpoint outlives the job that wrote it, so a resumed job does not need its source to
    still be running.
    """

    def __init__(self, connection: HostedConnection, cfg: LoadedConfig) -> None:
        self.connection = connection
        self.cfg = cfg
        self._client = None

    def __enter__(self) -> "HostedTransport":
        self._client = hosted_client(self.connection)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        client, self._client = self._client, None
        if client is not None:
            client.close()

    @property
    def client(self):
        if self._client is None:
            raise RuntimeError("HostedTransport.job() requires the transport to be entered first")
        return self._client

    @property
    def hardware(self) -> Optional[str]:
        """The GPU family the config's location names, in the spelling create-job takes."""
        gpu_type = self.cfg.gpu_type
        return None if gpu_type is None else gpu_type.upper()

    @contextlib.contextmanager
    def job(self, *, resume_from: Optional[Checkpoint] = None) -> Iterator[HostedTrainingJob]:
        client = self.client
        source = None
        if resume_from is not None:
            source = {"checkpoint_id": resume_from.reference, "source_job_id": resume_from.source_job}
        sub_job = training_sub_job(self.cfg, source_checkpoint_info=source)
        job_id = client.create_job(sub_jobs=[sub_job], hardware=self.hardware)
        print(
            f"  hosted job {job_id} created for {self.cfg.config_id}"
            + (
                f", resuming {resume_from.reference} from job {resume_from.source_job}"
                if resume_from is not None
                else ""
            ),
            flush=True,
        )
        try:
            client.wait_for_job(job_id)
            yield HostedTrainingJob(client, job_id)
        finally:
            try:
                client.cancel_job(job_id)
            except Exception as exc:  # noqa: BLE001 - a job the server already finished refuses to be
                # cancelled, and whatever finished it is the failure worth raising out of this block
                print(f"  cancel of hosted job {job_id} was refused: {exc}", flush=True)
