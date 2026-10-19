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

"""Cortex correctness jobs driven exclusively through the unified Arctic client."""

from __future__ import annotations

import contextlib
import json
import re
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import Iterator
from urllib.parse import urlsplit

from arctic_platform.client import CortexConfig

from .config import LoadedConfig
from .cortex import CorrectnessClient as ArcticClient
from .dss_driver import GatewaySession
from .dss_driver import _training_client_config
from .dss_driver import build_payload
from .dss_driver import cortex_training_body
from .seeds import SEED
from .transport import Checkpoint
from .workdir import correctness_workdir
from .workdir import require_data_fast_path

_STAGE_CHECKPOINT_ID = re.compile(r"/checkpoints/(cp_[0-9a-fA-F-]+)/")
_LOCAL_CHECKPOINT_TAG = re.compile(r"global_step(0|[1-9][0-9]*)")
_LOCAL_CHECKPOINT_ID = re.compile(r"cp_[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}")


@dataclass(frozen=True)
class LocalCortexCheckpoint(Checkpoint):
    checkpoint_path: str
    checkpoint_tag: str
    version: int
    stage_info: dict | None = field(default=None, repr=False)


def require_local_checkpoint_id(value: str) -> str:
    if not isinstance(value, str) or _LOCAL_CHECKPOINT_ID.fullmatch(value) is None:
        raise ValueError("local checkpoint load requires a durable cp_<uuid> checkpoint_id, not a FAST path or tag")
    return value


def _local_stage_info(value: dict | None, checkpoint_id: str) -> dict | None:
    if value is None:
        return None
    if (
        not isinstance(value, dict)
        or value.get("location_type") not in ("S3", "AZURE", "GCS")
        or any(not isinstance(value.get(key), str) or not value[key] for key in ("bucket", "prefix"))
        or set(value) - {"location_type", "bucket", "prefix", "region", "cred_expiry_time_ms", "aws", "azure", "gcp"}
    ):
        raise ValueError("local checkpoint load requires explicit cloud stage_info, not a filesystem restore address")
    if (
        value["bucket"].startswith("/")
        or "://" in value["bucket"]
        or value["prefix"].rstrip("/").rsplit("/", 1)[-1] != checkpoint_id
        or any(part in {".", ".."} for part in value["prefix"].split("/"))
    ):
        raise ValueError("local checkpoint stage prefix must name the requested durable checkpoint_id")
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError):
        raise ValueError("local checkpoint stage_info must be JSON metadata") from None


def _local_load_metadata(result: dict, checkpoint_id: str) -> dict:
    if not isinstance(result, dict) or result.get("checkpoint_id") != checkpoint_id:
        raise RuntimeError("local checkpoint load returned a different or missing checkpoint_id")
    tag = result.get("checkpoint_tag")
    match = _LOCAL_CHECKPOINT_TAG.fullmatch(tag) if isinstance(tag, str) else None
    version = result.get("version")
    if match is None or type(version) is not int or version != int(match.group(1)):
        raise RuntimeError("local checkpoint load returned inconsistent checkpoint_tag/version")
    return result


def read_pat(path: str | Path) -> str:
    token = Path(path).read_text().strip()
    if not token:
        raise ValueError(f"{path} holds no programmatic access token")
    return token


@dataclass(frozen=True)
class HostedConnection:
    """Compatibility connection for the hosted CLI; the PAT never appears in repr."""

    host: str
    database: str
    schema: str
    pat: str = field(repr=False)
    endpoint: str | None = None

    def backend(self) -> CortexConfig:
        options = {} if self.endpoint is None else {"endpoint": self.endpoint}
        return CortexConfig(host=self.host, database=self.database, schema=self.schema, pat=self.pat, **options)


def training_client_config(
    cfg: LoadedConfig,
    backend: CortexConfig,
    *,
    source_checkpoint_info: dict | None = None,
    attn_implementation: str | None = None,
):
    """Build a training-only unified config without rewriting the reviewed input."""
    payload = build_payload(
        cfg.training,
        cfg.model_name,
        cfg.sub_job.get("seed") if cfg.sub_job.get("seed") is not None else SEED,
        attn_implementation=attn_implementation,
        gradient_norms_per_param=False,
    )
    # DSS schedules request-wide accumulation from mb_spec, not the engine GAS.
    payload["dtype"] = cfg.sub_job.get("dtype") or payload["dtype"]
    payload["hardware"] = cfg.gpu_type.upper() if cfg.gpu_type is not None else None
    client_config = _training_client_config(GatewaySession(Path("."), cfg.n_gpus, backend), payload)
    if source_checkpoint_info is not None:
        client_config = type(client_config).model_validate(
            {**client_config.model_dump(), "source_checkpoint_info": source_checkpoint_info}
        )
    return client_config


def training_sub_job(cfg: LoadedConfig, *, source_checkpoint_info: dict | None = None) -> dict:
    """Compatibility helper exposing the unified client's create-job wire."""
    config = training_client_config(
        cfg, CortexConfig(base_url="http://unused"), source_checkpoint_info=source_checkpoint_info
    )
    return config.to_cortex()[0]


def save_and_resolve_checkpoint_id(client: ArcticClient, job_id: str, checkpoint_type: str) -> str:
    """Use a durable resource ID, never the node-local DeepSpeed checkpoint tag."""
    known = {entry["checkpoint_id"] for entry in client.transport.list_checkpoints()}
    result = client.save_checkpoint(checkpoint_type=checkpoint_type)
    match = _STAGE_CHECKPOINT_ID.search(str(result.get("stage_path", "")))
    if match:
        return match.group(1)
    checkpoint_id = result.get("checkpoint_id")
    if isinstance(checkpoint_id, str) and checkpoint_id.startswith("cp_"):
        return checkpoint_id
    added = [entry for entry in client.transport.list_checkpoints() if entry["checkpoint_id"] not in known]
    if len(added) != 1:
        raise RuntimeError(
            f"{checkpoint_type} save on job {job_id} added {len(added)} checkpoints; "
            "the saved checkpoint cannot be identified"
        )
    return added[0]["checkpoint_id"]


def is_local_cortex_gateway(backend: CortexConfig) -> bool:
    """Select only the explicitly configured, plain-HTTP loopback DSS gateway."""
    parsed = urlsplit(backend.base_url or "")
    return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}


def require_local_cortex_gateway(backend: CortexConfig) -> None:
    """Refuse hosted identity, authentication, and URL decoration on the local route."""
    parsed = urlsplit(backend.base_url or "")
    if (
        not is_local_cortex_gateway(backend)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or backend.host is not None
        or backend.pat is not None
    ):
        raise ValueError("local Cortex gateway requires an unauthenticated HTTP loopback base URL")


def save_local_checkpoint(client: ArcticClient, checkpoint_type: str) -> Path:
    """Retain a local DSS save without a hosted catalog or stage credentials."""
    root, _ = _save_local_checkpoint(client, checkpoint_type)
    return root


def _save_local_checkpoint(client: ArcticClient, checkpoint_type: str) -> tuple[Path, dict]:
    if checkpoint_type not in {"resumable", "weights-only"}:
        raise ValueError("local checkpoint_type must be resumable or weights-only")
    root = require_data_fast_path(correctness_workdir("cortex-checkpoint-"), label="local checkpoint save")
    result = client.save_checkpoint(path=str(root), checkpoint_type=checkpoint_type)
    _local_checkpoint_metadata(result, root, "save")
    # Keep successful and partial saves: the correctness caller never owns deletion.
    return root, result


def _local_checkpoint_metadata(result: dict, root: Path, operation: str) -> str:
    if not isinstance(result, dict):
        raise RuntimeError(f"local checkpoint {operation} returned no checkpoint metadata")
    path = result.get("checkpoint_path")
    if not isinstance(path, str) or not Path(path).is_absolute():
        raise RuntimeError(f"local checkpoint {operation} returned no absolute checkpoint_path")
    saved = require_data_fast_path(path, label="local checkpoint response")
    if saved != root:
        raise RuntimeError(f"local checkpoint {operation} returned a different checkpoint_path")
    tag = result.get("checkpoint_tag")
    match = _LOCAL_CHECKPOINT_TAG.fullmatch(tag) if isinstance(tag, str) else None
    version = result.get("version")
    if match is None or type(version) is not int or version != int(match.group(1)):
        raise RuntimeError(f"local checkpoint {operation} returned inconsistent checkpoint_tag/version")
    return tag


def _local_resume_tag(root: Path) -> str:
    marker = root / "latest"
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("local checkpoint resume requires a regular latest marker")
    try:
        with marker.open("rb") as stream:
            raw = stream.read(129)
        tag = raw.decode("ascii").strip()
    except (OSError, UnicodeError):
        raise ValueError("local checkpoint resume requires a readable latest marker") from None
    if len(raw) > 128 or _LOCAL_CHECKPOINT_TAG.fullmatch(tag) is None:
        raise ValueError("local checkpoint resume requires a bounded global_stepN latest marker")
    return tag


@dataclass
class HostedTrainingJob:
    client: ArcticClient
    job_id: str
    local_gateway: bool = False

    def fwd_bwd_step(self, body: dict, *, learning_rate: float) -> float:
        result = self.client.fwd_bwd(cortex_training_body(body))
        loss = result.get("avg_loss", (result.get("metrics") or {}).get("loss"))
        if loss is None:
            raise RuntimeError(f"forward-backward on job {self.job_id} returned no avg_loss")
        self.client.step(learning_rate=learning_rate)
        return float(loss)

    def save_resumable(self) -> Checkpoint:
        if self.local_gateway:
            root, result = _save_local_checkpoint(self.client, "resumable")
            checkpoint_id = result.get("checkpoint_id")
            if checkpoint_id is None:
                raise RuntimeError(
                    "local resumable save returned only a FAST path/tag; the gateway has no durable checkpoint "
                    "catalog, so cross-job resume is unavailable (saved files retained)"
                )
            require_local_checkpoint_id(checkpoint_id)
            return LocalCortexCheckpoint(
                reference=checkpoint_id,
                source_job=self.job_id,
                checkpoint_path=str(root),
                checkpoint_tag=result["checkpoint_tag"],
                version=result["version"],
                stage_info=_local_stage_info(result.get("stage_info"), checkpoint_id),
            )
        return Checkpoint(
            reference=save_and_resolve_checkpoint_id(self.client, self.job_id, "resumable"),
            source_job=self.job_id,
        )


class HostedTransport:
    """One unified client per training job; checkpoints outlive the saving client."""

    def __init__(
        self,
        connection: HostedConnection | CortexConfig,
        cfg: LoadedConfig,
        *,
        attn_implementation: str | None = None,
    ) -> None:
        self.backend = connection.backend() if isinstance(connection, HostedConnection) else connection
        self.cfg = cfg
        self.attn_implementation = attn_implementation
        self._entered = False

    def __enter__(self) -> "HostedTransport":
        self._entered = True
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._entered = False

    @contextlib.contextmanager
    def job(self, *, resume_from: Checkpoint | None = None) -> Iterator[HostedTrainingJob]:
        if not self._entered:
            raise RuntimeError("HostedTransport.job() requires the transport to be entered first")
        local_gateway = is_local_cortex_gateway(self.backend)
        if local_gateway:
            require_local_cortex_gateway(self.backend)
        local_checkpoint = None
        source = None
        if resume_from is not None:
            if local_gateway:
                if not isinstance(resume_from, LocalCortexCheckpoint):
                    raise ValueError(
                        "local cross-job resume requires a saved durable checkpoint identity with explicit stage_info;"
                        " FAST paths and source_job_id cannot be loaded by this gateway"
                    )
                require_local_checkpoint_id(resume_from.reference)
                if _local_stage_info(resume_from.stage_info, resume_from.reference) is None:
                    raise RuntimeError(
                        "local cross-job resume requires explicit stage_info; this gateway cannot resolve "
                        "another parent's checkpoint store"
                    )
                _local_load_metadata(
                    {
                        "checkpoint_id": resume_from.reference,
                        "checkpoint_tag": resume_from.checkpoint_tag,
                        "version": resume_from.version,
                    },
                    resume_from.reference,
                )
                local_checkpoint = resume_from
            else:
                source = {"checkpoint_id": resume_from.reference, "source_job_id": resume_from.source_job}
        config = training_client_config(
            self.cfg,
            self.backend,
            source_checkpoint_info=source,
            attn_implementation=self.attn_implementation,
        )
        with ArcticClient(config) as client:
            if local_checkpoint is not None:
                result = client.transport.load_checkpoint(
                    local_checkpoint.reference,
                    stage_info=local_checkpoint.stage_info,
                )
                if result["checkpoint_tag"] != local_checkpoint.checkpoint_tag:
                    raise RuntimeError("local checkpoint load restored a different global step")
            job_id = str(client.jobs.require("training")).split(":", 1)[0]
            print(f"  Cortex job {job_id} ready for {self.cfg.config_id}", flush=True)
            yield HostedTrainingJob(client, job_id, local_gateway=local_gateway)
