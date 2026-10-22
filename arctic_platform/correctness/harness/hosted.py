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

"""Supervised Cortex config and checkpoint-save helpers for the first two tests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from urllib.parse import urlsplit

from arctic_platform.client import CortexConfig

from .batches import SEED
from .config import LoadedConfig
from .cortex import CorrectnessClient as ArcticClient
from .dss_driver import GatewaySession
from .dss_driver import _training_client_config
from .dss_driver import build_payload
from .storage import correctness_workdir
from .storage import require_data_fast_path

_STAGE_CHECKPOINT_ID = re.compile(r"/checkpoints/(cp_[0-9a-fA-F-]+)/")
_LOCAL_CHECKPOINT_TAG = re.compile(r"global_step(0|[1-9][0-9]*)")


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
    if "weight_format" in cfg.sub_job:
        payload["weight_format"] = cfg.sub_job["weight_format"]
    return _training_client_config(GatewaySession(Path("."), cfg.n_gpus, backend), payload)


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
