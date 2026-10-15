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

"""Capture one managed optimizer update through durable resumable snapshots."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from arctic_platform.client import CortexConfig

from .dss_driver import ArcticJob
from .dss_driver import GatewaySession
from .dss_driver import StepResult
from .dss_driver import fwd_bwd_step
from .hosted import require_local_cortex_gateway
from .hosted import save_and_resolve_checkpoint_id
from .workdir import require_data_fast_path


def _save_local_snapshot(client, root: Path, *, step: int, expected_dp_world_size: int) -> Path:
    from .cortex_optimizer_decode import _checkpoint_files

    root = require_data_fast_path(root, label="local optimizer checkpoint")
    if root.exists() or root.is_symlink():
        raise FileExistsError("local optimizer checkpoint must be new")
    result = client.save_checkpoint(path=str(root), checkpoint_type="resumable", export_hf=False)
    if not isinstance(result, dict):
        raise ValueError("local optimizer save requires checkpoint metadata")
    reported = result.get("checkpoint_path")
    tag = f"global_step{step}"
    if (
        not isinstance(reported, str)
        or reported != str(root)
        or require_data_fast_path(reported, label="local optimizer checkpoint receipt") != root
        or result.get("checkpoint_tag") != tag
        or type(result.get("version")) is not int
        or result["version"] != step
        or result.get("stage_path")
    ):
        raise ValueError("local optimizer checkpoint receipt does not match its requested path and step")
    for index, path in enumerate(root.rglob("*")):
        if index >= 4096 or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError("local optimizer checkpoint has unsupported entries or symlinks")
    model, shards = _checkpoint_files(root)
    if model.parent != root / tag or len(shards) != expected_dp_world_size:
        raise ValueError("local optimizer checkpoint has incorrect tag or DP shard count")
    if any(path.stat().st_size == 0 for path in [model, *shards]):
        raise ValueError("local optimizer checkpoint contains empty tensor files")
    return root


def capture_optimizer_step(
    session: GatewaySession,
    job: ArcticJob,
    body: dict,
    *,
    learning_rate: float,
    output_dir: Path,
    expected_dp_world_size: int,
    optimizer_config: dict,
    local_checkpoint: bool = False,
) -> StepResult:
    """Save immediately around the existing step; decode without worker-local diagnostics."""
    if not isinstance(session.backend, CortexConfig):
        raise ValueError("managed optimizer capture requires CortexConfig")
    transport = job.client.transport
    job_id = transport.job_id
    if not isinstance(job_id, str) or not job_id:
        raise ValueError("managed optimizer capture requires an initialized parent job")
    if type(expected_dp_world_size) is not int or expected_dp_world_size < 1:
        raise ValueError("expected_dp_world_size must be a positive integer")
    if local_checkpoint:
        require_local_cortex_gateway(session.backend)
        if job.client.config.backend != session.backend:
            raise ValueError("local optimizer capture requires the same verified loopback client backend")

    from .cortex_optimizer_decode import decode_optimizer_capture

    output_dir = require_data_fast_path(output_dir, label="managed optimizer capture output")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "states"
    if destination.exists():
        raise FileExistsError("managed optimizer capture output already exists")
    with TemporaryDirectory(prefix="snapshots-", dir=output_dir) as scratch:
        scratch = Path(scratch)
        if local_checkpoint:
            before = _save_local_snapshot(
                job.client, scratch / "before", step=0, expected_dp_world_size=expected_dp_world_size
            )
            result = fwd_bwd_step(session, job, body, learning_rate=learning_rate)
            after = _save_local_snapshot(
                job.client, scratch / "after", step=1, expected_dp_world_size=expected_dp_world_size
            )
        else:
            from .cortex_checkpoint_export import export_checkpoint

            before_id = save_and_resolve_checkpoint_id(job.client, job_id, "resumable")
            result = fwd_bwd_step(session, job, body, learning_rate=learning_rate)
            after_id = save_and_resolve_checkpoint_id(job.client, job_id, "resumable")
            if before_id == after_id:
                raise ValueError("managed optimizer snapshots must have distinct durable checkpoint IDs")
            before = export_checkpoint(
                job.client, job_id, before_id, scratch / "before", expected_dp_world_size=expected_dp_world_size
            )
            after = export_checkpoint(
                job.client, job_id, after_id, scratch / "after", expected_dp_world_size=expected_dp_world_size
            )
        decoded = scratch / "decoded"
        manifest = Path(
            decode_optimizer_capture(
                before,
                after,
                decoded,
                expected_optimizer_config=optimizer_config,
                expected_learning_rate=learning_rate,
            )
        )
        if not manifest.is_file() or not manifest.resolve().is_relative_to(decoded.resolve()):
            raise ValueError("managed optimizer decoder must return its output manifest")
        step = json.loads(manifest.read_text()).get("step")
        if type(step) is not int or step != 1:
            raise ValueError("managed optimizer artifact must describe exactly step 1")
        relative_manifest = manifest.relative_to(decoded)
        decoded.rename(destination)
    return replace(result, optimizer_state_manifest=str(destination / relative_manifest))
