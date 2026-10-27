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

# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0

"""Owned local DSS gateway with full-state, same-allocation checkpoint restoration."""

from __future__ import annotations

import asyncio
import contextlib
import re
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from .storage import require_data_fast_path

NATIVE_STORE_SCHEMA = "correctness-native-checkpoint-v1"
_TAG = re.compile(r"global_step(0|[1-9][0-9]*)")


@dataclass(frozen=True)
class _SavedCheckpoint:
    store: dict
    job_binding: dict
    tag: str
    version: int
    source_child: str


def _metadata(
    result: dict,
    *,
    child: str,
    path: str,
    tag: str | None = None,
    version: int | None = None,
) -> tuple[str, int]:
    if not isinstance(result, dict) or result.get("checkpoint_path") != path:
        raise RuntimeError("native checkpoint response does not name the owned checkpoint directory")
    if result.get("job_id") != child or result.get("stage_path") not in (None, ""):
        raise RuntimeError("native checkpoint response has a different child or unexpected stage metadata")
    match = _TAG.fullmatch(result.get("checkpoint_tag", "")) if isinstance(result.get("checkpoint_tag"), str) else None
    observed = result.get("version")
    if match is None or type(observed) is not int or observed != int(match.group(1)):
        raise RuntimeError("native checkpoint response has inconsistent checkpoint tag and step")
    if (tag is not None and result["checkpoint_tag"] != tag) or (version is not None and observed != version):
        raise RuntimeError("native checkpoint load restored a different training step")
    return result["checkpoint_tag"], observed


def _training_config(config) -> dict:
    raw = deepcopy(config.model_dump(mode="json", by_alias=True, exclude_none=True))
    training = raw.get("training_config")
    if (
        str(raw.get("job_type")) != "training"
        or not isinstance(training, dict)
        or training.get("load_optimizer_states") is not True
        or raw.get("source_checkpoint_info") is not None
        or raw.get("resume_from_checkpoint") is not None
        or raw.get("resume_checkpoint_tag") is not None
    ):
        raise ValueError("native correctness resume requires a fresh full-state training configuration")
    raw.pop("job_id", None)
    return raw


def install_checkpoint_routes(gw, store) -> None:
    """Compose the installed gateway's existing job creation and public zone operations."""
    from fastapi import Body
    from fastapi import HTTPException
    from pydantic import BaseModel
    from pydantic import ConfigDict

    class SaveBody(BaseModel):
        model_config = ConfigDict(extra="forbid")
        path: str
        checkpoint_type: str

    class LoadBody(BaseModel):
        model_config = ConfigDict(extra="forbid")
        checkpoint_id: str

    bindings: dict[str, dict] = {}
    checkpoints: dict[str, _SavedCheckpoint] = {}
    operation_lock = asyncio.Lock()
    original_create = gw._create_child_job
    original_lifespan = gw.app.router.lifespan_context

    async def create_child(config):
        try:
            raw = _training_config(config)
        except ValueError:
            raw = None
        if raw is not None:
            # The retained FAST store is node-local: each node needs its common model state.
            training = raw["training_config"]
            ds_config = deepcopy(training.get("ds_config") or {})
            checkpoint = deepcopy(ds_config.get("checkpoint") or {})
            checkpoint["use_node_local_storage"] = True
            ds_config["checkpoint"] = checkpoint
            training["ds_config"] = ds_config
            config = type(config).model_validate({**raw, "job_id": config.job_id})
        child = await original_create(config)
        # Non-training and weights-only configurations still use the unmodified service.
        if raw is not None:
            zone = gw._zone_for_job(child)
            bindings[child] = {
                "config": raw,
                "num_gpus": zone.num_gpus,
                "nodes": deepcopy([node.model_dump(mode="json") for node in zone.nodes]),
            }
        return child

    def owned_training(child: str):
        from dss.ray_dss import gateway_snowapi

        parents = [
            parent
            for parent in gateway_snowapi._parents.values()
            if parent.state == "running"
            and not parent.cancel_requested
            and (training := parent.children.get("training")) is not None
            and training.job_id == child
            and training.status == "ready"
        ]
        if len(parents) != 1 or child not in bindings:
            raise HTTPException(
                409,
                "checkpoint operation requires one newly owned full-state training child",
            )
        zone = gw._zone_for_job(child)
        binding = bindings[child]
        current = {
            "num_gpus": zone.num_gpus,
            "nodes": [node.model_dump(mode="json") for node in zone.nodes],
        }
        if any(binding[key] != current[key] for key in current):
            raise HTTPException(409, "checkpoint operation's training allocation changed")
        return zone, binding

    async def save_checkpoint(job_id: str, body: SaveBody = Body(...)):
        async with operation_lock:
            zone, binding = owned_training(job_id)
            if body.checkpoint_type != "resumable":
                raise HTTPException(
                    400,
                    "native checkpoint catalog accepts full resumable checkpoints only",
                )
            root = require_data_fast_path(body.path, label="native resumable save")
            if str(root) != body.path:
                raise HTTPException(400, "native resumable save requires a canonical FAST directory")
            retained = await asyncio.to_thread(store.create, root)
            print(
                f"Retained native checkpoint store: FAST={retained['path']}; namespace={retained['native_path']}",
                flush=True,
            )
            await asyncio.to_thread(store.verify, retained)
            result = await gw._zone_request_and_wait(
                zone,
                "POST",
                f"/api/v0/jobs/{job_id}/save",
                json_body={"path": str(root), "checkpoint_type": "resumable"},
                no_keepalive=True,
            )
            tag, version = _metadata(result, child=job_id, path=str(root))
            await asyncio.to_thread(store.verify, retained)
            checkpoint_id = f"cp_{uuid4().hex}"
            checkpoints[checkpoint_id] = _SavedCheckpoint(
                store=deepcopy(retained),
                job_binding=deepcopy(binding),
                tag=tag,
                version=version,
                source_child=job_id,
            )
            return {
                **result,
                "checkpoint_id": checkpoint_id,
                "correctness_native_store": NATIVE_STORE_SCHEMA,
            }

    async def load_checkpoint(job_id: str, body: LoadBody = Body(...)):
        async with operation_lock:
            zone, binding = owned_training(job_id)
            saved = checkpoints.get(body.checkpoint_id)
            if saved is None:
                raise HTTPException(
                    404,
                    "checkpoint is absent from this owned gateway's retained catalog",
                )
            if saved.source_child == job_id or saved.job_binding != binding:
                raise HTTPException(
                    409,
                    "checkpoint resume requires a fresh child with the identical model and layout",
                )
            if saved.source_child in gw._jobs:
                raise HTTPException(409, "saving child must be destroyed before checkpoint resume")
            await asyncio.to_thread(store.verify, saved.store)
            result = await gw._zone_request_and_wait(
                zone,
                "POST",
                f"/api/v0/jobs/{job_id}/load",
                json_body={
                    "checkpoint_id": saved.tag,
                    "source_job_id": saved.store["source_job_id"],
                },
                no_keepalive=True,
            )
            _metadata(
                result,
                child=job_id,
                path=saved.store["native_path"],
                tag=saved.tag,
                version=saved.version,
            )
            if result.get("checkpoint_id") != saved.tag:
                raise RuntimeError("native checkpoint load returned a different requested tag")
            await asyncio.to_thread(store.verify, saved.store)
            return {
                **result,
                "checkpoint_id": body.checkpoint_id,
                "correctness_native_store": NATIVE_STORE_SCHEMA,
            }

    @contextlib.asynccontextmanager
    async def lifespan(app):
        async with original_lifespan(app):
            yield
        # Upstream teardown can swallow worker failures, so process exit alone never permits deletion.
        print(
            "Native checkpoint stores retained for allocation reset or independently verified owned cleanup.",
            flush=True,
        )

    gw._create_child_job = create_child
    gw.app.router.lifespan_context = lifespan
    # Inner request models must be concrete annotations for FastAPI, not unresolved local names.
    save_checkpoint.__annotations__["body"] = SaveBody
    load_checkpoint.__annotations__["body"] = LoadBody
    gw.app.add_api_route("/correctness/save-checkpoint", save_checkpoint, methods=["POST"])
    gw.app.add_api_route("/correctness/load-checkpoint", load_checkpoint, methods=["POST"])


def main() -> None:
    from dss.ray_dss import gateway_server

    from .checkpoint_store import NativeCheckpointStore
    from .native_lora import gateway_scope

    hostfiles = [sys.argv[index + 1] for index, arg in enumerate(sys.argv[:-1]) if arg == "--hostfile"]
    if len(hostfiles) != 1:
        raise ValueError("owned correctness gateway requires one explicit --hostfile")
    hosts = [sys.argv[index + 1] for index, arg in enumerate(sys.argv[:-1]) if arg == "--host"]
    if hosts != ["127.0.0.1"]:
        raise ValueError("owned correctness gateway must bind explicitly to 127.0.0.1")
    store = NativeCheckpointStore(Path(hostfiles[0]))
    install_checkpoint_routes(gateway_server, store)
    with gateway_scope(gateway_server):
        gateway_server.main()


if __name__ == "__main__":
    main()
