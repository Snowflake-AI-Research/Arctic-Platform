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

"""Route the native optimizer owner split between the owned gateway, the training zone head and the runner.

DSS keeps parent records in the gateway process and the training ``ZoneManager`` with its Ray driver in the zone
head process, so ``optimizer.resolve_native_optimizer_parent`` and ``optimizer.open_native_optimizer_zone_scope``
each run in their own owner behind a loopback route. The runner holds only the source-bound request, the client
shutdown fence and the HTTP calls. Admission, collection, consolidation and cleanup stay in ``optimizer``.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib
import importlib.util
import inspect
import json
import threading
import traceback
import uuid
from pathlib import Path

PARENT_ROUTE = "/correctness/native-optimizer/parent"
OPEN_ROUTE = "/correctness/native-optimizer/open"
SUPPLY_ROUTE = "/correctness/native-optimizer/supply"
CLOSE_ROUTE = "/correctness/native-optimizer/close"
# The upper bound ``optimizer._native_scope_sources`` admits; one EP16 consolidation read stays inside it.
RPC_TIMEOUT_SECONDS = 900


class NativeOwnerError(RuntimeError):
    """A route refusal; ``cleanup_complete`` is False when the owner still holds armed workers."""

    def __init__(self, route: str, detail):
        detail = detail if isinstance(detail, dict) else {"error": str(detail)}
        super().__init__(f"{route}: {detail.get('error')}")
        self.cleanup_complete = detail.get("cleanup_complete")


def _descriptor(path) -> dict:
    return {"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def _installed_source(module: str) -> dict:
    # Owner modules are pinned by installed origin; the runner never imports the gateway or zone servers.
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.origin:
        raise ValueError(f"native optimizer owner module {module} is not installed")
    return _descriptor(spec.origin)


def _callback_source(callback) -> dict:
    from arctic_platform.correctness.harness.optimizer import _native_callback_origin

    return _descriptor(inspect.getsourcefile(_native_callback_origin(callback)))


def owner_split_source() -> dict:
    """The source descriptor of the runner scope, its supplier and the zone-side parent verifier."""
    return _callback_source(owner_split_scope)


def owner_split_request(cfg, *, checkpoint_parent: Path, hostfile: Path) -> dict:
    """Bind one EP training width to the producer code installed here and to the admitted allocation hosts."""
    import ray
    import ray.actor
    from deepspeed.runtime.engine import DeepSpeedEngine
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    from dss.ray_dss.jobs.checkpointing import build_checkpointer

    from arctic_platform.correctness.harness import optimizer
    from arctic_platform.correctness.harness.storage import parse_hostfile

    checkpointer = type(build_checkpointer(ep_size=cfg.training.get("ep_size", 1)))
    request = {
        "schema": "native-optimizer-owner-split-request-v1",
        "source_files": {
            "training": _installed_source("dss.ray_dss.jobs.gpu.training"),
            "expert_checkpointer": _callback_source(checkpointer.save_resumable),
            "base_checkpointer": _callback_source(checkpointer.save_adapters),
            "engine": _callback_source(DeepSpeedEngine.save_checkpoint),
            "zero": _callback_source(DeepSpeedZeroOptimizer.state_dict),
            "zero_base": _callback_source(DeepSpeedZeroOptimizer.backward),
        },
        "route_sources": {
            "gateway_snowapi": _installed_source("dss.ray_dss.gateway_snowapi"),
            "gateway": _installed_source("dss.ray_dss.gateway_server"),
            "manager": _installed_source("dss.ray_dss.zones.manager"),
            # ``optimizer._native_scope_origin`` pins these two by the wrapper's own file, without unwrapping.
            "ray_get": _descriptor(inspect.getsourcefile(ray.get)),
            "ray_actor": _descriptor(inspect.getsourcefile(ray.actor.ActorMethod.remote)),
            "ray_is_initialized": _callback_source(ray.is_initialized),
        },
        "helper_source": _callback_source(optimizer.make_native_optimizer_scope),
        "verify_parent_binding_source": owner_split_source(),
        "checkpoint_parent": str(checkpoint_parent),
        "reserve_bytes": 0,
        "rpc_timeout_seconds": RPC_TIMEOUT_SECONDS,
        "allocation_hosts": [entry.host for entry in parse_hostfile(hostfile)],
        "expected_world_size": cfg.n_gpus,
    }
    optimizer._native_split_request(request)
    return request


def owner_split_execution(cfg) -> dict:
    """``execute`` keywords for one EP width; checkpoint leases live under the run's own workdir."""
    from arctic_platform.correctness.harness.gateway import _allocation_hostfile
    from arctic_platform.correctness.harness.storage import correctness_workdir

    hostfile = _allocation_hostfile()
    if hostfile is None:
        raise ValueError("native optimizer owner split requires the admitted allocation hostfile")
    workdir = correctness_workdir("dss-correctness-")
    request = owner_split_request(cfg, checkpoint_parent=workdir, hostfile=hostfile)
    return {
        "workdir": workdir,
        "native_optimizer_scope": owner_split_scope(request),
        "native_optimizer_scope_source": owner_split_source(),
    }


def _detail(error: BaseException, **fields) -> dict:
    traceback.print_exception(error)
    return {"error": "".join(traceback.format_exception_only(error)).strip(), **fields}


def _reply(route: str, response) -> dict:
    if response.status_code != 200:
        try:
            detail = response.json().get("detail")
        except ValueError:
            detail = response.text
        raise NativeOwnerError(route, detail)
    return response.json()


def _post(base_url: str, route: str, body: dict) -> dict:
    import requests

    with requests.Session() as session:
        session.trust_env = False
        response = session.post(f"{base_url.rstrip('/')}{route}", json=body, timeout=(30, None), allow_redirects=False)
    return _reply(route, response)


def install_gateway_route(gateway) -> None:
    """Resolve the parent binding in the owned gateway, the only process holding DSS parent records."""
    from fastapi import Body
    from fastapi import HTTPException

    from arctic_platform.correctness.harness.optimizer import resolve_native_optimizer_parent

    def parent(body: dict = Body(...)):
        try:
            binding = resolve_native_optimizer_parent(
                body["request"],
                body["parent_job_id"],
                gateway_snowapi=importlib.import_module("dss.ray_dss.gateway_snowapi"),
                gateway=gateway,
            )
            return {"binding": binding, "zone_url": gateway._zones[binding["zone_id"]].url}
        except Exception as error:
            raise HTTPException(409, _detail(error)) from error

    gateway.app.add_api_route(PARENT_ROUTE, parent, methods=["POST"])


def install_zone_routes(app) -> None:
    """Open, supply and close owner-split scopes in the zone head holding ``app.state.manager`` and its Ray driver."""
    from fastapi import Body
    from fastapi import HTTPException

    from arctic_platform.client import CortexConfig
    from arctic_platform.correctness.harness.gateway import require_local_cortex_gateway
    from arctic_platform.correctness.harness.optimizer import open_native_optimizer_zone_scope

    # A closed scope whose owned cleanup failed stays registered so another close can retry it.
    scopes = {}
    lock = threading.Lock()

    def open_scope(body: dict = Body(...)):
        request, binding, gateway_url = body["request"], body["binding"], body["gateway_url"]
        with lock:
            try:
                require_local_cortex_gateway(CortexConfig(base_url=gateway_url))

                def verify_parent_binding(current):
                    return _post(
                        gateway_url, PARENT_ROUTE, {"request": request, "parent_job_id": current["parent_job_id"]}
                    )["binding"]

                scope = open_native_optimizer_zone_scope(
                    request,
                    binding,
                    manager=app.state.manager,
                    ray=importlib.import_module("ray"),
                    verify_parent_binding=verify_parent_binding,
                )
            except Exception as error:
                raise HTTPException(409, _detail(error, cleanup_complete=True)) from error
            try:
                supplier = scope.__enter__()
            except Exception as error:
                if not scope.cleanup_complete:
                    with contextlib.suppress(Exception):
                        scope.cleanup()
                raise HTTPException(409, _detail(error, cleanup_complete=scope.cleanup_complete)) from error
            scope_id = uuid.uuid4().hex
            scopes[scope_id] = scope, supplier
        return {"scope_id": scope_id}

    def supply(body: dict = Body(...)):
        with lock:
            scope, supplier = scopes.get(body["scope_id"], (None, None))
            if supplier is None:
                raise HTTPException(404, {"error": "native optimizer scope is not open"})
            try:
                return supplier(Path(body["root"]), step=body["step"])
            except Exception as error:
                raise HTTPException(409, _detail(error)) from error

    def close(body: dict = Body(...)):
        with lock:
            scope, _ = scopes.pop(body["scope_id"], (None, None))
            if scope is None:
                raise HTTPException(404, {"error": "native optimizer scope is not open"})
            errors = []
            try:
                if body["failed"]:
                    error = RuntimeError("runner capture failed inside the native optimizer scope")
                    scope.__exit__(RuntimeError, error, None)
                else:
                    scope.__exit__(None, None, None)
            except Exception as error:
                errors.append(_detail(error)["error"])
            if not scope.cleanup_complete:
                try:
                    scope.cleanup()
                except Exception as error:
                    errors.append(_detail(error)["error"])
            if not scope.cleanup_complete:
                scopes[body["scope_id"]] = scope, None
            return {"cleanup_complete": scope.cleanup_complete, "errors": errors}

    app.add_api_route(OPEN_ROUTE, open_scope, methods=["POST"])
    app.add_api_route(SUPPLY_ROUTE, supply, methods=["POST"])
    app.add_api_route(CLOSE_ROUTE, close, methods=["POST"])


def owner_split_scope(request: dict):
    """Return the runner's ``native_optimizer_scope``; child shutdown is fenced until the zone head cleans up."""
    from arctic_platform.correctness.harness.gateway import require_local_cortex_gateway
    from arctic_platform.correctness.harness.optimizer import _SCOPE_STATE
    from arctic_platform.correctness.harness.optimizer import _native_split_request

    request = json.loads(json.dumps(request, allow_nan=False))
    _native_split_request(request)

    @contextlib.contextmanager
    def scope(session, job):
        require_local_cortex_gateway(session.backend)
        client = job.client
        if client.config.backend != session.backend or job.job_type != "training":
            raise ValueError("native owner split requires the session's local training client")
        if _SCOPE_STATE in vars(client) or "shutdown" in vars(client):
            raise ValueError("native owner split requires an unmodified client shutdown lifecycle")
        gateway_url = session.backend.base_url
        resolved = _post(gateway_url, PARENT_ROUTE, {"request": request, "parent_job_id": client.transport.job_id})
        zone_url = resolved["zone_url"]
        fence = {"complete": False}
        shutdown = client.shutdown

        def guarded_shutdown(*args, **kwargs):
            if not fence["complete"]:
                raise RuntimeError("native optimizer owned cleanup incomplete; child release is blocked")
            return shutdown(*args, **kwargs)

        def release():
            if vars(client).get("shutdown") is not guarded_shutdown:
                raise ValueError("native owner split found a foreign client shutdown hook")
            del vars(client)["shutdown"]
            del vars(client)[_SCOPE_STATE]
            fence["complete"] = True

        client.shutdown = guarded_shutdown
        vars(client)[_SCOPE_STATE] = {"shutdown": guarded_shutdown}
        try:
            opened = _post(
                zone_url, OPEN_ROUTE, {"request": request, "binding": resolved["binding"], "gateway_url": gateway_url}
            )
        except NativeOwnerError as error:
            if error.cleanup_complete:
                release()
            raise
        scope_id = opened["scope_id"]

        def supplier(root, *, step):
            return _post(zone_url, SUPPLY_ROUTE, {"scope_id": scope_id, "root": str(root), "step": step})

        primary = None
        try:
            yield supplier
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                closed = _post(zone_url, CLOSE_ROUTE, {"scope_id": scope_id, "failed": primary is not None})
                if closed["cleanup_complete"]:
                    release()
                if closed["errors"] or not closed["cleanup_complete"]:
                    raise RuntimeError("native optimizer owned cleanup failed: " + "; ".join(closed["errors"]))
            except Exception as error:
                if primary is None:
                    raise
                primary.add_note(f"Native optimizer owner close also failed: {error}")

    return scope
