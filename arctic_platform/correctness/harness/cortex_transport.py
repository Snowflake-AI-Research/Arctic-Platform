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

"""Correctness-only DSS fields and an opt-in adapter for the existing local gateway."""

import asyncio
import json
import math
import os
import time
from urllib.parse import urlencode
from urllib.parse import urlsplit
from uuid import uuid4

from arctic_platform import wire
from arctic_platform.client.requests import list_checkpoints_request
from arctic_platform.client.transport import JobHandles
from arctic_platform.client.transport import Request
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.client.transports.cortex import _raise_for_status
from arctic_platform.client.transports.cortex import _Route
from arctic_platform.client.transports.cortex import _SendPolicy

from .cortex import CortexTrainingConfig

LOCAL_NATIVE_FORWARD_ENV = "ARCTIC_CORRECTNESS_LOCAL_NATIVE_FORWARD"
_NATIVE_ROLES = {"training": "training", "sampling": "sampling", "log_prob": "log_probability"}


class _LocalLoadRoute(_Route):
    def submitted(self, request, reply):
        if not isinstance(reply, dict):
            raise RuntimeError("local checkpoint load returned malformed submission metadata")
        request_id = reply.get("request_id")
        if request_id is not None:
            if not isinstance(request_id, str) or not request_id:
                raise RuntimeError("local checkpoint load returned no valid request_id")
            return request_id
        from .hosted import _local_load_metadata

        _local_load_metadata(reply, request.body["checkpoint_id"])
        return reply


class CorrectnessCortexTransport(CortexTransport):
    def _is_local_gateway(self) -> bool:
        parsed = urlsplit(self.config.base_url or "")
        return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}

    def initialize(self, client_config: CortexTrainingConfig) -> JobHandles:
        self.hardware = client_config.hardware
        self._native_gateway = os.environ.get(LOCAL_NATIVE_FORWARD_ENV) == "1"
        self._native_children = {}
        self._native_bindings = {}
        self._local_shutdown_parent = None
        self._local_shutdown_children = {}
        if self._native_gateway:
            from .hosted import require_local_cortex_gateway

            require_local_cortex_gateway(self.config)
            if JobHandles.from_config(client_config).any_set:
                raise ValueError(
                    "local native forward requires newly created, owned child jobs; reconnect is unsupported"
                )
            self._validate_native_api(client_config)
        jobs = super().initialize(client_config)
        if self._native_gateway:
            for role, gateway_role in _NATIVE_ROLES.items():
                token = getattr(jobs, role)
                if token is not None:
                    child = self._native_children.get(gateway_role)
                    if child is None or str(token) in self._native_bindings:
                        raise ValueError("local native forward received inconsistent child-job bindings")
                    self._native_bindings[str(token)] = (role, child)
        return jobs

    async def _local_session(self):
        import aiohttp

        if self._session is None:
            self._session = aiohttp.ClientSession(headers=self._auth_headers(), trust_env=False)
        return self._session

    async def _native_http(self, method: str, url: str, **kwargs) -> bytes:
        import aiohttp

        session = await self._local_session()
        async with session.request(
            method,
            url,
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout),
            allow_redirects=False,
            **kwargs,
        ) as response:
            if response.status >= 400:
                await _raise_for_status(response)
            if response.status != 200:
                raise RuntimeError("local native gateway must return HTTP 200 without redirects")
            return await response.read()

    def _validate_native_api(self, client_config) -> None:
        payload = self._on_loop(self._native_http("GET", f"{self.config.base_url.rstrip('/')}/openapi.json")).result()
        document = json.loads(payload)
        paths = document.get("paths") if isinstance(document, dict) else None
        required = []
        if client_config.gpus_for("training") > 0:
            required.append("/forward")
        if client_config.gpus_for("log_prob") > 0:
            required.append("/log-probs")
        for path in required:
            path_config = paths.get(path) if isinstance(paths, dict) else None
            operation = path_config.get("post") if isinstance(path_config, dict) else None
            parameters = operation.get("parameters", []) if isinstance(operation, dict) else []
            if not any(
                isinstance(param, dict)
                and param.get("name") == "job_id"
                and param.get("in") == "query"
                and param.get("required") is True
                for param in parameters
            ):
                raise RuntimeError(f"installed local gateway lacks supported POST {path}?job_id before job allocation")

    async def _send(self, method: str, url: str, **kwargs) -> dict:
        local_roles = None
        if self._is_local_gateway():
            kwargs["allow_redirects"] = False
            await self._local_session()
        if method == "POST" and url == self._prefix:
            payload = dict(kwargs["json"])
            if getattr(self, "_native_gateway", False):
                children = {}
                configs = []
                for sub in payload["sub_job_configs"]:
                    role = sub.get("job_type")
                    if role not in _NATIVE_ROLES.values() or role in children or sub.get("job_id") is not None:
                        raise ValueError("local native forward requires one fresh public job config per role")
                    child = str(uuid4())
                    children[role] = child
                    configs.append({**sub, "job_id": child})
                self._native_children = children
                payload["sub_job_configs"] = configs
            if getattr(self, "hardware", None) is not None:
                payload["hardware"] = self.hardware
            kwargs["json"] = payload
            if self._is_local_gateway():
                local_roles = [sub.get("job_type") for sub in payload["sub_job_configs"]]
                if (
                    not local_roles
                    or any(role not in _NATIVE_ROLES.values() for role in local_roles)
                    or len(set(local_roles)) != len(local_roles)
                ):
                    raise RuntimeError("local gateway create requires distinct supported child roles")
        result = await super()._send(method, url, **kwargs)
        if local_roles is not None:
            parent = result.get("job_id") if isinstance(result, dict) else None
            if not isinstance(parent, str) or not parent:
                raise RuntimeError("local gateway create returned no owned parent identity")
            segments = {gateway_role: role for role, gateway_role in _NATIVE_ROLES.items()}
            self._local_shutdown_parent = parent
            self._local_shutdown_children = {role: f"{parent}:{segments[role]}:0" for role in local_roles}
        return result

    def _native_request(self, request):
        binding = self._native_bindings.get(str(request.job_id))
        if self.job_id is None or binding is None:
            raise ValueError("local native request must target this client's owned child job")
        role, child = binding
        allowed = {"training"} if request.op == "forward" else {"log_prob", "sampling"}
        if role not in allowed:
            raise ValueError(f"local native {request.op} cannot target a {role} child job")
        url = f"{self.config.base_url.rstrip('/')}/{request.op}?{urlencode({'job_id': child})}"
        body = {key: value for key, value in request.body.items() if value is not None}
        if request.op == "forward":
            return url, {"data": wire.dumps(body), "headers": {"Content-Type": "application/octet-stream"}}
        return url, {"json": body}

    def _native_result(self, request, payload):
        result = wire.loads(payload)
        if not isinstance(result, dict):
            raise ValueError("local native gateway response must be a DSSST1 object")
        child = self._native_bindings[str(request.job_id)][1]
        if "job_id" in result and result["job_id"] != child:
            raise ValueError("local native gateway response names a different child job")
        return result

    def call(self, request) -> dict:
        if not getattr(self, "_native_gateway", False) or request.op not in {"forward", "log-probs"}:
            return super().call(request)
        url, kwargs = self._native_request(request)
        payload = self._on_loop(self._native_http("POST", url, **kwargs)).result()
        return self._native_result(request, payload)

    async def acall(self, request) -> dict:
        if not getattr(self, "_native_gateway", False) or request.op not in {"forward", "log-probs"}:
            return await super().acall(request)
        url, kwargs = self._native_request(request)
        payload = await asyncio.wrap_future(self._on_loop(self._native_http("POST", url, **kwargs)))
        return self._native_result(request, payload)

    def list_checkpoints(self) -> list[dict]:
        if self.job_id is None:
            raise RuntimeError("checkpoint catalog requires an initialized Cortex job")
        return self.call(list_checkpoints_request(self.job_id))

    def load_checkpoint(self, checkpoint_id: str, *, stage_info: dict | None = None) -> dict:
        return self.call(
            Request(
                "load-checkpoint",
                self.jobs.training,
                {"checkpoint_id": checkpoint_id, "stage_info": stage_info},
            )
        )

    def _submit(self, request):
        if request.op != "load-checkpoint" or not self._is_local_gateway():
            return super()._submit(request)
        from .hosted import _local_load_metadata
        from .hosted import _local_stage_info
        from .hosted import require_local_checkpoint_id
        from .hosted import require_local_cortex_gateway

        require_local_cortex_gateway(self.config)
        parent = getattr(self, "_local_shutdown_parent", None)
        token = None if parent is None else f"{parent}:training:0"
        if (
            parent is None
            or self.job_id != parent
            or getattr(self, "_local_shutdown_children", {}).get("training") != token
            or self.jobs.training != token
            or request.job_id != token
        ):
            raise ValueError("local checkpoint load requires this client's newly owned training child")
        body = request.body
        if request.binary or not isinstance(body, dict) or set(body) - {"checkpoint_id", "stage_info"}:
            raise ValueError(
                "local checkpoint load accepts checkpoint_id/stage_info only; no path, step or source_job_id"
            )
        checkpoint_id = require_local_checkpoint_id(body.get("checkpoint_id"))
        stage_info = _local_stage_info(body.get("stage_info"), checkpoint_id)
        canonical = Request(
            "load-checkpoint",
            request.job_id,
            {
                "checkpoint_id": checkpoint_id,
                "stage_info": stage_info,
            },
        )
        route = _LocalLoadRoute(
            method="POST",
            path="/{job}/load",
            poll=True,
            policy=_SendPolicy(max_retries=0),
            normalize=lambda result: _local_load_metadata(result, checkpoint_id),
        )
        url, sends = route.prepare(self._prefix, canonical)
        return route, self._on_loop(self._deliver(route, canonical, url, sends))

    def _shutdown_send(self, method: str, url: str, deadline: float) -> dict:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("local gateway owned-child cancellation deadline expired")
        policy = _SendPolicy(max_retries=0, timeout=remaining)
        result = self._on_loop(self._send(method, url, policy=policy)).result()
        if time.monotonic() >= deadline:
            raise TimeoutError("local gateway owned-child cancellation deadline expired")
        return result

    def _wait_local_cancelled(self, parent: str) -> None:
        timeout = self.config.poll_timeout
        interval = self.poll_interval
        if any(not math.isfinite(value) or value <= 0 for value in (timeout, interval, self.config.request_timeout)):
            raise ValueError("local gateway shutdown timeout and poll interval must be finite and positive")
        deadline = time.monotonic() + timeout
        if not math.isfinite(deadline):
            raise ValueError("local gateway shutdown deadline must be finite")
        expected = self._local_shutdown_children
        if not expected or self.job_id != parent:
            raise RuntimeError("local gateway shutdown has inconsistent owned parent/children")
        url = f"{self._prefix}/{parent}"
        self._shutdown_send("POST", f"{url}:cancel", deadline)
        while True:
            job = self._shutdown_send("GET", url, deadline)
            if not isinstance(job, dict) or job.get("job_id") != parent:
                raise RuntimeError("local gateway cancellation returned a different parent")
            if job.get("reason") or job.get("status") not in ("cancelled", "pending", "running"):
                raise RuntimeError(f"local gateway cancellation failed: {job!r}")
            children = job.get("sub_jobs")
            if not isinstance(children, list) or len(children) != len(expected):
                raise RuntimeError("local gateway cancellation has missing or unexpected owned children")
            seen = set()
            states = []
            for child in children:
                if not isinstance(child, dict):
                    raise RuntimeError("local gateway cancellation returned a malformed child")
                role = child.get("job_type")
                if (
                    not isinstance(role, str)
                    or role not in expected
                    or role in seen
                    or child.get("sub_job_id") != expected[role]
                ):
                    raise RuntimeError("local gateway cancellation returned duplicate or mismatched owned children")
                state = child.get("status")
                if child.get("reason") or state not in ("cancelled", "pending", "running"):
                    raise RuntimeError(f"local gateway child cancellation failed: {child!r}")
                seen.add(role)
                states.append(state)
            # Parent cancellation is only an ACK; each child must finish the awaited destroy.
            if job["status"] == "cancelled" and all(state == "cancelled" for state in states):
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"local gateway parent {parent} did not release its owned children")
            time.sleep(min(interval, remaining))

    def shutdown(self) -> None:
        try:
            parent = getattr(self, "_local_shutdown_parent", None)
            if parent is not None and self._is_local_gateway():
                self._wait_local_cancelled(parent)
            else:
                super().shutdown()
        finally:
            self.job_id = None
            self._native_children = {}
            self._native_bindings = {}
            self._local_shutdown_parent = None
            self._local_shutdown_children = {}
            self._on_loop(self._close_session()).result()
