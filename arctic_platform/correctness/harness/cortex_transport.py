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

import os
from urllib.parse import urlencode
from urllib.parse import urlsplit
from uuid import uuid4

from arctic_platform import wire
from arctic_platform.client.transport import JobHandles
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.client.transports.cortex import _raise_for_status

from .cortex import CortexTrainingConfig

LOCAL_NATIVE_FORWARD_ENV = "ARCTIC_CORRECTNESS_LOCAL_NATIVE_FORWARD"
_NATIVE_ROLES = {"training": "training", "sampling": "sampling", "log_prob": "log_probability"}


class CorrectnessCortexTransport(CortexTransport):
    def _is_local_gateway(self) -> bool:
        parsed = urlsplit(self.config.base_url or "")
        return parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}

    def _build_session(self):
        session = super()._build_session()
        if self._is_local_gateway():
            session.trust_env = False
        return session

    def initialize(self, client_config: CortexTrainingConfig) -> JobHandles:
        self.hardware = client_config.hardware
        self._native_gateway = os.environ.get(LOCAL_NATIVE_FORWARD_ENV) == "1"
        self._native_children = {}
        self._native_bindings = {}
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

    def _validate_native_api(self, client_config) -> None:
        response = self.session.get(
            f"{self.config.base_url.rstrip('/')}/openapi.json",
            timeout=self.config.request_timeout,
            allow_redirects=False,
        )
        _raise_for_status(response)
        if response.status_code != 200:
            raise RuntimeError("local native gateway OpenAPI must return HTTP 200 without redirects")
        document = response.json()
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

    def _send(self, method: str, url: str, **kwargs) -> dict:
        if self._is_local_gateway():
            kwargs["allow_redirects"] = False
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
        return super()._send(method, url, **kwargs)

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
        response = self.session.post(url, timeout=self.config.request_timeout, allow_redirects=False, **kwargs)
        _raise_for_status(response)
        if response.status_code != 200:
            raise RuntimeError("local native gateway must return HTTP 200 without redirects")
        return self._native_result(request, response.content)

    async def acall(self, request) -> dict:
        if not getattr(self, "_native_gateway", False) or request.op not in {"forward", "log-probs"}:
            return await super().acall(request)
        url, kwargs = self._native_request(request)
        session = await self._ensure_asession()
        async with session.post(url, allow_redirects=False, **kwargs) as response:
            if response.status != 200:
                body = await response.text()
                raise RuntimeError(f"local native gateway HTTP {response.status}: {body}")
            return self._native_result(request, await response.read())

    async def _asend(self, method: str, url: str, **kwargs) -> dict:
        if self._is_local_gateway():
            kwargs["allow_redirects"] = False
        return await super()._asend(method, url, **kwargs)

    def list_checkpoints(self) -> list[dict]:
        if self.job_id is None:
            raise RuntimeError("checkpoint catalog requires an initialized Cortex job")
        result = self._send("GET", f"{self._prefix}/{self.job_id}/checkpoints")
        return result if isinstance(result, list) else result.get("checkpoints", [])

    def shutdown(self) -> None:
        try:
            super().shutdown()
        finally:
            self.job_id = None
            self._native_children = {}
            self._native_bindings = {}
            self.session.close()
