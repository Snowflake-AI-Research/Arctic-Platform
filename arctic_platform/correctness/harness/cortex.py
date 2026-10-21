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


"Correctness gateway client, fixture admission, and checkpoint transport."

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlencode
from urllib.parse import urlsplit
from uuid import uuid4

from pydantic import BaseModel
from typing_extensions import Self

from arctic_platform import wire
from arctic_platform._dependency_groups import require_any_dep_group
from arctic_platform.client import ArcticClient
from arctic_platform.client import ArcticClientConfig
from arctic_platform.client import CortexConfig
from arctic_platform.client.requests import list_checkpoints_request
from arctic_platform.client.transport import JobHandles
from arctic_platform.client.transport import initialize_or_cleanup
from arctic_platform.client.transports.cortex import CortexTransport
from arctic_platform.client.transports.cortex import _raise_for_status
from arctic_platform.client.transports.cortex import _SendPolicy

from .config import ReviewedClientConfig
from .config import TestSpec
from .config import hash_directory
from .storage import require_data_fast_path


class CortexTrainingConfig(ReviewedClientConfig):
    """Keep DSS's extra creation fields outside the production config adapter."""

    hardware: str | None = None

    @classmethod
    def from_reviewed(cls, native: ArcticClientConfig, **updates) -> Self:
        raw = {**native.model_dump(), **updates}
        for key in ("training", "sampling"):
            if isinstance(raw.get(key), BaseModel):
                raw[key] = raw[key].model_dump()
        return cls.model_validate(raw)

    def _cortex_training_sub_job(self) -> dict:
        sub = super()._cortex_training_sub_job()
        extra = deepcopy(self.training.ds_worker_config or {})
        activation = extra.pop("enable_gradient_checkpointing", None)
        if activation is not None:
            extra["activation_checkpointing"] = activation
        debug = dict(extra.get("debug") or {})
        debug.setdefault("full_determinism", self.training.full_determinism)
        extra["debug"] = debug
        sub["training_config"] = {**extra, **sub["training_config"]}
        return sub


class CorrectnessClient(ArcticClient):
    def __init__(self, config: ArcticClientConfig) -> None:
        if config.sampling_gpus or config.log_prob_gpus:
            raise ValueError("part one supports training-only gradient and optimizer checks")
        if isinstance(config, ReviewedClientConfig) and isinstance(config.backend, CortexConfig):
            config = CortexTrainingConfig.from_reviewed(config)
        elif isinstance(config, ReviewedClientConfig):
            for role in ("training", "sampling", "log_prob"):
                if config.gpus_for(role) > 0:
                    config.to_onprem(role)
        if not isinstance(config, CortexTrainingConfig) or not isinstance(config.backend, CortexConfig):
            super().__init__(config)
            return
        require_any_dep_group("cortex")

        self.config = config
        self.transport = CorrectnessCortexTransport(config.backend)
        self.jobs = initialize_or_cleanup(self.transport, client_config=config)


MANIFEST = "_neutrino_manifest.json"
TOKENIZER_JSON_SIDECARS = frozenset(
    {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"}
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe_fixture(spec: TestSpec, root: Path) -> dict:
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"reviewed reference checkpoint is absent: {root}")
    # Legacy reduced specs predate tokenizer sidecars; publication still binds every file below.
    if (
        hash_directory(root) != spec.model.content_hash
        and hash_directory(root, exclude_names=TOKENIZER_JSON_SIDECARS) != spec.model.content_hash
    ):
        raise ValueError("reference files do not match the reviewed spec's file-name/size hash")
    config = json.loads((root / "config.json").read_text())
    text = config.get("text_config", config)
    if text.get("num_hidden_layers") != spec.model.num_hidden_layers:
        raise ValueError("reference layer count does not match the reviewed spec")
    files = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.name == MANIFEST or any(part in {".cache", ".git"} for part in rel.parts):
            continue
        if not path.is_file():
            continue
        files.append({"path": rel.as_posix(), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    names = {item["path"] for item in files}
    if not any(name.endswith(".safetensors") for name in names) or "tokenizer_config.json" not in names:
        raise ValueError("reference checkpoint requires safetensors weights and tokenizer sidecars")
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    basename = re.sub(r"[^A-Za-z0-9_-]", "-", Path(spec.model.cache_path).name)
    return {
        "schema_version": 1,
        "config_id": spec.config_id,
        "reference_path": spec.model.cache_path,
        "reviewed_hash": spec.model.content_hash,
        "fixture_sha256": digest,
        "model_name": f"ap-correctness/{basename}-{digest}",
        "files": files,
    }


def prepare_fixture(spec: TestSpec, staging_root: Path) -> tuple[Path, dict]:
    staging_root = require_data_fast_path(staging_root, label="Cortex fixture staging")
    source = Path(spec.model.cache_path).resolve()
    if staging_root == source or source in staging_root.parents:
        raise ValueError("staging must be outside the reviewed reference checkpoint")
    binding = describe_fixture(spec, source)
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="ap-cortex-fixture-", dir=staging_root))
    for item in binding["files"]:
        relative = Path(item["path"])
        destination = staging / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, destination)
    if describe_fixture(spec, staging) != binding:
        raise ValueError("reference files changed while preparing publication")
    return staging, binding


def validate_publication(publication: dict, binding: dict) -> dict:
    """Check a publisher's file receipt without prescribing its stage or authentication."""
    stage_path = publication.get("stage_path")
    if (
        not isinstance(stage_path, str)
        or not stage_path.startswith("@")
        or not stage_path.endswith(f"/{binding['model_name']}")
        or publication.get("files") != binding["files"]
    ):
        raise ValueError("publication does not match the reference fixture")
    return {"stage_path": stage_path, "files": publication["files"]}


def load_fixture(receipt: Path, spec: TestSpec) -> dict:
    binding = json.loads(Path(receipt).read_text())
    publication = binding.get("publication")
    if not isinstance(publication, dict):
        raise ValueError("fixture binding has no successful publication receipt")
    actual = describe_fixture(spec, Path(spec.model.cache_path))
    if any(binding.get(key) != value for key, value in actual.items()):
        raise ValueError("reference files or model identity no longer match the published fixture")
    validate_publication(publication, actual)
    return binding


def load_local_fixture(backend: CortexConfig, spec: TestSpec) -> dict:
    """Validate local weights and gateway health without attesting the server backend."""
    import requests

    try:
        return _load_local_fixture(backend, spec)
    except (requests.RequestException, OSError) as exc:
        raise ValueError(f"local Cortex fixture verification failed: {type(exc).__name__}: {exc}") from exc


def _load_local_fixture(backend: CortexConfig, spec: TestSpec) -> dict:
    import requests

    parsed = urlsplit(backend.base_url or "")
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or backend.host
        or backend.pat is not None
    ):
        raise ValueError("local Cortex fixture requires an unauthenticated loopback HTTP base URL")
    base_url = backend.base_url.rstrip("/")
    model_path = require_data_fast_path(spec.model.cache_path, label="local Cortex reference model")
    with requests.Session() as session:
        session.trust_env = False
        health = session.get(f"{base_url}/health", timeout=30, allow_redirects=False)
        if health.status_code != 200:
            raise ValueError("local gateway is not healthy")
        status = health.json()
        if not isinstance(status, dict) or status.get("status") != "OK":
            raise ValueError("local gateway is not healthy")
        binding = describe_fixture(spec, model_path)
    return {**binding, "model_name": str(model_path), "distribution": "local-gateway"}


LOCAL_NATIVE_FORWARD_ENV = "ARCTIC_CORRECTNESS_LOCAL_NATIVE_FORWARD"
_NATIVE_ROLES = {"training": "training"}


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
        if request.op != "forward" or role != "training":
            raise ValueError(f"local native {request.op} cannot target a {role} child job")
        url = f"{self.config.base_url.rstrip('/')}/{request.op}?{urlencode({'job_id': child})}"
        body = {key: value for key, value in request.body.items() if value is not None}
        return url, {"data": wire.dumps(body), "headers": {"Content-Type": "application/octet-stream"}}

    def _native_result(self, request, payload):
        result = wire.loads(payload)
        if not isinstance(result, dict):
            raise ValueError("local native gateway response must be a DSSST1 object")
        child = self._native_bindings[str(request.job_id)][1]
        if "job_id" in result and result["job_id"] != child:
            raise ValueError("local native gateway response names a different child job")
        return result

    def call(self, request) -> dict:
        if not getattr(self, "_native_gateway", False) or request.op != "forward":
            return super().call(request)
        url, kwargs = self._native_request(request)
        payload = self._on_loop(self._native_http("POST", url, **kwargs)).result()
        return self._native_result(request, payload)

    async def acall(self, request) -> dict:
        if not getattr(self, "_native_gateway", False) or request.op != "forward":
            return await super().acall(request)
        url, kwargs = self._native_request(request)
        payload = await asyncio.wrap_future(self._on_loop(self._native_http("POST", url, **kwargs)))
        return self._native_result(request, payload)

    def list_checkpoints(self) -> list[dict]:
        if self.job_id is None:
            raise RuntimeError("checkpoint catalog requires an initialized Cortex job")
        return self.call(list_checkpoints_request(self.job_id))

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
