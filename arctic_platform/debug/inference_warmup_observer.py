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

"""Opt-in, rank-local observation of vLLM kernel warmup."""

from __future__ import annotations

import functools
import hashlib
import importlib
import json
import os
import re
import socket
import stat
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

REQUEST_ENV = "ARCTIC_PLATFORM_INFERENCE_WARMUP_OBSERVER"
RAY_DEDUP_ENV = "RAY_DEDUP_LOGS"

_SCHEMA = "inference-warmup-observer-v1"
_CHECK_ID = "inference-checkpoint-loss"
_DESCRIPTOR_LIMIT_BYTES = 32 * 1024
_SOURCE_LIMIT_BYTES = 8 * 1024 * 1024
_RECORD_LIMIT_BYTES = 4 * 1024 * 1024
_SOURCE_MODULES = (
    "vllm.distributed.parallel_state",
    "vllm.model_executor.warmup.kernel_warmup",
    "vllm.v1.worker.gpu_worker",
)
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "warmup observer descriptor contains a duplicate JSON key")
        result[key] = value
    return result


def _descriptor(
    value: Any,
    *,
    expected_config_id: str | None = None,
    expected_world_size: int | None = None,
) -> dict[str, Any]:
    _require(type(value) is dict, "warmup observer descriptor must be an object")
    _require(
        set(value)
        == {
            "schema",
            "request_id",
            "config_id",
            "check_id",
            "world_size",
            "ranks",
            "sources",
        },
        "warmup observer descriptor fields differ",
    )
    _require(value["schema"] == _SCHEMA, "warmup observer descriptor schema differs")
    _require(
        type(value["request_id"]) is str and _IDENTIFIER.fullmatch(value["request_id"]) is not None,
        "warmup observer request_id is invalid",
    )
    _require(
        type(value["config_id"]) is str and _IDENTIFIER.fullmatch(value["config_id"]) is not None,
        "warmup observer config_id is invalid",
    )
    _require(value["check_id"] == _CHECK_ID, "warmup observer check identity differs")
    world_size = value["world_size"]
    _require(type(world_size) is int and 0 < world_size <= 1024, "warmup observer world size is invalid")
    _require(value["ranks"] == list(range(world_size)), "warmup observer ranks must cover the exact world")
    sources = value["sources"]
    _require(
        type(sources) is dict and set(sources) == set(_SOURCE_MODULES),
        "warmup observer source inventory differs",
    )
    _require(
        all(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest) for digest in sources.values()),
        "warmup observer source digest is invalid",
    )
    if expected_config_id is not None:
        _require(value["config_id"] == expected_config_id, "warmup observer config identity differs")
    if expected_world_size is not None:
        _require(world_size == expected_world_size, "warmup observer topology differs")
    return {
        **value,
        "ranks": list(value["ranks"]),
        "sources": dict(value["sources"]),
    }


def _request_from_context(ctx) -> Any | None:
    spec = getattr(ctx, "spec", None)
    settings = getattr(spec, "test_settings", None)
    if not isinstance(settings, dict):
        return None
    check_settings = settings.get(_CHECK_ID)
    if not isinstance(check_settings, dict):
        return None
    return check_settings.get("inference_warmup_observer")


def maybe_bind_inference_warmup_observer(payload, ctx) -> None:
    """Bind an authenticated observer request without changing ordinary sampling."""
    supplied = _request_from_context(ctx)
    if supplied is None:
        return
    inference = payload.get("inference_config")
    _require(type(inference) is dict, "warmup observer requires an inference payload")
    vllm = inference.get("vllm_config")
    _require(type(vllm) is dict, "warmup observer requires vllm configuration")
    tp = vllm.get("tensor_parallel_size")
    pp = vllm.get("pipeline_parallel_size", 1)
    allocation = inference.get("n_gpus")
    _require(
        type(tp) is int
        and type(pp) is int
        and type(allocation) is int
        and tp > 0
        and pp > 0
        and allocation == tp * pp,
        "warmup observer requires one exact TP/PP world",
    )
    config_id = getattr(ctx, "config_id", getattr(getattr(ctx, "cfg", None), "config_id", None))
    request = _descriptor(
        supplied,
        expected_config_id=config_id,
        expected_world_size=allocation,
    )
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":"))
    _require(len(encoded.encode()) <= _DESCRIPTOR_LIMIT_BYTES, "warmup observer descriptor exceeds its bound")
    current = vllm.get("extra_env")
    _require(current is None or type(current) is dict, "warmup observer extra_env must be an object")
    extra_env = dict(current or {})
    _require(REQUEST_ENV not in extra_env, "warmup observer descriptor must not be supplied or overwritten")
    _require(
        extra_env.get(RAY_DEDUP_ENV, "0") == "0",
        "warmup observer requires Ray log deduplication to remain disabled",
    )
    extra_env[REQUEST_ENV] = encoded
    extra_env[RAY_DEDUP_ENV] = "0"
    vllm["extra_env"] = extra_env


def _identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _sha256_fd(fd: int, limit: int) -> str:
    metadata = os.fstat(fd)
    if metadata.st_size > limit:
        raise RuntimeError("warmup observer source exceeds its bound")
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    remaining = limit + 1
    while remaining:
        chunk = os.read(fd, min(1024 * 1024, remaining))
        if not chunk:
            break
        digest.update(chunk)
        remaining -= len(chunk)
    if remaining == 0 and os.read(fd, 1):
        raise RuntimeError("warmup observer source exceeds its bound")
    return digest.hexdigest()


@dataclass
class _HeldFile:
    label: str
    path: Path
    parent_fd: int
    fd: int
    parent_identity: tuple[int, ...]
    file_identity: tuple[int, ...]
    sha256: str

    @classmethod
    def open(cls, label: str, path_value: str, expected_sha256: str | None = None) -> "_HeldFile":
        path = Path(path_value)
        if not path.is_absolute():
            raise RuntimeError(f"warmup observer {label} origin is not absolute")
        try:
            resolved = path.resolve(strict=True)
        except OSError:
            raise RuntimeError(f"warmup observer {label} origin is unavailable") from None
        if resolved != path:
            raise RuntimeError(f"warmup observer {label} origin is not canonical")
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            fd = os.open(
                path.name,
                os.O_RDONLY | os.O_NONBLOCK | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except BaseException:
            os.close(parent_fd)
            raise
        try:
            parent_metadata = os.fstat(parent_fd)
            file_metadata = os.fstat(fd)
            parent_slot = os.stat(path.parent, follow_symlinks=False)
            file_slot = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            if (
                not stat.S_ISDIR(parent_metadata.st_mode)
                or not stat.S_ISREG(file_metadata.st_mode)
                or _identity(parent_slot) != _identity(parent_metadata)
                or _identity(file_slot) != _identity(file_metadata)
            ):
                raise RuntimeError(f"warmup observer {label} requires a stable regular source")
            digest = _sha256_fd(fd, _SOURCE_LIMIT_BYTES)
            if expected_sha256 is not None and digest != expected_sha256:
                raise RuntimeError(f"warmup observer {label} source digest differs")
            held = cls(
                label=label,
                path=path,
                parent_fd=parent_fd,
                fd=fd,
                parent_identity=_identity(parent_metadata),
                file_identity=_identity(file_metadata),
                sha256=digest,
            )
            held.verify(hash_bytes=True)
            return held
        except BaseException:
            try:
                os.close(fd)
            finally:
                os.close(parent_fd)
            raise

    def verify(self, *, hash_bytes: bool) -> None:
        try:
            parent_metadata = os.fstat(self.parent_fd)
            file_metadata = os.fstat(self.fd)
            parent_slot = os.stat(self.path.parent, follow_symlinks=False)
            file_slot = os.stat(self.path.name, dir_fd=self.parent_fd, follow_symlinks=False)
        except OSError:
            raise RuntimeError(f"warmup observer {self.label} source changed") from None
        if (
            _identity(parent_metadata) != self.parent_identity
            or _identity(parent_slot) != self.parent_identity
            or _identity(file_metadata) != self.file_identity
            or _identity(file_slot) != self.file_identity
            or (hash_bytes and _sha256_fd(self.fd, _SOURCE_LIMIT_BYTES) != self.sha256)
        ):
            raise RuntimeError(f"warmup observer {self.label} source changed")

    def record(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "identity": list(self.file_identity),
            "parent_identity": list(self.parent_identity),
            "sha256": self.sha256,
            "size_bytes": self.file_identity[6],
        }

    def close(self) -> None:
        fd, parent_fd = self.fd, self.parent_fd
        self.fd = -1
        self.parent_fd = -1
        errors = []
        for descriptor in (fd, parent_fd):
            if descriptor < 0:
                continue
            try:
                os.close(descriptor)
            except OSError as exc:
                errors.append(exc)
        if errors:
            raise errors[0]


@dataclass
class _Patch:
    owner: Any
    name: str
    original: Any
    replacement: Any


class _Observer:
    def __init__(self, request: dict[str, Any]):
        self.request = request
        self.request_sha256 = hashlib.sha256(
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.sources: list[_HeldFile] = []
        self.caches: list[_HeldFile] = []
        self.cache_records: dict[str, dict[str, Any]] = {}
        self.modules: dict[str, ModuleType] = {}
        self.patches: list[_Patch] = []
        self._sequence = 0
        self._emit_lock = threading.Lock()

    def _load_module(self, module_name: str) -> ModuleType:
        module = importlib.import_module(module_name)
        spec = getattr(module, "__spec__", None)
        origin = getattr(spec, "origin", None)
        if not isinstance(origin, str):
            raise RuntimeError(f"warmup observer {module_name} has no source origin")
        source = _HeldFile.open(module_name, origin, self.request["sources"][module_name])
        self.sources.append(source)
        loader = getattr(spec, "loader", None)
        cached = getattr(module, "__cached__", None)
        cache_record: dict[str, Any] = {
            "loader": f"{type(loader).__module__}.{type(loader).__qualname__}" if loader is not None else None,
            "origin": origin,
            "loaded_bytecode_proven": False,
        }
        if isinstance(cached, str):
            try:
                held_cache = _HeldFile.open(f"{module_name} cache", cached)
            except (OSError, RuntimeError):
                cache_record["cache"] = {"path": cached, "status": "unavailable_or_unsafe"}
            else:
                self.caches.append(held_cache)
                cache_record["cache"] = {"status": "held", **held_cache.record()}
        else:
            cache_record["cache"] = {"status": "not_reported"}
        self.cache_records[module_name] = cache_record
        self.modules[module_name] = module
        return module

    def _verify_sources(self, *, hash_bytes: bool) -> None:
        for held in (*self.sources, *self.caches):
            held.verify(hash_bytes=hash_bytes)

    def _rank_identity(self, *, required: bool) -> dict[str, Any]:
        rank: int | None
        world_size: int | None
        try:
            distributed = importlib.import_module("torch.distributed")
            available = getattr(distributed, "is_available", lambda: True)()
            initialized = available and distributed.is_initialized()
            if not initialized:
                raise RuntimeError("distributed process group is not initialized")
            rank = int(distributed.get_rank())
            world_size = int(distributed.get_world_size())
        except BaseException:
            if required:
                raise RuntimeError("warmup observer requires initialized distributed rank identity") from None
            rank = _optional_env_int("RANK")
            world_size = _optional_env_int("WORLD_SIZE")
        if required and (rank not in self.request["ranks"] or world_size != self.request["world_size"]):
            raise RuntimeError("warmup observer runtime rank identity differs")
        return {
            "rank": rank,
            "local_rank": _optional_env_int("LOCAL_RANK"),
            "world_size": world_size,
        }

    def _source_records(self) -> dict[str, Any]:
        return {
            source.label: {
                **source.record(),
                **self.cache_records[source.label],
            }
            for source in self.sources
        }

    def _write_record(self, record: dict[str, Any]) -> None:
        with self._emit_lock:
            sequence = self._sequence
            self._sequence += 1
            complete = {
                "schema": _SCHEMA,
                "request_id": self.request["request_id"],
                "request_sha256": self.request_sha256,
                "config_id": self.request["config_id"],
                "check_id": self.request["check_id"],
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "sequence": sequence,
                "wall_time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                **self._rank_identity(required=record.pop("_rank_required", True)),
                **record,
            }
            encoded = (json.dumps(complete, sort_keys=True, separators=(",", ":")) + "\n").encode()
            if len(encoded) > _RECORD_LIMIT_BYTES:
                fallback = {key: value for key, value in complete.items() if key != "traceback"}
                fallback.update(
                    phase="observer_error",
                    evidence_error="complete traceback exceeds the warmup observer record bound",
                    omitted_record_bytes=len(encoded),
                )
                encoded = (json.dumps(fallback, sort_keys=True, separators=(",", ":")) + "\n").encode()
                if len(encoded) > _RECORD_LIMIT_BYTES:
                    raise RuntimeError("warmup observer evidence record exceeds its bound")
            offset = 0
            while offset < len(encoded):
                written = os.write(2, encoded[offset:])
                if written <= 0:
                    raise RuntimeError("warmup observer stderr write stopped")
                offset += written

    def _group_record(self, group: Any, src: Any, *, use_cpu_group: bool) -> dict[str, Any]:
        ranks = getattr(group, "ranks", None)
        rank_in_group = getattr(group, "rank_in_group", None)
        if (
            not isinstance(ranks, (list, tuple))
            or not ranks
            or any(type(rank) is not int for rank in ranks)
            or type(rank_in_group) is not int
            or not 0 <= rank_in_group < len(ranks)
            or type(src) is not int
            or not 0 <= src < len(ranks)
        ):
            raise RuntimeError("warmup observer group identity is invalid")
        process_group = getattr(group, "cpu_group" if use_cpu_group else "device_group", None)
        backend = None
        backend_name = getattr(process_group, "_get_backend_name", None)
        if callable(backend_name):
            backend = str(backend_name())
        return {
            "group": {
                "name": _bounded_text(getattr(group, "unique_name", None)),
                "ranks": list(ranks),
                "rank_in_group": rank_in_group,
                "backend": _bounded_text(backend),
                "source_rank_in_group": src,
                "source_global_rank": ranks[src],
            }
        }

    def _terminal_source_error(self) -> BaseException | None:
        try:
            self._verify_sources(hash_bytes=True)
        except BaseException as exc:
            return exc
        return None

    def _invoke(self, operation: str, original, args, kwargs, *, group_record=None):
        details = group_record or {}
        self._verify_sources(hash_bytes=False)
        self._write_record({"operation": operation, "phase": "enter", **details})
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            source_error = self._terminal_source_error()
            record = {
                "operation": operation,
                "phase": "error",
                "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                "traceback": formatted,
                "source_generation_verified": source_error is None,
                **details,
            }
            if source_error is not None:
                record["source_error_type"] = f"{type(source_error).__module__}.{type(source_error).__qualname__}"
                exc.add_note(f"warmup observer source revalidation also failed: {source_error!r}")
            try:
                self._write_record(record)
            except BaseException as evidence_error:
                exc.add_note(f"warmup observer error recording also failed: {evidence_error!r}")
            raise
        source_error = self._terminal_source_error()
        if source_error is not None:
            self._write_record(
                {
                    "operation": operation,
                    "phase": "observer_error",
                    "source_generation_verified": False,
                    "source_error_type": f"{type(source_error).__module__}.{type(source_error).__qualname__}",
                    **details,
                }
            )
            raise source_error
        self._write_record(
            {
                "operation": operation,
                "phase": "exit",
                "source_generation_verified": True,
                **details,
            }
        )
        return result

    def _function_wrapper(self, operation: str, original):
        @functools.wraps(original)
        def observed(*args, **kwargs):
            return self._invoke(operation, original, args, kwargs)

        setattr(observed, "__inference_warmup_observer_request__", self.request_sha256)
        return observed

    def _group_wrapper(self, operation: str, original, *, use_cpu_group: bool):
        @functools.wraps(original)
        def observed(group, *args, **kwargs):
            src = kwargs.get("src", args[1] if len(args) > 1 else 0)
            details = self._group_record(group, src, use_cpu_group=use_cpu_group)
            return self._invoke(operation, original, (group, *args), kwargs, group_record=details)

        setattr(observed, "__inference_warmup_observer_request__", self.request_sha256)
        return observed

    def _patch(self, owner: Any, name: str, replacement: Any) -> None:
        original = vars(owner).get(name)
        if original is None:
            raise RuntimeError(f"warmup observer target is missing: {name}")
        marker = getattr(original, "__inference_warmup_observer_request__", None)
        if marker is not None:
            if marker != self.request_sha256:
                raise RuntimeError("warmup observer target is already bound to another request")
            return
        self.patches.append(_Patch(owner, name, original, replacement))
        setattr(owner, name, replacement)

    def install(self) -> None:
        parallel = self._load_module("vllm.distributed.parallel_state")
        warmup = self._load_module("vllm.model_executor.warmup.kernel_warmup")
        gpu_worker = self._load_module("vllm.v1.worker.gpu_worker")
        kernel = vars(warmup).get("kernel_warmup")
        if kernel is None or vars(gpu_worker).get("kernel_warmup") is not kernel:
            raise RuntimeError("warmup observer requires the direct gpu_worker kernel_warmup alias")
        flashinfer = vars(warmup).get("flashinfer_autotune")
        coordinator = vars(parallel).get("GroupCoordinator")
        if flashinfer is None or not isinstance(coordinator, type):
            raise RuntimeError("warmup observer targets are unavailable")

        kernel_wrapper = self._function_wrapper("kernel_warmup", kernel)
        self._patch(warmup, "kernel_warmup", kernel_wrapper)
        self._patch(gpu_worker, "kernel_warmup", kernel_wrapper)
        self._patch(
            warmup,
            "flashinfer_autotune",
            self._function_wrapper("flashinfer_autotune", flashinfer),
        )
        for name, use_cpu_group in (("broadcast_object", True), ("broadcast_object_list", False)):
            original = vars(coordinator).get(name)
            if original is None:
                raise RuntimeError(f"warmup observer GroupCoordinator.{name} is unavailable")
            self._patch(
                coordinator,
                name,
                self._group_wrapper(name, original, use_cpu_group=use_cpu_group),
            )
        self._verify_sources(hash_bytes=True)
        self._write_record(
            {
                "_rank_required": False,
                "operation": "observer",
                "phase": "installed",
                "source_authentication": self._source_records(),
                "limitation": (
                    "held source and cache representations do not prove already-loaded bytecode; "
                    "all-rank retention still requires complete logs before signals"
                ),
            }
        )

    def close(self) -> None:
        errors = []
        for patch in reversed(self.patches):
            try:
                if vars(patch.owner).get(patch.name) is patch.replacement:
                    setattr(patch.owner, patch.name, patch.original)
            except BaseException as exc:
                errors.append(exc)
        self.patches.clear()
        for held in reversed([*self.caches, *self.sources]):
            try:
                held.close()
            except BaseException as exc:
                errors.append(exc)
        self.caches.clear()
        self.sources.clear()
        if errors:
            raise errors[0]


def _optional_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or re.fullmatch(r"(?:0|[1-9][0-9]*)", value) is None:
        return None
    return int(value)


def _bounded_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    if len(text) > 256:
        raise RuntimeError("warmup observer identity text exceeds its bound")
    return text


_INSTALL_LOCK = threading.Lock()
_ACTIVE: _Observer | None = None


def maybe_install_inference_warmup_observer() -> None:
    """Install worker-local observers only for an exact transported request."""
    encoded = os.environ.get(REQUEST_ENV)
    if encoded is None:
        return
    if len(encoded.encode()) > _DESCRIPTOR_LIMIT_BYTES:
        raise ValueError("warmup observer descriptor exceeds its bound")
    request = _descriptor(json.loads(encoded, object_pairs_hook=_unique_object))
    with _INSTALL_LOCK:
        global _ACTIVE
        if _ACTIVE is not None:
            if _ACTIVE.request != request:
                raise RuntimeError("warmup observer process is already bound to another request")
            return
        observer = _Observer(request)
        try:
            observer.install()
        except BaseException:
            observer.close()
            raise
        _ACTIVE = observer


def _reset_inference_warmup_observer_for_tests() -> None:
    """Restore fake-module patches and descriptors between focused controls."""
    with _INSTALL_LOCK:
        global _ACTIVE
        active, _ACTIVE = _ACTIVE, None
        if active is not None:
            active.close()
