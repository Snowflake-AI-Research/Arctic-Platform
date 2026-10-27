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

"""Opt-in durable records for vLLM startup exceptions."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import secrets
import socket
import stat
import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal
from typing import Mapping

REQUEST_ENV = "ARCTIC_PLATFORM_INFERENCE_STARTUP_ERROR"
ATTEMPT_ENV = "ARCTIC_PLATFORM_INFERENCE_STARTUP_ERROR_ATTEMPT"

_SCHEMA = "inference-startup-error-v1"
_CHECK_ID = "inference-checkpoint-loss"
_DESCRIPTOR_LIMIT_BYTES = 32 * 1024
_SOURCE_LIMIT_BYTES = 8 * 1024 * 1024
_RECORD_LIMIT_BYTES = 4 * 1024 * 1024
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_STARTUP_FAILURES = frozenset({"EngineCore failed to start.", "WorkerProc failed to start."})

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_SOURCE_PATHS = {
    "producer": Path(__file__).resolve(),
    "startup_error": _PACKAGE_ROOT / "inference" / "vllm" / "startup_error.py",
    "plugin": _PACKAGE_ROOT / "inference" / "vllm" / "plugin.py",
    "worker": _PACKAGE_ROOT / "inference" / "server" / "worker.py",
    "replica_pool": _PACKAGE_ROOT / "inference" / "server" / "replica_pool.py",
    "payload": _PACKAGE_ROOT / "correctness" / "checks" / "inference_checkpoint_loss.py",
}

logger = logging.getLogger(__name__)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        _require(
            key not in result,
            "startup error descriptor contains a duplicate JSON key",
        )
        result[key] = value
    return result


def _directory_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_nlink,
        metadata.st_uid,
        metadata.st_gid,
    )


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
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
        raise RuntimeError("startup error source exceeds its bound")
    digest = hashlib.sha256()
    offset = 0
    while offset <= limit:
        chunk = os.pread(fd, min(1024 * 1024, limit + 1 - offset), offset)
        if not chunk:
            break
        digest.update(chunk)
        offset += len(chunk)
    if offset > limit:
        raise RuntimeError("startup error source exceeds its bound")
    return digest.hexdigest()


def _canonical_path(value: str, label: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise RuntimeError(f"startup error {label} path is not absolute")
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        raise RuntimeError(f"startup error {label} path is unavailable") from None
    if resolved != path:
        raise RuntimeError(f"startup error {label} path is not canonical")
    if path.name == "":
        raise RuntimeError(f"startup error {label} path is invalid")
    return path


@dataclass
class _HeldSource:
    label: str
    path: Path
    parent_fd: int
    fd: int
    parent_identity: tuple[int, ...]
    file_identity: tuple[int, ...]
    sha256: str

    @classmethod
    def open(
        cls,
        label: str,
        path_value: str,
        expected_sha256: str,
    ) -> "_HeldSource":
        path = _canonical_path(path_value, label)
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            fd = os.open(
                path.name,
                os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
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
                or _directory_identity(parent_slot) != _directory_identity(parent_metadata)
                or _file_identity(file_slot) != _file_identity(file_metadata)
            ):
                raise RuntimeError(f"startup error {label} requires a stable regular source")
            digest = _sha256_fd(fd, _SOURCE_LIMIT_BYTES)
            if digest != expected_sha256:
                raise RuntimeError(f"startup error {label} source digest differs")
            held = cls(
                label=label,
                path=path,
                parent_fd=parent_fd,
                fd=fd,
                parent_identity=_directory_identity(parent_metadata),
                file_identity=_file_identity(file_metadata),
                sha256=digest,
            )
            held.verify()
            return held
        except BaseException:
            try:
                os.close(fd)
            finally:
                os.close(parent_fd)
            raise

    def verify(self) -> None:
        try:
            parent_metadata = os.fstat(self.parent_fd)
            file_metadata = os.fstat(self.fd)
            parent_slot = os.stat(self.path.parent, follow_symlinks=False)
            file_slot = os.stat(
                self.path.name,
                dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
        except OSError:
            raise RuntimeError(f"startup error {self.label} source changed") from None
        if (
            _directory_identity(parent_metadata) != self.parent_identity
            or _directory_identity(parent_slot) != self.parent_identity
            or _file_identity(file_metadata) != self.file_identity
            or _file_identity(file_slot) != self.file_identity
            or _sha256_fd(self.fd, _SOURCE_LIMIT_BYTES) != self.sha256
        ):
            raise RuntimeError(f"startup error {self.label} source changed")

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
class _HeldRoot:
    path: Path
    parent_fd: int
    fd: int
    parent_identity: tuple[int, ...]
    root_identity: tuple[int, ...]

    @classmethod
    def open(cls, root: Mapping[str, Any]) -> "_HeldRoot":
        path = _canonical_path(root["path"], "capture root")
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
        )
        try:
            fd = os.open(
                path.name,
                os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW,
                dir_fd=parent_fd,
            )
        except BaseException:
            os.close(parent_fd)
            raise
        try:
            parent_metadata = os.fstat(parent_fd)
            root_metadata = os.fstat(fd)
            parent_slot = os.stat(path.parent, follow_symlinks=False)
            root_slot = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            parent_identity = _directory_identity(parent_metadata)
            root_identity = _directory_identity(root_metadata)
            if (
                not stat.S_ISDIR(parent_metadata.st_mode)
                or not stat.S_ISDIR(root_metadata.st_mode)
                or _directory_identity(parent_slot) != parent_identity
                or _directory_identity(root_slot) != root_identity
                or list(parent_identity) != root["parent_identity"]
                or list(root_identity) != root["identity"]
                or root_metadata.st_uid != os.getuid()
                or stat.S_IMODE(root_metadata.st_mode) != 0o700
            ):
                raise RuntimeError("startup error capture root identity or ownership differs")
            held = cls(
                path=path,
                parent_fd=parent_fd,
                fd=fd,
                parent_identity=parent_identity,
                root_identity=root_identity,
            )
            held.verify()
            return held
        except BaseException:
            try:
                os.close(fd)
            finally:
                os.close(parent_fd)
            raise

    def verify(self) -> None:
        try:
            parent_metadata = os.fstat(self.parent_fd)
            root_metadata = os.fstat(self.fd)
            parent_slot = os.stat(self.path.parent, follow_symlinks=False)
            root_slot = os.stat(
                self.path.name,
                dir_fd=self.parent_fd,
                follow_symlinks=False,
            )
        except OSError:
            raise RuntimeError("startup error capture root changed") from None
        if (
            _directory_identity(parent_metadata) != self.parent_identity
            or _directory_identity(parent_slot) != self.parent_identity
            or _directory_identity(root_metadata) != self.root_identity
            or _directory_identity(root_slot) != self.root_identity
        ):
            raise RuntimeError("startup error capture root changed")

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


def _descriptor(
    value: Any,
    *,
    expected_config_id: str | None = None,
) -> dict[str, Any]:
    _require(type(value) is dict, "startup error descriptor must be an object")
    _require(
        set(value)
        == {
            "schema",
            "request_id",
            "config_id",
            "check_id",
            "generation",
            "root",
            "sources",
        },
        "startup error descriptor fields differ",
    )
    _require(value["schema"] == _SCHEMA, "startup error descriptor schema differs")
    for field in ("request_id", "config_id", "generation"):
        _require(
            type(value[field]) is str and _IDENTIFIER.fullmatch(value[field]) is not None,
            f"startup error {field} is invalid",
        )
    _require(
        value["check_id"] == _CHECK_ID,
        "startup error check identity differs",
    )
    if expected_config_id is not None:
        _require(
            value["config_id"] == expected_config_id,
            "startup error config identity differs",
        )
    root = value["root"]
    _require(
        type(root) is dict
        and set(root) == {"path", "identity", "parent_identity"}
        and type(root["path"]) is str
        and type(root["identity"]) is list
        and type(root["parent_identity"]) is list
        and len(root["identity"]) == 6
        and len(root["parent_identity"]) == 6
        and all(type(item) is int for item in root["identity"])
        and all(type(item) is int for item in root["parent_identity"]),
        "startup error root descriptor is invalid",
    )
    sources = value["sources"]
    _require(
        type(sources) is dict and set(sources) == set(_SOURCE_PATHS),
        "startup error source inventory differs",
    )
    _require(
        all(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest) is not None for digest in sources.values()),
        "startup error source digest is invalid",
    )
    return {
        **value,
        "root": {
            "path": root["path"],
            "identity": list(root["identity"]),
            "parent_identity": list(root["parent_identity"]),
        },
        "sources": dict(sources),
    }


def _request_from_context(ctx) -> Any | None:
    spec = getattr(ctx, "spec", None)
    settings = getattr(spec, "test_settings", None)
    if not isinstance(settings, dict):
        return None
    check_settings = settings.get(_CHECK_ID)
    if not isinstance(check_settings, dict):
        return None
    return check_settings.get("inference_startup_error")


def maybe_bind_inference_startup_error(payload, ctx) -> None:
    """Bind the request before ReplicaPool creates its Ray coordinator."""
    supplied = _request_from_context(ctx)
    if supplied is None:
        return
    config_id = getattr(
        ctx,
        "config_id",
        getattr(getattr(ctx, "cfg", None), "config_id", None),
    )
    request = _descriptor(supplied, expected_config_id=config_id)
    encoded = json.dumps(request, sort_keys=True, separators=(",", ":"))
    _require(
        len(encoded.encode()) <= _DESCRIPTOR_LIMIT_BYTES,
        "startup error descriptor exceeds its bound",
    )
    inference = payload.get("inference_config")
    _require(type(inference) is dict, "startup error requires an inference payload")
    vllm = inference.get("vllm_config")
    _require(type(vllm) is dict, "startup error requires vllm configuration")
    current = vllm.get("extra_env")
    _require(
        current is None or type(current) is dict,
        "startup error extra_env must be an object",
    )
    extra_env = dict(current or {})
    _require(
        REQUEST_ENV not in extra_env and ATTEMPT_ENV not in extra_env,
        "startup error environment must not be supplied or overwritten",
    )
    probe = _Producer(request)
    probe.close()
    extra_env[REQUEST_ENV] = encoded
    vllm["extra_env"] = extra_env


def _read_proc(path: str, limit: int) -> str:
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        chunks = []
        size = 0
        while size <= limit:
            chunk = os.read(fd, min(4096, limit + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        if size > limit:
            raise RuntimeError("startup error process identity exceeds its bound")
        return b"".join(chunks).decode("ascii").strip()
    finally:
        os.close(fd)


def _process_identity() -> tuple[str, int]:
    boot_id = _read_proc("/proc/sys/kernel/random/boot_id", 128)
    if re.fullmatch(r"[0-9a-f-]{36}", boot_id) is None:
        raise RuntimeError("startup error boot identity is invalid")
    process_stat = _read_proc("/proc/self/stat", 16 * 1024)
    end = process_stat.rfind(")")
    fields = process_stat[end + 2 :].split() if end >= 0 else []
    if len(fields) <= 19 or not fields[19].isdigit():
        raise RuntimeError("startup error process start ticks are unavailable")
    return boot_id, int(fields[19])


def _optional_env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or re.fullmatch(r"(?:0|[1-9][0-9]*)", value) is None:
        return None
    return int(value)


def _rank_identity() -> dict[str, Any]:
    try:
        import torch.distributed as distributed

        if not distributed.is_available() or not distributed.is_initialized():
            raise RuntimeError
        rank = int(distributed.get_rank())
        world_size = int(distributed.get_world_size())
    except BaseException:
        return {
            "rank_state": "unknown",
            "global_rank": None,
            "local_rank": None,
            "world_size": None,
        }
    return {
        "rank_state": "initialized",
        "global_rank": rank,
        "local_rank": _optional_env_int("LOCAL_RANK"),
        "world_size": world_size,
    }


def _attempt_identity(request_sha256: str) -> dict[str, Any]:
    encoded = os.environ.get(ATTEMPT_ENV)
    if encoded is None:
        return {
            "attempt_state": "unknown",
            "attempt_id": None,
            "attempt_ordinal": None,
        }
    try:
        value = json.loads(encoded, object_pairs_hook=_unique_object)
        _require(
            type(value) is dict
            and set(value)
            == {
                "request_sha256",
                "attempt_id",
                "attempt_ordinal",
                "coordinator_pid",
                "coordinator_start_ticks",
            },
            "startup error attempt descriptor fields differ",
        )
        _require(
            value["request_sha256"] == request_sha256,
            "startup error attempt request differs",
        )
        _require(
            type(value["attempt_id"]) is str and re.fullmatch(r"[0-9a-f]{32}", value["attempt_id"]) is not None,
            "startup error attempt id is invalid",
        )
        _require(
            type(value["attempt_ordinal"]) is int and value["attempt_ordinal"] > 0,
            "startup error attempt ordinal is invalid",
        )
        _require(
            type(value["coordinator_pid"]) is int
            and value["coordinator_pid"] > 0
            and type(value["coordinator_start_ticks"]) is int
            and value["coordinator_start_ticks"] > 0,
            "startup error attempt coordinator identity is invalid",
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return {
            "attempt_state": "invalid",
            "attempt_id": None,
            "attempt_ordinal": None,
        }
    return {
        "attempt_state": "bound",
        "attempt_id": value["attempt_id"],
        "attempt_ordinal": value["attempt_ordinal"],
    }


class _Producer:
    def __init__(self, request: dict[str, Any]) -> None:
        self.request = request
        self.request_sha256 = hashlib.sha256(
            json.dumps(request, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.root = _HeldRoot.open(request["root"])
        self.sources = []
        try:
            for label, path in _SOURCE_PATHS.items():
                self.sources.append(
                    _HeldSource.open(
                        label,
                        str(path),
                        request["sources"][label],
                    )
                )
            self.pid = os.getpid()
            self.boot_id, self.start_ticks = _process_identity()
            self.hostname = socket.gethostname()
            self.host_id = hashlib.sha256(self.hostname.encode()).hexdigest()[:16]
            self._sequence = 0
            self._write_lock = threading.Lock()
        except BaseException:
            self.close()
            raise

    def _verify(self) -> None:
        if os.getpid() != self.pid:
            raise RuntimeError("startup error producer crossed a process boundary")
        self.root.verify()
        for source in self.sources:
            source.verify()

    def write(self, phase: str, details: Mapping[str, Any]) -> None:
        with self._write_lock:
            self._verify()
            sequence = self._sequence
            self._sequence += 1
            record = {
                "schema": _SCHEMA,
                "request_id": self.request["request_id"],
                "request_sha256": self.request_sha256,
                "config_id": self.request["config_id"],
                "check_id": self.request["check_id"],
                "generation": self.request["generation"],
                "source_sha256": self.request["sources"],
                "host": self.hostname,
                "boot_id": self.boot_id,
                "pid": self.pid,
                "start_ticks": self.start_ticks,
                "sequence": sequence,
                "wall_time_ns": time.time_ns(),
                "monotonic_ns": time.monotonic_ns(),
                **_attempt_identity(self.request_sha256),
                **_rank_identity(),
                "phase": phase,
                **details,
            }
            encoded = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
            if len(encoded) > _RECORD_LIMIT_BYTES:
                raise RuntimeError("startup error complete traceback exceeds its record bound")
            name = f"{self.host_id}-{self.pid}-{self.start_ticks}-{sequence:08d}.json"
            fd = os.open(
                name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
                0o600,
                dir_fd=self.root.fd,
            )
            try:
                offset = 0
                while offset < len(encoded):
                    written = os.write(fd, encoded[offset:])
                    if written <= 0:
                        raise RuntimeError("startup error record write stopped")
                    offset += written
                os.fsync(fd)
                metadata = os.fstat(fd)
                slot = os.stat(name, dir_fd=self.root.fd, follow_symlinks=False)
                if (
                    not stat.S_ISREG(metadata.st_mode)
                    or metadata.st_uid != os.getuid()
                    or metadata.st_nlink != 1
                    or stat.S_IMODE(metadata.st_mode) != 0o600
                    or metadata.st_size != len(encoded)
                    or _file_identity(slot) != _file_identity(metadata)
                    or os.pread(fd, len(encoded) + 1, 0) != encoded
                ):
                    raise RuntimeError("startup error record identity differs")
                self._verify()
            finally:
                os.close(fd)

    def close(self) -> None:
        errors = []
        sources, self.sources = getattr(self, "sources", []), []
        for source in reversed(sources):
            try:
                source.close()
            except OSError as exc:
                errors.append(exc)
        root, self.root = getattr(self, "root", None), None
        if root is not None:
            try:
                root.close()
            except OSError as exc:
                errors.append(exc)
        if errors:
            raise errors[0]


class DurableStartupErrorHandler(logging.Handler):
    """Write complete qualifying startup exceptions without replacing records."""

    def __init__(self) -> None:
        super().__init__(level=logging.ERROR)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            if not isinstance(record.msg, str) or record.msg not in _STARTUP_FAILURES or record.exc_info is None:
                return
            exc = record.exc_info[1]
            if exc is None:
                return
            producer = _active_producer()
            formatted = "".join(traceback.format_exception(*record.exc_info))
            producer.write(
                "startup_error",
                {
                    "process_name": (
                        getattr(
                            record,
                            "vllm_process_name",
                            None,
                        )
                        or record.processName
                    ),
                    "logger_name": record.name,
                    "log_message": record.msg,
                    "exception_type": f"{type(exc).__module__}.{type(exc).__qualname__}",
                    "exception_message": str(exc),
                    "traceback": formatted,
                    "traceback_complete": True,
                },
            )
        except BaseException:
            self.handleError(record)


_INSTALL_LOCK = threading.Lock()
_ACTIVE: _Producer | None = None


def _encoded_request() -> dict[str, Any] | None:
    encoded = os.environ.get(REQUEST_ENV)
    if encoded is None:
        return None
    if len(encoded.encode()) > _DESCRIPTOR_LIMIT_BYTES:
        raise ValueError("startup error descriptor exceeds its bound")
    return _descriptor(json.loads(encoded, object_pairs_hook=_unique_object))


def _active_producer() -> _Producer:
    request = _encoded_request()
    if request is None:
        raise RuntimeError("startup error producer request is unavailable")
    global _ACTIVE
    if _ACTIVE is None:
        _ACTIVE = _Producer(request)
    elif _ACTIVE.request != request:
        raise RuntimeError("startup error process is already bound to another request")
    return _ACTIVE


def maybe_install_inference_startup_error_producer() -> None:
    """Install only when a request reached this process before plugin loading."""
    if os.environ.get(REQUEST_ENV) is None:
        return
    with _INSTALL_LOCK:
        _active_producer()
        vllm_logger = logging.getLogger("vllm")
        if not any(isinstance(handler, DurableStartupErrorHandler) for handler in vllm_logger.handlers):
            vllm_logger.addHandler(DurableStartupErrorHandler())


@dataclass(frozen=True)
class StartupAttempt:
    producer: _Producer
    previous_attempt: str | None
    attempt_id: str
    attempt_ordinal: int


def maybe_begin_inference_startup_attempt(
    attempt_ordinal: int,
) -> StartupAttempt | None:
    """Bind one process-local retry generation before vLLM creates children."""
    if os.environ.get(REQUEST_ENV) is None:
        return None
    _require(
        type(attempt_ordinal) is int and attempt_ordinal > 0,
        "startup error attempt ordinal is invalid",
    )
    with _INSTALL_LOCK:
        producer = _active_producer()
    previous = os.environ.get(ATTEMPT_ENV)
    if previous is not None:
        raise RuntimeError("startup error attempt environment is already bound")
    attempt_id = secrets.token_hex(16)
    encoded = json.dumps(
        {
            "request_sha256": producer.request_sha256,
            "attempt_id": attempt_id,
            "attempt_ordinal": attempt_ordinal,
            "coordinator_pid": producer.pid,
            "coordinator_start_ticks": producer.start_ticks,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    os.environ[ATTEMPT_ENV] = encoded
    try:
        producer.write(
            "attempt_begin",
            {"coordinator": True},
        )
    except BaseException:
        os.environ.pop(ATTEMPT_ENV, None)
        raise
    return StartupAttempt(
        producer=producer,
        previous_attempt=previous,
        attempt_id=attempt_id,
        attempt_ordinal=attempt_ordinal,
    )


def maybe_finish_inference_startup_attempt(
    attempt: StartupAttempt | None,
    outcome: Literal["ready", "retry", "error"],
    error: BaseException | None,
) -> None:
    """Finish diagnostics without replacing the engine's result or exception."""
    if attempt is None:
        return
    diagnostic_error = None
    try:
        details: dict[str, Any] = {
            "coordinator": True,
            "outcome": outcome,
        }
        if error is not None:
            details.update(
                exception_type=f"{type(error).__module__}.{type(error).__qualname__}",
                exception_message=str(error),
                traceback="".join(
                    traceback.format_exception(
                        type(error),
                        error,
                        error.__traceback__,
                    )
                ),
                traceback_complete=True,
            )
        attempt.producer.write("attempt_end", details)
    except BaseException as exc:
        diagnostic_error = exc
    finally:
        if attempt.previous_attempt is None:
            os.environ.pop(ATTEMPT_ENV, None)
        else:
            os.environ[ATTEMPT_ENV] = attempt.previous_attempt
    if diagnostic_error is not None:
        if error is not None:
            try:
                error.add_note(f"durable startup error recording also failed: {diagnostic_error!r}")
            except BaseException:
                pass
        else:
            logger.error(
                "Durable startup error attempt finalization failed",
                exc_info=(
                    type(diagnostic_error),
                    diagnostic_error,
                    diagnostic_error.__traceback__,
                ),
            )


def _after_fork() -> None:
    global _ACTIVE, _INSTALL_LOCK
    active, _ACTIVE = _ACTIVE, None
    _INSTALL_LOCK = threading.Lock()
    if active is not None:
        try:
            active.close()
        except BaseException:
            pass


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)


def _reset_inference_startup_error_for_tests() -> None:
    """Close held descriptors and remove handlers between focused controls."""
    global _ACTIVE
    with _INSTALL_LOCK:
        active, _ACTIVE = _ACTIVE, None
        if active is not None:
            active.close()
        vllm_logger = logging.getLogger("vllm")
        for handler in vllm_logger.handlers[:]:
            if isinstance(handler, DurableStartupErrorHandler):
                vllm_logger.removeHandler(handler)
