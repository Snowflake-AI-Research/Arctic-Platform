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

"""Allocation-bound retained native checkpoint namespaces on FAST storage."""

from __future__ import annotations

import base64
import contextlib
import fcntl
import hashlib
import json
import os
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import traceback
import uuid
from pathlib import Path

from .storage import _ds_ssh
from .storage import parse_hostfile

_FAST_ROOT = Path("/data-fast")
_MARKER = ".correctness-checkpoint-owner.json"
_RECEIPT = "NATIVE_CHECKPOINT_STORE "


class CheckpointStoreError(RuntimeError):
    def __init__(
        self,
        message: str,
        receipts: dict | None = None,
        *,
        transport_stdout: str | bytes | None = None,
        transport_stderr: str | bytes | None = None,
    ) -> None:
        super().__init__(message)
        self.receipts = receipts or {}
        self.transport_stdout = (
            transport_stdout.decode(errors="replace")
            if isinstance(transport_stdout, bytes)
            else transport_stdout or ""
        )
        self.transport_stderr = (
            transport_stderr.decode(errors="replace")
            if isinstance(transport_stderr, bytes)
            else transport_stderr or ""
        )
        for name, output in (("stdout", self.transport_stdout), ("stderr", self.transport_stderr)):
            if output:
                self.add_note(f"Checkpoint launcher {name}:\n{output}")


def _identity(info: os.stat_result) -> list[int]:
    return [info.st_dev, info.st_ino, info.st_mode, info.st_uid]


def _canonical_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.is_absolute() or root.resolve() != root or not root.is_relative_to(_FAST_ROOT):
        raise ValueError("checkpoint root must be a canonical FAST descendant")
    if root == _FAST_ROOT:
        raise ValueError("checkpoint root cannot be the FAST root")
    return root


def _native_path(source_job_id: str) -> Path:
    from dss.ray_dss.jobs.checkpointing import _job_ckpt_dir

    return Path(_job_ckpt_dir(source_job_id))


def _marker(binding: dict, request: dict) -> dict:
    return {
        "schema": "native-checkpoint-store-v1",
        "store_id": request["store_id"],
        "allocation_sha256": request["allocation_sha256"],
        **binding,
    }


@contextlib.contextmanager
def _native_lock(native: Path, *, exclusive: bool):
    native.parent.mkdir(parents=True, exist_ok=True)
    if native.parent.resolve() != native.parent:
        raise ValueError("native checkpoint parent is not canonical")
    lock = Path(f"{native}.lock")
    descriptor = os.open(lock, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_uid != os.geteuid()
            or info.st_nlink != 1
            or _identity(lock.lstat()) != _identity(info)
        ):
            raise ValueError("native checkpoint lock ownership changed")
        fcntl.flock(descriptor, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB)
        try:
            yield _identity(info)
            if _identity(lock.lstat()) != _identity(info):
                raise ValueError("native checkpoint lock inode changed")
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _root_identity(root: Path, binding: dict, request: dict) -> dict:
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid():
        raise ValueError("checkpoint root must be an owned real directory")
    marker = root / _MARKER
    descriptor = os.open(marker, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        marker_info = os.fstat(descriptor)
        if (
            not stat.S_ISREG(marker_info.st_mode)
            or marker_info.st_uid != os.geteuid()
            or marker_info.st_nlink != 1
            or marker_info.st_size > 8192
            or _identity(marker.lstat()) != _identity(marker_info)
        ):
            raise ValueError("checkpoint ownership marker is not a bounded owned file")
        with os.fdopen(descriptor, "r", closefd=False) as stream:
            if json.load(stream) != _marker(binding, request):
                raise ValueError("checkpoint ownership marker differs")
    finally:
        os.close(descriptor)
    return {"root": _identity(info), "marker": _identity(marker_info)}


def _no_payload_links(root: Path) -> None:
    def failed(error):
        raise error

    for directory, directories, files in os.walk(root, followlinks=False, onerror=failed):
        for name in directories + files:
            info = (Path(directory) / name).lstat()
            if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
                raise ValueError("checkpoint payload contains a symlink or special file")


def _verified_identity(root: Path, native: Path, binding: dict, request: dict, expected: dict, lock: list) -> dict:
    actual = _root_identity(root, binding, request)
    link = None
    if os.path.lexists(native):
        info = native.lstat()
        if not stat.S_ISLNK(info.st_mode) or info.st_uid != os.geteuid() or os.readlink(native) != str(root):
            raise ValueError("native checkpoint link ownership or literal target differs")
        link = _identity(info)
    actual["link"] = link
    actual["lock"] = lock
    if actual != expected:
        raise ValueError("checkpoint root, marker or native link identity changed")
    _no_payload_links(root)
    return actual


def _node_receipt(request: dict) -> dict:
    receipt = {
        "host": socket.gethostname(),
        "token": request["token"],
        "operation": request["operation"],
        "binding": dict(request["binding"]),
        "ok": False,
        "identity": None,
        "error": None,
    }
    try:
        binding = receipt["binding"]
        source = binding["source_job_id"]
        if not isinstance(source, str) or len(source) != 44 or not source.startswith("correctness-"):
            raise ValueError("unique correctness checkpoint source required")
        uuid.UUID(hex=source.removeprefix("correctness-"))
        root = _canonical_root(binding["path"])
        native = _native_path(source)
        if not native.is_absolute() or native.parent.resolve() != native.parent:
            raise ValueError("canonical native checkpoint namespace required")
        if binding.get("native_path", str(native)) != str(native):
            raise ValueError("native checkpoint path differs from the native resolver")
        binding["native_path"] = str(native)
        operation = request["operation"]
        if operation not in {"create", "verify", "close"}:
            raise ValueError("unknown checkpoint store operation")
        with _native_lock(native, exclusive=operation != "verify") as lock:
            if operation == "create":
                if os.path.lexists(native):
                    raise FileExistsError(f"native checkpoint namespace already exists: {native}")
                if root.exists():
                    info = root.lstat()
                    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or any(root.iterdir()):
                        raise ValueError("checkpoint root must be an empty owned real directory")
                else:
                    root.mkdir(mode=0o700)
                with (root / _MARKER).open("x", encoding="utf-8") as stream:
                    json.dump(_marker(binding, request), stream, sort_keys=True)
                receipt["identity"] = {**_root_identity(root, binding, request), "link": None, "lock": lock}
                os.symlink(root, native)
                receipt["identity"]["link"] = _identity(native.lstat())
            else:
                expected = request["expected"].get(receipt["host"])
                if expected is None:
                    raise ValueError("no acknowledged ownership identity for this node")
                if expected.get("closed"):
                    if operation == "verify" or os.path.lexists(root) or os.path.lexists(native):
                        raise ValueError("closed checkpoint namespace was replaced")
                    if expected["lock"] != lock:
                        raise ValueError("native checkpoint lock inode changed")
                    receipt["identity"] = expected
                else:
                    receipt["identity"] = _verified_identity(root, native, binding, request, expected, lock)
                    if operation == "verify" and receipt["identity"]["link"] is None:
                        raise ValueError("checkpoint native link is absent")
                    if operation == "close":
                        if not shutil.rmtree.avoids_symlink_attacks:
                            raise RuntimeError("checkpoint cleanup requires symlink-safe tree removal")
                        if receipt["identity"]["link"] is not None:
                            native.unlink()
                            receipt["identity"]["link"] = None
                        shutil.rmtree(root)
                        receipt["identity"] = {"closed": True, "lock": lock}
            receipt["ok"] = True
    except BaseException:
        receipt["error"] = traceback.format_exc()
    return receipt


class NativeCheckpointStore:
    """Keep source namespaces until all native consumers have quiesced."""

    def __init__(self, hostfile: Path, *, timeout: float = 120) -> None:
        self.hostfile = Path(hostfile).absolute()
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
            raise ValueError("checkpoint fanout timeout must be finite and positive")
        self.timeout = timeout
        self.store_id = uuid.uuid4().hex
        self._host_identity = self._hostfile_identity()
        entries = parse_hostfile(self.hostfile)
        self.hosts = {entry.host for entry in entries}
        if not self.hosts or len(self.hosts) != len(entries):
            raise ValueError("checkpoint store requires exact unique allocation nodes")
        self._check_hostfile()
        self._records: dict[str, tuple[dict, dict]] = {}
        self._closed = False

    def _hostfile_identity(self) -> dict:
        info = self.hostfile.lstat()
        target = self.hostfile.resolve(strict=True)
        actual = target.stat()
        if not stat.S_ISREG(actual.st_mode) or not (stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode)):
            raise ValueError("allocation hostfile must name a regular file")
        content = target.read_bytes()

        def identity(value):
            return [*_identity(value), value.st_size, value.st_mtime_ns, value.st_ctime_ns]

        if identity(info) != identity(self.hostfile.lstat()) or identity(actual) != identity(target.stat()):
            raise ValueError("allocation hostfile changed while reading")
        return {
            "link": os.readlink(self.hostfile) if stat.S_ISLNK(info.st_mode) else None,
            "target": str(target),
            "path_identity": identity(info),
            "target_identity": identity(actual),
            "sha256": hashlib.sha256(content).hexdigest(),
        }

    def _check_hostfile(self) -> None:
        observed = self._hostfile_identity()
        if observed != self._host_identity:
            changes = {
                field: {"expected": expected, "observed": observed[field]}
                for field, expected in self._host_identity.items()
                if observed[field] != expected
            }
            raise CheckpointStoreError(
                f"allocation hostfile changed: path={self.hostfile}; differences={json.dumps(changes, sort_keys=True)}"
            )

    def _fanout(self, operation: str, binding: dict, identities: dict) -> dict:
        self._check_hostfile()
        request = {
            "operation": operation,
            "binding": binding,
            "store_id": self.store_id,
            "allocation_sha256": self._host_identity["sha256"],
            "token": uuid.uuid4().hex,
            "expected": {receipt["host"]: receipt["identity"] for receipt in identities.values()},
        }
        results = {}
        done = None
        try:
            source = Path(__file__).resolve(strict=True)
            source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
            # SSH login cwd/environment need not retain the gateway's private overlay.
            import_roots = list(
                dict.fromkeys(
                    [
                        str(source.parents[3]),
                        *(str(Path(root).resolve()) for root in sys.path if isinstance(root, str)),
                    ]
                )
            )
            script = (
                "import hashlib, json, sys\n"
                "from pathlib import Path\n"
                f"sys.path[:] = {import_roots!r}\n"
                f"source = Path({str(source)!r})\n"
                f"expected_sha = {source_sha!r}\n"
                "if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha:\n"
                "    raise RuntimeError('checkpoint store source SHA256 differs')\n"
                "from arctic_platform.correctness.harness import checkpoint_store as module\n"
                "if Path(module.__file__).resolve() != source:\n"
                "    raise RuntimeError('checkpoint store imported source path differs')\n"
                "if hashlib.sha256(source.read_bytes()).hexdigest() != expected_sha:\n"
                "    raise RuntimeError('checkpoint store source SHA256 differs after import')\n"
                f"receipt = module._node_receipt(json.loads({json.dumps(request)!r}))\n"
                f"print({_RECEIPT!r} + json.dumps(receipt), flush=True)\n"
            )
            # ds_ssh forwards unquoted arguments to pdsh, flattening multiline source.
            encoded = base64.b64encode(script.encode()).decode()
            done = subprocess.run(
                [_ds_ssh(), "-f", str(self.hostfile), f"echo {encoded} | base64 -d | {shlex.quote(sys.executable)} -"],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            for line in done.stdout.splitlines():
                prefix, separator, raw = line.partition(_RECEIPT)
                if not separator:
                    continue
                host = prefix.strip().removesuffix(":").strip()
                if host not in self.hosts or host in results:
                    raise ValueError(f"unexpected or duplicate checkpoint receipt from {host!r}")
                receipt = json.loads(raw)
                if (
                    not isinstance(receipt, dict)
                    or receipt.get("token") != request["token"]
                    or receipt.get("operation") != operation
                    or not isinstance(receipt.get("host"), str)
                    or type(receipt.get("ok")) is not bool
                    or not isinstance(receipt.get("binding"), dict)
                    or any(receipt["binding"].get(key) != value for key, value in binding.items())
                    or not receipt.get("host")
                ):
                    raise ValueError(f"invalid checkpoint receipt from {host}")
                if receipt["ok"]:
                    identity = receipt.get("identity")
                    if not isinstance(identity, dict) or not isinstance(receipt["binding"].get("native_path"), str):
                        raise ValueError(f"missing checkpoint ownership identity from {host}")
                    fields = ("lock",) if identity.get("closed") is True else ("root", "marker", "link", "lock")
                    if operation == "close" and identity.get("closed") is not True:
                        raise ValueError(f"checkpoint cleanup incomplete on {host}")
                    if any(
                        not isinstance(identity.get(field), list)
                        or len(identity[field]) != 4
                        or any(type(value) is not int for value in identity[field])
                        for field in fields
                    ):
                        raise ValueError(f"invalid checkpoint ownership identity from {host}")
                if host in identities and receipt["host"] != identities[host]["host"]:
                    raise ValueError(f"checkpoint node identity changed for {host}")
                if receipt["host"] in {value["host"] for value in results.values()}:
                    raise ValueError("allocation receipts describe duplicate executing nodes")
                results[host] = receipt
            self._check_hostfile()
            if done.returncode or results.keys() != self.hosts or any(not value["ok"] for value in results.values()):
                errors = {host: value.get("error") for host, value in results.items() if not value["ok"]}
                missing = sorted(self.hosts - results.keys())
                raise ValueError(
                    f"checkpoint {operation} failed: rc={done.returncode}, missing hosts={missing}, receipts={errors}"
                )
            return results
        except Exception as exc:
            output = exc if isinstance(exc, subprocess.TimeoutExpired) else done
            raise CheckpointStoreError(
                f"allocation checkpoint {operation}: {exc}",
                results,
                transport_stdout=getattr(output, "stdout", None),
                transport_stderr=getattr(output, "stderr", None),
            ) from exc

    def create(self, path: Path) -> dict:
        if self._closed:
            raise CheckpointStoreError("checkpoint store is closed")
        binding = {"source_job_id": f"correctness-{uuid.uuid4().hex}", "path": str(_canonical_root(path))}
        receipts = {}
        try:
            receipts = self._fanout("create", binding, {})
            native_paths = {value["binding"].get("native_path") for value in receipts.values()}
            if len(native_paths) != 1 or not all(isinstance(value, str) for value in native_paths):
                raise CheckpointStoreError("allocation native checkpoint paths disagree", receipts)
            binding["native_path"] = native_paths.pop()
            return dict(binding)
        except CheckpointStoreError as exc:
            receipts = exc.receipts
            if receipts:
                binding = dict(next(iter(receipts.values()))["binding"])
            raise
        finally:
            self._records[binding["source_job_id"]] = (binding, receipts)

    def verify(self, binding: dict) -> None:
        record = self._records.get(binding.get("source_job_id"))
        if record is None or record[0] != binding:
            raise CheckpointStoreError("checkpoint binding is not owned by this store")
        self._fanout("verify", record[0], record[1])

    @property
    def bindings(self) -> tuple[dict, ...]:
        return tuple(
            {key: binding.get(key) for key in ("source_job_id", "path", "native_path")}
            for binding, _ in self._records.values()
        )

    def close(self) -> None:
        failures = []
        for source, (binding, receipts) in list(self._records.items()):
            try:
                self._fanout("close", binding, receipts)
            except CheckpointStoreError as exc:
                updated = dict(receipts)
                for host, receipt in exc.receipts.items():
                    if receipt.get("identity") is not None:
                        updated[host] = receipt
                self._records[source] = (binding, updated)
                failures.append(exc)
            else:
                del self._records[source]
        if failures:
            raise CheckpointStoreError("; ".join(str(error) for error in failures)) from failures[0]
        self._closed = True
