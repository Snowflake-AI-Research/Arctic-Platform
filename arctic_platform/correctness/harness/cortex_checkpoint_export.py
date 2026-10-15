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

"""Bounded, correctness-local downloads of public resumable checkpoint exports."""

import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import time
from pathlib import Path
from urllib.parse import urljoin
from urllib.parse import urlsplit

import requests

from .workdir import require_data_fast_path

_CHUNK_BYTES = 1024 * 1024
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}
_RECEIPT_NAME = "export_receipt.json"
_OPTIMIZER_NAME = re.compile(r"(bf16_)?zero_pp_rank_(0|[1-9][0-9]*)_mp_rank_00_optim_states\.pt")


class CheckpointExportError(RuntimeError):
    """An export is unavailable, unsafe, incomplete, or outside its download budget."""


def _positive_int(value, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise CheckpointExportError(f"{label} must be a positive integer")
    return value


def _identifier(value, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", value):
        raise CheckpointExportError(f"invalid {label}")
    return value


def _size(value) -> int:
    # SnowAPI may encode uint64 sizes as decimal strings.
    if isinstance(value, str) and re.fullmatch(r"[0-9]+", value):
        value = int(value)
    return _positive_int(value, "declared file size")


def _https_url(value) -> str:
    if not isinstance(value, str) or any(ord(char) <= 32 or ord(char) >= 127 for char in value):
        raise CheckpointExportError("export URL must be an HTTPS URL")
    try:
        parts = urlsplit(value)
        valid = (
            parts.scheme == "https"
            and parts.hostname
            and parts.username is None
            and parts.password is None
            and not parts.fragment
        )
        parts.port
    except ValueError:
        valid = False
    if not valid:
        raise CheckpointExportError("export URL must be an HTTPS URL")
    return value


def _relative_filename(value) -> str:
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9_./-]+", value)
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise CheckpointExportError("unsafe export filename")
    return value


def _manifest(reply, checkpoint_id: str, dp_size: int, max_total_bytes: int) -> tuple[list[dict], str]:
    if not isinstance(reply, dict) or reply.get("checkpoint_id") != checkpoint_id:
        raise CheckpointExportError("export checkpoint identity mismatch")
    files = reply.get("files")
    if not isinstance(files, list) or len(files) != dp_size + 2:
        raise CheckpointExportError("export must contain the expected optimizer shards, model and metadata")
    entries = []
    names = set()
    ranks = set()
    tags = set()
    prefixes = set()
    model_count = 0
    metadata_count = 0
    total = 0
    for item in files:
        if not isinstance(item, dict):
            raise CheckpointExportError("invalid export file record")
        name = _relative_filename(item.get("filename"))
        if name in names:
            raise CheckpointExportError("duplicate export filename")
        names.add(name)
        if item.get("type", "file") != "file" or item.get("is_symlink", False) or "link_target" in item:
            raise CheckpointExportError("export symlinks are not supported")
        if name == "metadata.json":
            metadata_count += 1
        else:
            parts = name.split("/")
            if len(parts) != 2 or not re.fullmatch(r"global_step[0-9]+", parts[0]):
                raise CheckpointExportError("unsupported dense checkpoint layout")
            tags.add(parts[0])
            if parts[1] == "mp_rank_00_model_states.pt":
                model_count += 1
            else:
                match = _OPTIMIZER_NAME.fullmatch(parts[1])
                if match is None:
                    raise CheckpointExportError("unsupported dense optimizer shard filename")
                rank = int(match[2])
                if rank in ranks or rank >= dp_size:
                    raise CheckpointExportError("duplicate or unexpected optimizer rank")
                ranks.add(rank)
                prefixes.add(match[1] or "")
        size = _size(item.get("size_bytes"))
        total += size
        if total > max_total_bytes:
            raise CheckpointExportError("export exceeds the total download byte budget")
        # Only the documented public field is accepted; never guess URL aliases.
        entries.append({"path": name, "size_bytes": size, "url": _https_url(item.get("url"))})
    if ranks != set(range(dp_size)) or len(tags) != 1 or len(prefixes) != 1 or model_count != 1 or metadata_count != 1:
        raise CheckpointExportError("incomplete or inconsistent dense checkpoint layout")
    return sorted(entries, key=lambda entry: entry["path"]), tags.pop()


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise CheckpointExportError("checkpoint download deadline exceeded")
    return remaining


def _download_response(session: requests.Session, url: str, deadline: float):
    # Follow presigned cross-host redirects without the Cortex PAT or netrc auth,
    # and reject HTTPS downgrades before sending a redirected request.
    for hop in range(6):
        remaining = _remaining(deadline)
        response = session.get(
            url,
            stream=True,
            allow_redirects=False,
            timeout=(min(30.0, remaining), min(60.0, remaining)),
            headers={"Accept-Encoding": "identity"},
        )
        if response.status_code not in _REDIRECT_STATUSES:
            return response
        try:
            if hop == 5:
                raise CheckpointExportError("checkpoint download redirect limit exceeded")
            location = response.headers.get("Location")
            if not isinstance(location, str) or not location:
                raise CheckpointExportError("checkpoint download redirect has no location")
            url = _https_url(urljoin(url, location))
        finally:
            response.close()
    raise CheckpointExportError("checkpoint download redirect limit exceeded")


def _open_exclusive(root: Path, name: str):
    # Anchor directory traversal to no-follow directory descriptors, not paths
    # that might have been replaced with symlinks between mkdir and open.
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        parts = name.split("/")
        for part in parts[:-1]:
            try:
                os.mkdir(part, mode=0o700, dir_fd=fd)
            except FileExistsError:
                pass
            child_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            os.close(fd)
            fd = child_fd
        file_fd = os.open(parts[-1], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, mode=0o600, dir_fd=fd)
        return os.fdopen(file_fd, "wb")
    finally:
        os.close(fd)


def _download_file(session: requests.Session, root: Path, entry: dict, deadline: float) -> dict:
    response = _download_response(session, entry["url"], deadline)
    try:
        if response.status_code != 200:
            raise CheckpointExportError(f"checkpoint download HTTP status {int(response.status_code)}")
        if response.headers.get("Content-Encoding", "identity").lower() != "identity":
            raise CheckpointExportError("checkpoint download must be uncompressed")
        expected = entry["size_bytes"]
        length = response.headers.get("Content-Length")
        if length is not None and _size(length) != expected:
            raise CheckpointExportError("checkpoint Content-Length differs from the declared size")
        raw = getattr(response, "raw", None)
        if response.headers.get("Transfer-Encoding") or getattr(raw, "chunked", False):
            raise CheckpointExportError("checkpoint download chunked framing cannot enforce the deadline")
        read1 = getattr(raw, "read1", None)
        sock = getattr(getattr(raw, "connection", None), "sock", None)
        if sock is None:
            # Connection: close transfers ownership from the connection to the
            # http.client response's buffered SocketIO while the body stays open.
            buffer = getattr(getattr(raw, "_fp", None), "fp", None)
            sock = getattr(getattr(buffer, "raw", None), "_sock", None)
        settimeout = getattr(sock, "settimeout", None)
        if not callable(read1) or not callable(settimeout):
            raise CheckpointExportError("checkpoint download requires bounded raw reads")
        digest = hashlib.sha256()
        count = 0
        with _open_exclusive(root, entry["path"]) as output:
            # read1 returns available bytes after at most one socket read, unlike
            # iter_content which can wait for a full chunk despite a slow trickle.
            while count < expected or length is None:
                settimeout(min(60.0, _remaining(deadline)))
                limit = min(_CHUNK_BYTES, expected - count + 1)
                chunk = read1(limit, decode_content=False)
                _remaining(deadline)
                if not isinstance(chunk, bytes) or len(chunk) > limit:
                    raise CheckpointExportError("checkpoint download requires bounded raw bytes")
                if not chunk:
                    break
                count += len(chunk)
                if count > expected:
                    raise CheckpointExportError("checkpoint download exceeds the declared size")
                output.write(chunk)
                digest.update(chunk)
            if count != expected:
                raise CheckpointExportError("checkpoint download is truncated")
            _remaining(deadline)
            output.flush()
            os.fsync(output.fileno())
        return {"path": entry["path"], "size_bytes": count, "sha256": digest.hexdigest()}
    finally:
        response.close()


def _destination_parent(output_dir) -> Path:
    parent = Path(output_dir).expanduser().absolute()
    if any(path.is_symlink() for path in (parent, *parent.parents)):
        raise CheckpointExportError("checkpoint destination must not traverse a symlink")
    try:
        parent = require_data_fast_path(parent, label="checkpoint export destination")
    except ValueError as exc:
        raise CheckpointExportError(str(exc)) from None
    if parent.exists() and not parent.is_dir():
        raise CheckpointExportError("checkpoint destination must be a directory")
    return parent


def export_checkpoint(
    client,
    job_id: str,
    checkpoint_id: str,
    output_dir: Path,
    *,
    expected_dp_world_size: int,
    max_total_bytes: int = 128 << 30,
    timeout_s: float = 1800.0,
) -> Path:
    """Return a fresh complete snapshot root; the caller owns successful cleanup.

    Only dense single-model-rank layouts are supported. The receipt binds bytes,
    not tensor semantics: the optimizer decoder must validate stage and identity.
    """
    job_id = _identifier(job_id, "job ID")
    checkpoint_id = _identifier(checkpoint_id, "checkpoint ID")
    dp_size = _positive_int(expected_dp_world_size, "expected DP world size")
    _positive_int(max_total_bytes, "maximum total bytes")
    if (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not math.isfinite(timeout_s)
        or timeout_s <= 0
    ):
        raise CheckpointExportError("download timeout must be finite and positive")
    transport = client.transport
    if transport.job_id != job_id:
        raise CheckpointExportError("export job does not match the client parent job")
    root = None
    failure = None
    complete = False
    try:
        parent = _destination_parent(output_dir)
        # Preserve the original public client's framing, auth and retry policy.
        reply = transport._send("POST", f"{transport._prefix}/{job_id}/checkpoints/{checkpoint_id}:export")
        entries, tag = _manifest(reply, checkpoint_id, dp_size, max_total_bytes)
        parent.mkdir(parents=True, exist_ok=True)
        if not parent.is_dir() or any(path.is_symlink() for path in (parent, *parent.parents)):
            raise CheckpointExportError("checkpoint destination must be a directory without symlinks")
        root = Path(tempfile.mkdtemp(prefix="checkpoint-export-", dir=parent))
        deadline = time.monotonic() + timeout_s
        with requests.Session() as session:
            session.trust_env = False
            session.auth = None
            files = [_download_file(session, root, entry, deadline) for entry in entries]
        receipt = {
            "schema_version": 1,
            "complete": True,
            "source_job_id": job_id,
            "checkpoint_id": checkpoint_id,
            "dp_world_size": dp_size,
            "tag": tag,
            "total_bytes": sum(entry["size_bytes"] for entry in files),
            "files": files,
        }
        with _open_exclusive(root, _RECEIPT_NAME) as output:
            output.write((json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("ascii"))
            output.flush()
            os.fsync(output.fileno())
        complete = True
        return root
    except CheckpointExportError as exc:
        failure = str(exc)
    except Exception:
        # HTTP/IO exception text can include signed URLs, bodies or credentials.
        failure = "checkpoint export or download failed"
    finally:
        if not complete and root is not None:
            try:
                shutil.rmtree(root)
            except OSError:
                raise CheckpointExportError("checkpoint export failed and snapshot cleanup failed") from None
    raise CheckpointExportError(failure) from None
