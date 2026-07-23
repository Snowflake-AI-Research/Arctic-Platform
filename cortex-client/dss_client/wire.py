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

"""DSS binary wire protocol (DSSST1).

This is the single source of truth for how dss serializes binary training
payloads. ``dss-platform`` and ``ArcticRL`` depend on ``dss-client`` and import
this module, so all three sides share one implementation.

It owns three concerns behind one small surface:

1. Codec -- ``dumps`` / ``loads`` serialize a nested dict/list/tuple of tensors
   and JSON-able data as a single **safetensors** blob. Pickle is never used, so
   nothing can execute during decode (unlike ``torch.load``).

2. Operation metadata -- ``dumps(obj, metadata=...)`` stores small control-plane
   metadata (router replay, chunk descriptors) inside the safetensors
   ``__metadata__`` header. ``read_metadata`` reads it back **without**
   deserializing tensors. This replaces the old hand-rolled ``DSSMETA1`` byte
   envelope -- there is now exactly one framing.

3. Byte chunking -- ``encode_byte_chunks`` splits an oversized DSSST1 frame into
   self-describing byte-range frames. ``decode_byte_chunks`` reassembles them.
   The split/merge contract lives in one place so client and server cannot
   drift.

Wire layout of one frame is just a safetensors blob::

    [ u64 header_len | JSON header (__metadata__: {dss: <structure>, op: <metadata>}) | tensor bytes ]
"""

from __future__ import annotations

import json
import hashlib
from typing import Any

import torch
from safetensors.torch import load as st_load
from safetensors.torch import save as st_save

WIRE_FORMAT_VERSION = "DSSST1"

_STRUCT_KEY = "dss"  # structure skeleton (internal)
_OP_KEY = "op"  # operation metadata (router_replay, request/result chunks, ...)
_REQUEST_CHUNK_KEY = "request_chunk"
_RESULT_CHUNK_KEY = "result_chunk"
_PLACEHOLDER_KEY = "__dss_empty__"
_ROOT_KEY = "__root__"
_MAX_HEADER_BYTES = 64 * 1024 * 1024
_CHUNK_PAYLOAD_KEY = "payload"
_DEFAULT_CHUNK_OVERHEAD_BYTES = 8192

_PICKLE_SIGNATURES = (b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05")
_ZIP_SIGNATURE = b"PK\x03\x04"

class WireError(ValueError):
    """Raised for malformed or incompatible wire payloads."""


def _is_tensor(obj: Any) -> bool:
    return torch.is_tensor(obj)


# ---------------------------------------------------------------------------
# Codec
# ---------------------------------------------------------------------------


def _subtree_has_tensor(obj: Any) -> bool:
    if _is_tensor(obj):
        return True
    if isinstance(obj, dict):
        return any(_subtree_has_tensor(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_subtree_has_tensor(v) for v in obj)
    return False


def _encode(obj: Any, tensors: dict, seen: set, path: str) -> list:
    if _is_tensor(obj):
        key = path or _ROOT_KEY
        tensor = obj.detach().cpu().contiguous()
        node: list = ["t", key]
        if tensor.dim() == 0:
            tensor = tensor.reshape(1)
            node = ["t", key, {"scalar": True}]
        # safetensors rejects tensors that share storage; clone on collision.
        if tensor.data_ptr() in seen and tensor.numel() > 0:
            tensor = tensor.clone()
        seen.add(tensor.data_ptr())
        tensors[key] = tensor
        return node
    if isinstance(obj, dict):
        if not _subtree_has_tensor(obj):
            return ["j", obj]
        items = []
        for raw_key, value in obj.items():
            if not isinstance(raw_key, str):
                raise WireError(f"DSSST1 dict keys must be strings, got {type(raw_key)!r}")
            child_path = f"{path}.{raw_key}" if path else raw_key
            items.append([raw_key, _encode(value, tensors, seen, child_path)])
        return ["d", items]
    if isinstance(obj, tuple):
        return ["u", [_encode(v, tensors, seen, f"{path}.{i}" if path else str(i)) for i, v in enumerate(obj)]]
    if isinstance(obj, list):
        if not _subtree_has_tensor(obj):
            return ["j", obj]
        return ["l", [_encode(v, tensors, seen, f"{path}.{i}" if path else str(i)) for i, v in enumerate(obj)]]
    return ["j", obj]


def dumps(obj: Any, *, metadata: dict | None = None) -> bytes:
    """Serialize ``obj`` to one safetensors frame, with optional op ``metadata``."""
    tensors: dict = {}
    tree = _encode(obj, tensors, set(), "")
    if not tensors:
        tensors[_PLACEHOLDER_KEY] = torch.zeros(1, dtype=torch.uint8)
    header = {_STRUCT_KEY: json.dumps({"v": WIRE_FORMAT_VERSION, "tree": tree})}
    if metadata:
        header[_OP_KEY] = json.dumps(metadata)
    return st_save(tensors, metadata=header)


def _decode(node: list, tensors: dict) -> Any:
    tag = node[0]
    if tag == "t":
        tensor = tensors[node[1]]
        if len(node) > 2 and isinstance(node[2], dict) and node[2].get("scalar"):
            return tensor.reshape(())
        return tensor
    if tag == "j":
        return node[1]
    if tag == "d":
        return {key: _decode(child, tensors) for key, child in node[1]}
    if tag == "l":
        return [_decode(child, tensors) for child in node[1]]
    if tag == "u":
        return tuple(_decode(child, tensors) for child in node[1])
    raise WireError(f"unknown DSSST1 node tag {tag!r}")


def _read_header(data: bytes) -> dict:
    if len(data) < 8:
        raise WireError("payload too short to be a DSSST1 safetensors blob")
    header_len = int.from_bytes(data[:8], "little")
    if header_len <= 0 or 8 + header_len > len(data) or header_len > _MAX_HEADER_BYTES:
        raise WireError("invalid DSSST1 safetensors header length")
    header = json.loads(data[8 : 8 + header_len].decode("utf-8"))
    return header.get("__metadata__", {}) or {}


def loads(data: Any) -> Any:
    """Deserialize a frame produced by :func:`dumps`.

    Never invokes pickle/``torch.load``; a legacy pickle or ``torch.save``
    payload is rejected with a clear error instead of being executed.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise WireError("DSSST1 loads expects bytes")
    data = bytes(data)
    try:
        tensors = st_load(data)
        header = _read_header(data)
    except Exception as exc:  # not safetensors -> never feed it to pickle
        if data[:2] in _PICKLE_SIGNATURES or data[:4] == _ZIP_SIGNATURE:
            raise WireError(
                "refusing to deserialize a legacy pickle/torch.save payload; "
                "the peer must use the DSSST1 safetensors wire protocol"
            ) from exc
        raise WireError(f"not a valid DSSST1 safetensors payload: {exc}") from exc

    raw = header.get(_STRUCT_KEY)
    if raw is None:
        raise WireError("DSSST1 structure missing from safetensors header")
    info = json.loads(raw)
    if info.get("v") != WIRE_FORMAT_VERSION:
        raise WireError(f"unsupported DSSST1 wire version {info.get('v')!r}")
    tensors.pop(_PLACEHOLDER_KEY, None)
    return _decode(info["tree"], tensors)


def read_metadata(data: Any) -> dict:
    """Return the operation metadata from a frame header without loading tensors.

    This is a lenient peek: bytes that aren't a valid DSSST1 frame simply have no
    operation metadata, so ``{}`` is returned. The authoritative (pickle-safe)
    decode happens in :func:`loads`.
    """
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise WireError("DSSST1 read_metadata expects bytes")
    try:
        raw = _read_header(bytes(data)).get(_OP_KEY)
    except (WireError, ValueError):
        return {}
    return json.loads(raw) if raw else {}


# ---------------------------------------------------------------------------
# Byte chunking -- split and reassemble DSSST1 frames
# ---------------------------------------------------------------------------


def _chunk_key(kind: str) -> str:
    if kind == "request":
        return _REQUEST_CHUNK_KEY
    if kind == "result":
        return _RESULT_CHUNK_KEY
    raise WireError(f"unknown byte chunk kind {kind!r}")


def _bytes_to_tensor(data: bytes) -> torch.Tensor:
    # bytearray makes the buffer writable; clone detaches from the temporary.
    return torch.frombuffer(bytearray(data), dtype=torch.uint8).clone()


def _tensor_to_bytes(tensor: Any) -> bytes:
    if not torch.is_tensor(tensor):
        raise WireError("byte chunk payload must be a tensor")
    if tensor.dtype != torch.uint8:
        raise WireError(f"byte chunk payload tensor must be uint8, got {tensor.dtype}")
    if tensor.ndim != 1:
        raise WireError(f"byte chunk payload tensor must be 1-D, got shape {tuple(tensor.shape)}")
    return tensor.cpu().contiguous().numpy().tobytes()


def _make_byte_chunk_frame(
    frame: bytes,
    *,
    kind: str,
    operation: str | None,
    group_id: str | None,
    chunk_idx: int,
    total_chunks: int,
    start: int,
    end: int,
) -> bytes:
    key = _chunk_key(kind)
    desc: dict[str, Any] = {
        "chunk_idx": chunk_idx,
        "total_chunks": total_chunks,
        "frame_sha256": hashlib.sha256(frame).hexdigest(),
        "frame_size_bytes": len(frame),
    }
    if group_id is not None:
        desc["chunk_group_id"] = group_id
    if operation is not None:
        desc["operation"] = operation
    return dumps({_CHUNK_PAYLOAD_KEY: _bytes_to_tensor(frame[start:end])}, metadata={key: desc})


def encode_byte_chunks(
    frame: bytes,
    *,
    kind: str,
    operation: str | None = None,
    max_bytes: int = 0,
    chunk_group_id: str | None = None,
) -> list[bytes]:
    """Split a DSSST1 frame into byte-range DSSST1 chunk frames.

    Returns ``[frame]`` when no chunking is needed. When chunking is needed,
    every returned frame contains a 1-D uint8 ``payload`` tensor plus metadata
    under ``request_chunk`` or ``result_chunk``.
    """
    if not isinstance(frame, (bytes, bytearray, memoryview)):
        raise WireError("encode_byte_chunks expects frame bytes")
    frame = bytes(frame)
    if max_bytes <= 0 or len(frame) <= max_bytes:
        return [frame]

    group_id = chunk_group_id if kind == "request" else None
    if kind == "request" and not group_id:
        import uuid

        group_id = str(uuid.uuid4())

    payload_size = max(1, max_bytes - _DEFAULT_CHUNK_OVERHEAD_BYTES)
    while True:
        ranges = [(start, min(start + payload_size, len(frame))) for start in range(0, len(frame), payload_size)]
        total = len(ranges)
        chunks = [
            _make_byte_chunk_frame(
                frame,
                kind=kind,
                operation=operation,
                group_id=group_id,
                chunk_idx=idx,
                total_chunks=total,
                start=start,
                end=end,
            )
            for idx, (start, end) in enumerate(ranges)
        ]
        if all(len(chunk) <= max_bytes for chunk in chunks):
            return chunks
        if payload_size == 1:
            raise WireError("max_bytes is too small to hold one DSSST1 byte chunk frame")
        payload_size = max(1, payload_size // 2)


def read_byte_chunk_metadata(frame: Any) -> dict | None:
    """Return byte chunk metadata with ``kind`` included, or ``None``."""
    metadata = read_metadata(frame)
    if _REQUEST_CHUNK_KEY in metadata:
        desc = dict(metadata[_REQUEST_CHUNK_KEY])
        desc["kind"] = "request"
        return desc
    if _RESULT_CHUNK_KEY in metadata:
        desc = dict(metadata[_RESULT_CHUNK_KEY])
        desc["kind"] = "result"
        return desc
    return None


def decode_byte_chunks(chunks: list[bytes], *, kind: str | None = None) -> bytes:
    """Reassemble byte-range DSSST1 chunks into the original DSSST1 frame."""
    if not chunks:
        raise WireError("decode_byte_chunks requires at least one frame")
    if len(chunks) == 1 and read_byte_chunk_metadata(chunks[0]) is None:
        return bytes(chunks[0])

    descriptors = [read_byte_chunk_metadata(chunk) for chunk in chunks]
    if any(desc is None for desc in descriptors):
        raise WireError("decode_byte_chunks received a frame without byte chunk metadata")
    descriptors = [desc for desc in descriptors if desc is not None]
    actual_kind = descriptors[0]["kind"]
    if kind is not None and actual_kind != kind:
        raise WireError(f"expected {kind} chunks, got {actual_kind} chunks")

    total = int(descriptors[0].get("total_chunks"))
    frame_sha256 = descriptors[0].get("frame_sha256")
    frame_size_bytes = int(descriptors[0].get("frame_size_bytes"))
    group_id = descriptors[0].get("chunk_group_id")
    operation = descriptors[0].get("operation")

    if len(chunks) != total:
        raise WireError(f"expected {total} byte chunks, got {len(chunks)}")

    seen: set[int] = set()
    ordered: list[tuple[int, bytes]] = []
    for chunk, desc in zip(chunks, descriptors):
        if desc["kind"] != actual_kind:
            raise WireError("mixed byte chunk kinds")
        if int(desc.get("total_chunks")) != total:
            raise WireError("byte chunk total_chunks differs across frames")
        if desc.get("frame_sha256") != frame_sha256:
            raise WireError("byte chunk frame_sha256 differs across frames")
        if int(desc.get("frame_size_bytes")) != frame_size_bytes:
            raise WireError("byte chunk frame_size_bytes differs across frames")
        if desc.get("chunk_group_id") != group_id:
            raise WireError("byte chunk chunk_group_id differs across frames")
        if desc.get("operation") != operation:
            raise WireError("byte chunk operation differs across frames")
        idx = int(desc.get("chunk_idx"))
        if idx < 0 or idx >= total:
            raise WireError("byte chunk_idx out of range")
        if idx in seen:
            raise WireError("duplicate byte chunk_idx")
        seen.add(idx)
        payload = loads(chunk).get(_CHUNK_PAYLOAD_KEY)
        ordered.append((idx, _tensor_to_bytes(payload)))

    if seen != set(range(total)):
        missing = sorted(set(range(total)) - seen)
        raise WireError(f"missing byte chunks: {missing}")

    frame = b"".join(payload for _, payload in sorted(ordered))
    if len(frame) != frame_size_bytes:
        raise WireError("reassembled frame size mismatch")
    if hashlib.sha256(frame).hexdigest() != frame_sha256:
        raise WireError("reassembled frame sha256 mismatch")
    return frame


def encode_result_chunks(result: Any, *, max_bytes: int = 0) -> list[bytes]:
    """Serialize and optionally byte-chunk a result object."""
    return encode_byte_chunks(dumps(result), kind="result", max_bytes=max_bytes)


def decode_result_chunks(chunks: list[bytes]) -> Any:
    """Decode a result object returned by :func:`encode_result_chunks`."""
    return loads(decode_byte_chunks(chunks, kind="result"))
