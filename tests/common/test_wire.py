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

import pytest
import torch

from arctic_platform import wire


@pytest.mark.parametrize("max_bytes", [1, 4096, 8192])
def test_encode_byte_chunks_rejects_ceiling_at_reserved_overhead(max_bytes):
    original = wire.dumps({"x": torch.arange(2048, dtype=torch.int64)})
    with pytest.raises(wire.WireError, match="max_bytes must exceed 8192 bytes"):
        wire.encode_byte_chunks(original, kind="request", max_bytes=max_bytes)


def test_encode_byte_chunks_hashes_the_logical_frame_once(monkeypatch):
    """Chunking a frame must not scale full-frame hashing with the chunk count.

    Every chunk descriptor carries the digest of the same whole logical frame, so
    one pass is enough. Recomputing it per chunk made the cost of chunking a
    large frame quadratic-ish in practice, since the sizing loop can rebuild the
    chunk list more than once before it settles.
    """
    original = wire.dumps({"x": torch.arange(8192, dtype=torch.int64)})
    real_sha256 = wire.hashlib.sha256
    hashed = []

    def recording_sha256(payload=b""):
        hashed.append(payload)
        return real_sha256(payload)

    monkeypatch.setattr(wire.hashlib, "sha256", recording_sha256)
    chunks = wire.encode_byte_chunks(original, kind="request", max_bytes=16_384)

    assert len(chunks) > 1
    assert all(len(chunk) <= 16_384 for chunk in chunks)
    assert hashed == [original]
    digests = {wire.read_byte_chunk_metadata(chunk)["frame_sha256"] for chunk in chunks}
    assert digests == {real_sha256(original).hexdigest()}
    assert wire.decode_byte_chunks(chunks, kind="request") == original


def test_force_chunk_empty_frame_emits_one_chunk():
    chunks = wire.encode_byte_chunks(b"", kind="request", max_bytes=16_384, force_chunk=True)

    assert len(chunks) == 1
    assert wire.decode_byte_chunks(chunks, kind="request") == b""


def test_read_frame_index_reads_the_header_alone():
    frame = wire.dumps({"ids": torch.arange(6, dtype=torch.int32).view(2, 3), "note": "x", "rows": [torch.ones(4)]})
    index, tree = wire.read_frame_index(frame)
    assert index["ids"]["dtype"] == "I32" and index["ids"]["shape"] == [2, 3]
    assert index["rows.0"]["shape"] == [4]
    assert tree[0] == "d" and dict(tree[1])["note"] == ["j", "x"]
    header_end = 8 + int.from_bytes(frame[:8], "little")
    start, end = index["ids"]["data_offsets"]
    assert torch.frombuffer(bytearray(frame[header_end + start : header_end + end]), dtype=torch.int32).tolist() == [
        0, 1, 2, 3, 4, 5,
    ]
    # Only the header has to be there, and a memoryview is read without copying the frame.
    assert wire.read_frame_index(memoryview(frame)[:header_end]) == (index, tree)


def test_read_frame_index_has_no_placeholder_for_a_tensorless_frame():
    index, tree = wire.read_frame_index(wire.dumps({"a": 1}))
    assert index == {} and tree == ["j", {"a": 1}]


@pytest.mark.parametrize(
    "data",
    [b"", b"short", b"\x08\x00\x00\x00\x00\x00\x00\x00{not json", b"\xff" * 8 + b"{}", 3],
    ids=["empty", "short", "bad-json", "huge-length", "not-bytes"],
)
def test_read_frame_index_is_none_for_what_is_not_a_whole_header(data):
    assert wire.read_frame_index(data) is None
    frame = wire.dumps({"x": torch.ones(2)})
    assert wire.read_frame_index(frame[: 8 + int.from_bytes(frame[:8], "little") - 1]) is None


def test_public_size_constants_match_the_private_ones():
    assert wire.MAX_HEADER_BYTES == wire._MAX_HEADER_BYTES
    assert wire.DEFAULT_CHUNK_OVERHEAD_BYTES == wire._DEFAULT_CHUNK_OVERHEAD_BYTES
