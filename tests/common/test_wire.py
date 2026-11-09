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
