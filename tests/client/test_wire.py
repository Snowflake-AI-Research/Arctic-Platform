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
"""DSSST1 wire codec tests.

`wire.py` is the single source of truth for the binary wire format shared across
dss-platform, ArcticRL, and this client, so silent codec bugs corrupt every
payload. These tests are pure (CPU torch + safetensors), no GPU or network.
"""

from __future__ import annotations

import io
import pickle

import pytest
import torch

from arctic_platform.client import wire


class TestCodec:
    def test_roundtrip_nested_mixed(self):
        """Nested dict/list/tuple mixing tensors and JSON survive a roundtrip."""
        obj = {
            "weights": torch.arange(6).reshape(2, 3),
            "meta": {"lr": 0.1, "name": "x"},
            "list": [torch.ones(2), 5, "s"],
            "tup": (torch.zeros(1), 2),
        }
        back = wire.loads(wire.dumps(obj))
        assert torch.equal(back["weights"], obj["weights"])
        assert back["meta"] == {"lr": 0.1, "name": "x"}
        assert torch.equal(back["list"][0], torch.ones(2))
        assert back["list"][1:] == [5, "s"]
        assert torch.equal(back["tup"][0], torch.zeros(1))
        assert back["tup"][1] == 2

    def test_scalar_tensor_dim_preserved(self):
        """A 0-dim tensor comes back 0-dim (not reshaped to [1])."""
        back = wire.loads(wire.dumps({"s": torch.tensor(3.5)}))
        assert back["s"].shape == torch.Size([])
        assert float(back["s"]) == pytest.approx(3.5)

    def test_tuple_vs_list_distinction_preserved(self):
        """tuple stays tuple, list stays list."""
        back = wire.loads(wire.dumps({"t": (torch.ones(1),), "l": [torch.ones(1)]}))
        assert isinstance(back["t"], tuple)
        assert isinstance(back["l"], list)

    def test_no_tensor_payload(self):
        """A tensor-free object roundtrips via the placeholder path."""
        obj = {"only": "json", "nums": [1, 2, 3]}
        assert wire.loads(wire.dumps(obj)) == obj

    def test_metadata_read_without_loading_tensors(self):
        """read_metadata returns op metadata without decoding tensors."""
        frame = wire.dumps({"x": torch.ones(3)}, metadata={"router_replay": [1, 2], "op": "fwd-bwd"})
        assert wire.read_metadata(frame) == {"router_replay": [1, 2], "op": "fwd-bwd"}

    def test_read_metadata_lenient_on_garbage(self):
        """Non-frame bytes have no metadata rather than raising."""
        assert wire.read_metadata(b"not a frame") == {}


class TestByteChunks:
    def _big_frame(self) -> bytes:
        return wire.dumps({"big": torch.arange(100_000, dtype=torch.int64)})

    def test_single_frame_passthrough(self):
        """No chunking needed -> the frame is returned as-is and decodes back."""
        frame = self._big_frame()
        chunks = wire.encode_byte_chunks(frame, kind="request")
        assert chunks == [frame]
        assert wire.decode_byte_chunks(chunks) == frame

    def test_multichunk_roundtrip(self):
        """A frame larger than max_bytes splits and reassembles losslessly."""
        frame = self._big_frame()
        chunks = wire.encode_byte_chunks(frame, kind="request", operation="fwd-bwd", max_bytes=16_384)
        assert len(chunks) > 1
        assert wire.decode_byte_chunks(chunks, kind="request") == frame

    def test_missing_chunk_detected(self):
        """Dropping a chunk is caught by the count check."""
        chunks = wire.encode_byte_chunks(self._big_frame(), kind="request", max_bytes=16_384)
        with pytest.raises(wire.WireError, match="byte chunks"):
            wire.decode_byte_chunks(chunks[:-1], kind="request")

    def test_duplicate_chunk_detected(self):
        """A repeated chunk index is rejected."""
        chunks = wire.encode_byte_chunks(self._big_frame(), kind="request", max_bytes=16_384)
        tampered = chunks[:-1] + [chunks[0]]
        with pytest.raises(wire.WireError, match="duplicate"):
            wire.decode_byte_chunks(tampered, kind="request")

    def test_wrong_kind_rejected(self):
        """Asking for result chunks when given request chunks fails."""
        chunks = wire.encode_byte_chunks(self._big_frame(), kind="request", max_bytes=16_384)
        with pytest.raises(wire.WireError, match="expected result"):
            wire.decode_byte_chunks(chunks, kind="result")

    def test_cross_frame_chunks_detected(self):
        """Chunks from two different frames (same group) fail the sha256 cross-check."""
        frame_a = wire.dumps({"x": torch.arange(50_000, dtype=torch.int64)})
        frame_b = wire.dumps({"y": torch.arange(50_000, dtype=torch.int64) + 1})
        ca = wire.encode_byte_chunks(frame_a, kind="request", max_bytes=16_384, chunk_group_id="g")
        cb = wire.encode_byte_chunks(frame_b, kind="request", max_bytes=16_384, chunk_group_id="g")
        assert len(ca) == len(cb) > 1
        with pytest.raises(wire.WireError, match="frame_sha256 differs"):
            wire.decode_byte_chunks([ca[0]] + cb[1:], kind="request")

    def test_result_chunks_roundtrip(self):
        """encode_result_chunks/decode_result_chunks roundtrip a tensor result."""
        result = {"results": [torch.arange(100_000, dtype=torch.int64)]}
        chunks = wire.encode_result_chunks(result, max_bytes=16_384)
        assert len(chunks) > 1
        back = wire.decode_result_chunks(chunks)
        assert torch.equal(back["results"][0], result["results"][0])


class TestSafety:
    def test_pickle_payload_rejected(self):
        """A pickle payload is refused, never fed to pickle."""
        with pytest.raises(wire.WireError, match="refusing to deserialize"):
            wire.loads(pickle.dumps({"a": 1}))

    def test_torch_save_payload_rejected(self):
        """A torch.save payload is refused (zip signature)."""
        buf = io.BytesIO()
        torch.save({"a": torch.ones(1)}, buf)
        with pytest.raises(wire.WireError, match="refusing to deserialize"):
            wire.loads(buf.getvalue())

    def test_garbage_payload_rejected(self):
        """Random bytes are rejected as not-a-frame."""
        with pytest.raises(wire.WireError, match="not a valid DSSST1"):
            wire.loads(b"\x00\x01\x02\x03not safetensors")
