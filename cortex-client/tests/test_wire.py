import io

import pytest
import torch

from dss_client import wire


def _assert_equal(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        assert torch.is_tensor(a) and torch.is_tensor(b)
        assert a.dtype == b.dtype
        assert a.shape == b.shape
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b)
        for key in a:
            _assert_equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b):
            _assert_equal(left, right)
    else:
        assert a == b


# ---------------------------------------------------------------------------
# Codec round-trips
# ---------------------------------------------------------------------------


def test_roundtrip_nested_structure():
    obj = {
        "input_ids": torch.tensor([[1, 2], [3, 4]]),
        "advantages": torch.tensor([0.5, -0.5]),
        "meta": {"loss_agg_mode": "prompt-mean", "steps": 3},
        "sample_ids": ["s0", "s1"],
        "nested": [torch.tensor([1, 2, 3]), {"x": torch.tensor([[7.0]])}],
        "pair": (torch.tensor([9]), "tag"),
    }
    _assert_equal(wire.loads(wire.dumps(obj)), obj)


def test_roundtrip_scalar_tensor_preserves_zero_dim():
    obj = {"temperature": torch.tensor(1.5)}
    out = wire.loads(wire.dumps(obj))
    assert out["temperature"].ndim == 0
    assert out["temperature"].item() == pytest.approx(1.5)


@pytest.mark.parametrize(
    "dtype",
    [torch.float32, torch.bfloat16, torch.int64, torch.int32, torch.bool],
)
def test_roundtrip_dtypes(dtype):
    if dtype == torch.bool:
        t = torch.tensor([[True, False], [False, True]])
    else:
        t = torch.arange(6, dtype=dtype).reshape(2, 3)
    _assert_equal(wire.loads(wire.dumps({"t": t})), {"t": t})


def test_roundtrip_json_only_payload_has_no_tensors():
    obj = {"only": {"json": [1, 2, 3], "flag": True}}
    _assert_equal(wire.loads(wire.dumps(obj)), obj)


def test_roundtrip_shared_storage_tensors_are_cloned():
    base = torch.arange(4)
    obj = {"a": base, "b": base}  # same storage
    out = wire.loads(wire.dumps(obj))
    _assert_equal(out, obj)


# ---------------------------------------------------------------------------
# Operation metadata
# ---------------------------------------------------------------------------


def test_metadata_roundtrip_via_read_metadata():
    meta = {"router_replay": {"sampling_job_id": "20"}}
    frame = wire.dumps({"x": torch.tensor([1])}, metadata=meta)
    assert wire.read_metadata(frame) == meta
    # tensors still decode normally
    _assert_equal(wire.loads(frame), {"x": torch.tensor([1])})


def test_read_metadata_absent_returns_empty():
    frame = wire.dumps({"x": torch.tensor([1])})
    assert wire.read_metadata(frame) == {}


def test_read_metadata_lenient_on_garbage():
    assert wire.read_metadata(b"not-a-frame") == {}


# ---------------------------------------------------------------------------
# Safety: never feed pickle/torch.save to the decoder
# ---------------------------------------------------------------------------


def test_loads_rejects_torch_save_payload():
    buf = io.BytesIO()
    torch.save({"a": torch.tensor([1, 2, 3])}, buf)
    with pytest.raises(wire.WireError):
        wire.loads(buf.getvalue())


def test_loads_rejects_raw_pickle():
    import pickle

    with pytest.raises(wire.WireError):
        wire.loads(pickle.dumps({"a": 1}))


def test_loads_rejects_non_frame_bytes():
    with pytest.raises(wire.WireError):
        wire.loads(b"short")


# ---------------------------------------------------------------------------
# Byte chunking: encode_byte_chunks / decode_byte_chunks
# ---------------------------------------------------------------------------


def test_encode_byte_chunks_single_frame_when_unbounded():
    frame = wire.dumps({"x": torch.tensor([1])})
    assert wire.encode_byte_chunks(frame, kind="request", max_bytes=0) == [frame]


def test_encode_byte_chunks_and_decodes_roundtrip():
    original = wire.dumps({"x": torch.arange(2048, dtype=torch.int64)})
    chunks = wire.encode_byte_chunks(
        original, kind="request", operation="fwd-bwd", max_bytes=4096
    )
    assert len(chunks) > 1
    for chunk in chunks:
        desc = wire.read_byte_chunk_metadata(chunk)
        assert desc is not None
        assert desc["kind"] == "request"
        assert desc["operation"] == "fwd-bwd"
        assert desc["total_chunks"] == len(chunks)

    assert wire.decode_byte_chunks(list(reversed(chunks)), kind="request") == original


def test_encode_result_chunks_decodes_result_roundtrip():
    result = {
        "job_id": "j1",
        "results": [{"text": "hello"}, {"token_ids": [1, 2, 3]}],
    }
    chunks = wire.encode_result_chunks(result, max_bytes=4096)
    out = wire.decode_result_chunks(chunks)
    _assert_equal(out, result)


def test_decode_byte_chunks_detects_missing_chunk():
    original = wire.dumps({"x": torch.arange(2048, dtype=torch.int64)})
    chunks = wire.encode_byte_chunks(original, kind="request", max_bytes=4096)
    assert len(chunks) > 1
    with pytest.raises(wire.WireError, match="expected .* byte chunks"):
        wire.decode_byte_chunks(chunks[:-1], kind="request")


def test_decode_byte_chunks_detects_mismatched_kind():
    original = wire.dumps({"x": torch.arange(2048, dtype=torch.int64)})
    chunks = wire.encode_byte_chunks(original, kind="result", max_bytes=4096)
    with pytest.raises(wire.WireError, match="expected request chunks"):
        wire.decode_byte_chunks(chunks, kind="request")


def test_decode_byte_chunks_rejects_plain_frame_in_multiframe_input():
    plain = wire.dumps({"x": torch.tensor([1])})
    chunked = wire.encode_byte_chunks(
        wire.dumps({"x": torch.arange(2048, dtype=torch.int64)}),
        kind="request",
        max_bytes=4096,
    )
    with pytest.raises(wire.WireError):
        wire.decode_byte_chunks([plain, chunked[0]], kind="request")
