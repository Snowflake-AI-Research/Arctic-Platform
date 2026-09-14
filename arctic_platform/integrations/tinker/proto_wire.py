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
"""Protobuf codec for the two Tinker verbs that no longer speak JSON.

Modern SDKs post ``forward_backward`` as ``application/x-protobuf`` and refuse a
JSON reply for ``ForwardBackwardOutput`` and ``SampleResponse``::

    Server returned a JSON payload for {self.model_cls=}, which this SDK version
    only supports as proto. The server predates proto response serialization.

So the codec is not optional, and it is not symmetric: ``sample`` *requests*
are still JSON while its responses are proto.

The schema is the SDK's own generated ``tinker_public_pb2``, not a vendored
copy, so upstream changes surface as import or field errors instead of as
wrong bytes. The import is local to this module, leaving ``tinker`` optional
for anyone serving only the JSON verbs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

import numpy as np

if TYPE_CHECKING:
    from arctic_platform.integrations.tinker.router import ForwardBackwardRequest

__all__ = [
    "PROTO_CONTENT_TYPE",
    "decode_forward_backward_request",
    "encode_forward_backward_output",
    "encode_sample_response",
    "wants_proto",
]

PROTO_CONTENT_TYPE = "application/x-protobuf"


def _pb():
    try:
        from tinker.proto import tinker_public_pb2

        return tinker_public_pb2
    except ImportError as exc:  # pragma: no cover - exercised by the import guard test
        raise RuntimeError(
            "Serving Tinker's forward_backward or sample verbs needs the wire schema "
            "from the `tinker` SDK (pip install tinker). Those verbs are protobuf-only "
            "in current SDKs, so there is no JSON fallback to degrade to."
        ) from exc


def wants_proto(accept: str | None, content_type: str | None = None) -> bool:
    return PROTO_CONTENT_TYPE in ((accept or "") + " " + (content_type or ""))


# ── requests: proto → the router's own pydantic models ───────────────────────


def _tensor_to_lists(tensor: Any) -> dict[str, Any]:
    """One proto ``Tensor`` as ``TensorData`` kwargs.

    Only float32 and int64 appear on the public request wire; the SDK's encoder
    refuses anything else before it reaches us.
    """
    pb = _pb()
    dtype_name = {pb.DTYPE_FLOAT32: "float32", pb.DTYPE_INT64: "int64"}.get(tensor.dtype)
    if dtype_name is None:
        raise ValueError(f"unsupported request tensor dtype {tensor.dtype}")
    np_dtype = np.float32 if dtype_name == "float32" else np.int64

    if tensor.WhichOneof("encoding") == "sparse_csr":
        csr = tensor.sparse_csr
        return {
            "dtype": dtype_name,
            "data": np.frombuffer(csr.values, dtype=np_dtype).tolist(),
            "shape": list(tensor.shape) or None,
            "sparse_crow_indices": np.frombuffer(csr.crow_indices, dtype=np.int64).tolist(),
            "sparse_col_indices": np.frombuffer(csr.col_indices, dtype=np.int64).tolist(),
        }
    return {
        "dtype": dtype_name,
        "data": np.frombuffer(tensor.dense, dtype=np_dtype).tolist(),
        "shape": list(tensor.shape) or None,
    }


def decode_forward_backward_request(body: bytes) -> tuple[ForwardBackwardRequest, bool]:
    """``(request, forward_only)`` from a proto body.

    ``forward`` and ``forward_backward`` share one endpoint upstream, separated
    only by ``forward_only``; the caller routes on the returned flag.
    """
    from arctic_platform.integrations.tinker.router import Datum
    from arctic_platform.integrations.tinker.router import EncodedTextChunk
    from arctic_platform.integrations.tinker.router import ForwardBackwardInput
    from arctic_platform.integrations.tinker.router import ForwardBackwardRequest
    from arctic_platform.integrations.tinker.router import ModelInput
    from arctic_platform.integrations.tinker.router import TensorData

    pb = _pb()
    msg = pb.ForwardBackwardRequest()
    msg.ParseFromString(body)

    data = []
    for datum in msg.data:
        chunks = []
        for chunk in datum.model_input:
            kind = chunk.WhichOneof("chunk")
            if kind != "encoded_text":
                raise ValueError(
                    f"model input chunk {kind!r} is not supported; this layer handles "
                    "pre-tokenized text only"
                )
            # int32 here, unlike the int64 the rest of the request uses.
            tokens = np.frombuffer(chunk.encoded_text.tokens, dtype=np.int32).tolist()
            chunks.append(EncodedTextChunk(tokens=tokens))
        data.append(
            Datum(
                model_input=ModelInput(chunks=chunks),
                loss_fn_inputs={
                    name: TensorData(**_tensor_to_lists(tensor))
                    for name, tensor in datum.loss_fn_inputs.items()
                },
            )
        )

    request = ForwardBackwardRequest(
        model_id=msg.model_id,
        seq_id=msg.seq_id,
        forward_backward_input=ForwardBackwardInput(
            data=data,
            loss_fn=msg.loss_fn,
            loss_fn_config=dict(msg.loss_fn_config) or None,
        ),
    )
    return request, bool(msg.forward_only)


# ── responses: the router's json dicts → proto ───────────────────────────────


def _batched_tensor(per_datum: list[dict[str, Any]], field: str) -> Any:
    """Pack one ``loss_fn_outputs`` field across datums into a ``BatchedTensor``.

    ``offsets`` are *byte* offsets with ``len(datums) + 1`` entries, and
    ``trailing_shape`` is every dimension after the first: the reader recovers
    the leading dimension by division, so a ragged batch stays decodable.
    """
    pb = _pb()
    first = per_datum[0][field]
    dtype_name = first.get("dtype", "float32")
    np_dtype = np.float32 if dtype_name == "float32" else np.int64
    proto_dtype = pb.DTYPE_FLOAT32 if dtype_name == "float32" else pb.DTYPE_INT64
    trailing_shape = list(first.get("shape") or [])[1:]

    buffers, offsets = [], [0]
    for datum in per_datum:
        arr = np.asarray(datum[field].get("data") or [], dtype=np_dtype)
        buffers.append(arr.tobytes())
        offsets.append(offsets[-1] + arr.nbytes)

    return pb.BatchedTensor(
        data=b"".join(buffers),
        offsets=np.asarray(offsets, dtype=np.int64).tobytes(),
        dtype=proto_dtype,
        trailing_shape=trailing_shape,
    )


def encode_forward_backward_output(payload: dict[str, Any]) -> bytes:
    """Serialize the router's ``ForwardBackwardOutput`` dict."""
    pb = _pb()
    outputs = payload.get("loss_fn_outputs") or []
    type_tag = payload.get("loss_fn_output_type") or "TorchLossReturn"

    records = []
    # Only fields present on *every* datum: a BatchedTensor's offsets span the
    # whole batch, so one missing field shifts every later datum's slice.
    shared = set(outputs[0]) if outputs else set()
    for datum in outputs[1:]:
        shared &= set(datum)
    if outputs:
        records.append(
            pb.ArrayRecord(
                type_tag=type_tag,
                num_datums=len(outputs),
                fields={name: _batched_tensor(outputs, name) for name in sorted(shared)},
            )
        )

    return pb.ForwardBackwardOutput(
        loss_fn_output_type=type_tag,
        loss_fn_outputs=records,
        metrics={k: float(v) for k, v in (payload.get("metrics") or {}).items()},
    ).SerializeToString()


def encode_sample_response(payload: dict[str, Any]) -> bytes:
    """Serialize the router's ``SampleResponse`` dict."""
    pb = _pb()
    stop_reasons = {"stop": pb.STOP_REASON_STOP, "length": pb.STOP_REASON_LENGTH}

    sequences = []
    for seq in payload.get("sequences") or []:
        reason = seq.get("stop_reason")
        if reason not in stop_reasons:
            raise ValueError(f"unknown stop_reason {reason!r}; expected one of {sorted(stop_reasons)}")
        logprobs = seq.get("logprobs")
        sequences.append(
            pb.SampledSequence(
                # Tokens are int32 on the wire; log-probs float32.
                tokens=np.asarray(seq.get("tokens") or [], dtype=np.int32).tobytes(),
                logprobs=(
                    np.asarray(logprobs, dtype=np.float32).tobytes() if logprobs is not None else b""
                ),
                stop_reason=stop_reasons[reason],
            )
        )

    message = pb.SampleResponse(sequences=sequences)
    prompt_logprobs = payload.get("prompt_logprobs")
    if prompt_logprobs is not None:
        message.prompt_logprobs = np.asarray(prompt_logprobs, dtype=np.float32).tobytes()
    return message.SerializeToString()
