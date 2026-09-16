# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""PR 95 Cortex ``fwd_no_grad``: JSON ``/operation`` + Cortex RPC envelope.

Workspace-only monkeypatch. Do not edit ``arctic_trl``.

SnowAPI has no octet ``/{job}/forward`` (404 here). PR 95 posts
``operation_type=forward`` with a base64 DSSST1 frame on ``/{job}/operation``.
The zone ``wire.loads`` one complete DSSST1 per POST; do not byte-slice a
frame or reuse the fwd-bwd octet chunker.
"""

from __future__ import annotations

import base64

_OPERATION_MAX_JSON_BYTES = 16 * 1024 * 1024
_OPERATION_ENVELOPE_OVERHEAD_BYTES = 4096
# Spec: /operation forward is not request-chunked; decoded DSSST1 over 60 MiB is rejected.
_OPERATION_MAX_RAW_FRAME_BYTES = 60 * 1024 * 1024


def _max_forward_frame_bytes() -> int:
    """Largest DSSST1 that still fits in one JSON /operation envelope (base64)."""
    json_cap = max(1, (_OPERATION_MAX_JSON_BYTES - _OPERATION_ENVELOPE_OVERHEAD_BYTES) * 3 // 4)
    return min(json_cap, _OPERATION_MAX_RAW_FRAME_BYTES)


def _is_batch_tensor(obj: object) -> bool:
    import torch

    return torch.is_tensor(obj) and obj.ndim >= 1


def _batch_size(body: dict) -> int:
    kwargs = body.get("kwargs") or {}
    ids = kwargs.get("input_ids")
    if _is_batch_tensor(ids):
        return int(ids.shape[0])
    ctx = body.get("context") or {}
    ids = ctx.get("input_ids")
    if _is_batch_tensor(ids):
        return int(ids.shape[0])
    return 1


def _slice_batch_value(obj: object, start: int, end: int, n: int) -> object:
    import torch

    if torch.is_tensor(obj) and obj.ndim >= 1 and obj.shape[0] == n:
        return obj[start:end]
    if isinstance(obj, dict):
        return {k: _slice_batch_value(v, start, end, n) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) == n:
            return obj[start:end]
        return [_slice_batch_value(v, start, end, n) for v in obj]
    if isinstance(obj, tuple):
        if len(obj) == n:
            return obj[start:end]
        return tuple(_slice_batch_value(v, start, end, n) for v in obj)
    return obj


def _slice_forward_body(body: dict, start: int, end: int) -> dict:
    n = _batch_size(body)
    out = dict(body)
    if "kwargs" in body:
        out["kwargs"] = _slice_batch_value(body["kwargs"], start, end, n)
    if "context" in body:
        out["context"] = _slice_batch_value(body["context"], start, end, n)
    return out


def _dump_forward_frame(body: dict) -> bytes:
    from arctic_platform import wire

    return wire.dumps(body, metadata={"response_options": {"format": "dssst1", "delivery": "chunked"}})


def _split_forward_bodies(body: dict) -> list[dict]:
    """Row-split so each DSSST1 fits one /operation JSON post.

    SnowAPI ``forward`` is not octet request-chunked. Byte-slicing a safetensors
    frame makes the zone ``wire.loads`` a truncated header (EOF / invalid JSON).
    """
    max_raw = _max_forward_frame_bytes()
    n = _batch_size(body)
    if n <= 1:
        frame = _dump_forward_frame(body)
        if len(frame) <= max_raw:
            return [body]
        raise RuntimeError(
            f"cortex /forward DSSST1 is {len(frame)} bytes but a single row cannot be split "
            f"(cap {max_raw} bytes)"
        )

    probe = _dump_forward_frame(_slice_forward_body(body, 0, 1))
    if len(probe) > max_raw:
        raise RuntimeError(
            f"cortex /forward single-row DSSST1 is {len(probe)} bytes (cap {max_raw} bytes)"
        )
    rows_hint = max(1, min(n, max_raw // max(len(probe), 1)))
    if rows_hint >= n:
        full = _dump_forward_frame(body)
        if len(full) <= max_raw:
            return [body]
        rows_hint = max(1, n // 2)

    slices: list[dict] = []
    start = 0
    while start < n:
        width = min(rows_hint, n - start)
        while True:
            piece = _slice_forward_body(body, start, start + width)
            frame = _dump_forward_frame(piece)
            if len(frame) <= max_raw:
                slices.append(piece)
                start += width
                rows_hint = width
                break
            if width == 1:
                raise RuntimeError(
                    f"cortex /forward row {start} DSSST1 is {len(frame)} bytes (cap {max_raw})"
                )
            width = max(1, width // 2)
    print(
        f"[parity] cortex /forward split {n} rows into {len(slices)} /operation posts (cap={max_raw})",
        flush=True,
    )
    return slices


def _concat_forward_results(parts: list[dict]) -> dict:
    import torch

    if not parts:
        raise RuntimeError("cortex /forward split produced no results")
    if len(parts) == 1:
        return parts[0]

    def _cat(key: str):
        tensors = []
        for part in parts:
            batch = part.get("batch") if isinstance(part, dict) else None
            if not isinstance(batch, dict) or key not in batch:
                raise RuntimeError(f"cortex /forward split result missing batch.{key}")
            tensors.append(batch[key])
        if all(torch.is_tensor(t) for t in tensors):
            return torch.cat(tensors, dim=0)
        return tensors

    first = parts[0]
    batch = dict(first.get("batch") or {})
    if "logprobs" in batch or "log_probs" in batch:
        key = "logprobs" if "logprobs" in batch else "log_probs"
        batch[key] = _cat(key)
        if "logprobs" in batch and "log_probs" in batch:
            batch["log_probs"] = batch["logprobs"]
    if "entropy" in batch or "entropies" in batch:
        key = "entropy" if "entropy" in batch else "entropies"
        batch[key] = _cat(key)
    out = dict(first)
    out["batch"] = batch
    return out


def to_cortex_fwd_payload(batch: dict) -> dict:
    """On-prem ``{batch, meta, processing}`` → Cortex ``{args, kwargs, context, processing}``.

    Cortex zones register ``compute_logprobs`` only. ``apply_temperature`` /
    ``compute_entropy_and_logprobs`` are rejected (PR 100). Recipe is T=1 so
    skipping apply_temperature is correct.
    """
    if "batch" in batch and isinstance(batch["batch"], dict):
        tensors = dict(batch["batch"])
    else:
        tensors = dict(batch)
    input_ids = tensors.get("input_ids")
    attention_mask = tensors.get("attention_mask")
    if input_ids is None or attention_mask is None:
        raise ValueError("cortex /forward requires input_ids and attention_mask")
    forwarded = {"input_ids": input_ids}
    if "position_ids" in tensors:
        forwarded["position_ids"] = tensors["position_ids"]
    from arctic_platform.integrations.verl.cortex_payload import _left_align

    forwarded, _, attention_mask = _left_align(forwarded, attention_mask, {})
    input_ids = forwarded["input_ids"]
    kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
    if "position_ids" in forwarded:
        kwargs["position_ids"] = forwarded["position_ids"]
    return {
        "args": (),
        "kwargs": kwargs,
        "context": {"input_ids": input_ids},
        "processing": {"post": ["compute_logprobs"], "loss_fn": None},
    }


def normalize_forward_result(result: dict) -> dict:
    """Zone ``{job_id, logprobs}`` or ``payload_b64`` → on-prem ``{batch: {logprobs, entropy}}``."""
    import torch
    from arctic_platform import wire

    if isinstance(result, dict) and isinstance(result.get("payload_b64"), str) and result.get("wire_format") is None:
        result = wire.loads(base64.b64decode(result["payload_b64"]))
    if not isinstance(result, dict):
        return result
    batch = result.get("batch")
    if isinstance(batch, dict) and ("logprobs" in batch or "log_probs" in batch):
        if "logprobs" not in batch and "log_probs" in batch:
            batch["logprobs"] = batch["log_probs"]
        if "entropy" not in batch and "entropies" in batch:
            batch["entropy"] = batch["entropies"]
        if "entropy" not in batch:
            lp = batch["logprobs"]
            batch["entropy"] = torch.zeros_like(lp) if torch.is_tensor(lp) else 0
        return result
    lp = result.get("logprobs", result.get("log_probs"))
    if lp is None:
        return result
    ent = result.get("entropy", result.get("entropies"))
    if ent is None:
        ent = torch.zeros_like(lp) if torch.is_tensor(lp) else 0
    return {"batch": {"logprobs": lp, "entropy": ent}}


def _envelope_for_frame(request, frame: bytes) -> dict:
    max_raw = _max_forward_frame_bytes()
    if len(frame) > max_raw:
        raise RuntimeError(
            f"cortex /forward DSSST1 is {len(frame)} bytes; cap is {max_raw} "
            f"(JSON /operation is not octet request-chunked)"
        )
    envelope = {
        "operation_type": "forward",
        "payload": {
            "content_type": "application/octet-stream",
            "payload_b64": base64.b64encode(frame).decode("ascii"),
        },
    }
    if request.job_id is not None:
        envelope["sub_job_id"] = str(request.job_id)
    return envelope


def _forward_operation_bodies(transport, request) -> tuple[str, list[list[dict]]]:
    """One JSON envelope group per row-split (each group is a complete DSSST1)."""
    body = {k: v for k, v in request.body.items() if v is not None}
    url = f"{transport._prefix}/{transport.job_id}/operation"
    groups = []
    for piece in _split_forward_bodies(body):
        groups.append([_envelope_for_frame(request, _dump_forward_frame(piece))])
    return url, groups


def _forward_request_id(final: dict, posted: int) -> str:
    request_id = final.get("request_id")
    if request_id is None:
        raise RuntimeError(
            f"cortex forward staged {posted} chunk(s) but the last response carried no request_id: {final}"
        )
    return str(request_id)


_PATCHED = False


def patch_cortex_transport() -> None:
    """Wake/sleep no-ops, inline ``/operation``, and PR 95 ``operation_type=forward``."""
    global _PATCHED
    if _PATCHED:
        return
    import arctic_platform.client.transports.cortex as cx
    from arctic_platform.client.transports.cortex import CortexTransport

    # Do not add "forward" to _OCTET_OPS: POST /{job}/forward is 404 on this GS.
    if hasattr(cx, "_WIRE_OPERATION"):
        cx._WIRE_OPERATION = {**cx._WIRE_OPERATION, "forward": "forward"}

    noop = {"wake-inference", "sleep-inference", "wake-training", "sleep-training"}
    inline_ops = {
        "bootstrap-router-replay",
        "cancel-request",
        "reset-prefix-cache",
        "router-replay-discard",
        "tail-logs",
    }
    _call, _acall = CortexTransport.call, CortexTransport.acall
    _poll, _apoll = CortexTransport._poll, CortexTransport._apoll

    def _submitted(op, body, response):
        request_id = response.get("request_id")
        if request_id is not None:
            return str(request_id)
        operation_type = body.get("operation_type")
        if operation_type in inline_ops:
            return response
        raise RuntimeError(f"cortex {operation_type or op} response carried no request_id: {response}")

    def _submit_forward(self, request):
        url, groups = _forward_operation_bodies(self, request)
        request_ids = []
        for envelopes in groups:
            final = {}
            for envelope in envelopes:
                final = self._send("POST", url, json=envelope)
            request_ids.append(_forward_request_id(final, len(envelopes)))
        return request_ids

    async def _asubmit_forward(self, request):
        url, groups = _forward_operation_bodies(self, request)
        request_ids = []
        for envelopes in groups:
            final = {}
            for envelope in envelopes:
                final = await self._asend("POST", url, json=envelope)
            request_ids.append(_forward_request_id(final, len(envelopes)))
        return request_ids

    def call(self, request):
        if request.op in noop:
            return {}
        if request.op == "forward":
            parts = [
                normalize_forward_result(_poll(self, request_id))
                for request_id in _submit_forward(self, request)
            ]
            return _concat_forward_results(parts)
        if request.op in ("step", "save", "operation"):
            body = {k: v for k, v in request.body.items() if v is not None}
            url = f"{self._prefix}/{self.job_id}/{request.op}"
            submitted = _submitted(request.op, body, self._send("POST", url, json=body))
            result = submitted if isinstance(submitted, dict) else _poll(self, submitted)
            return result
        return _call(self, request)

    async def acall(self, request):
        if request.op in noop:
            return {}
        if request.op == "forward":
            parts = [
                normalize_forward_result(await _apoll(self, request_id))
                for request_id in await _asubmit_forward(self, request)
            ]
            return _concat_forward_results(parts)
        if request.op in ("step", "save", "operation"):
            body = {k: v for k, v in request.body.items() if v is not None}
            url = f"{self._prefix}/{self.job_id}/{request.op}"
            submitted = _submitted(request.op, body, await self._asend("POST", url, json=body))
            result = submitted if isinstance(submitted, dict) else await _apoll(self, submitted)
            return result
        return await _acall(self, request)

    CortexTransport.call = call
    CortexTransport.acall = acall
    _PATCHED = True
    print("[parity] patched CortexTransport: wake/sleep no-ops + inline /operation + forward via /operation", flush=True)
