# Copyright 2025 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
"""Fake Cortex GS — a plumbing-only stand-in for the SnowAPI Cortex-training
endpoint.

The point of this server is to let us drive verl and SkyRL end-to-end against
the ``CortexTransport`` without a real Neutrino GS deployment. It speaks every
route ``arctic_platform.client.transports.cortex.CortexTransport`` actually
hits, decodes DSSST1 chunked requests correctly, and returns shape-plausible
canned responses. Losses are random; convergence validation still needs a real
server.

Not a spec for the Cortex server team — just a running executable of what the
client currently sends. Handy as a starting point when Neutrino GS writes their
own tests against the same client.

Endpoints implemented:

    POST   {prefix}                              → CreateJob → {"job_id"}
    GET    {prefix}/{job_id}                     → GetJob → running + sub_jobs
    POST   {prefix}/{job_id}:cancel              → {}
    POST   {prefix}/{job_id}/forward-backward    → octet chunks → request_id
    POST   {prefix}/{job_id}/forward-no-grad     → octet chunks → request_id
    POST   {prefix}/{job_id}/generate            → octet chunks → request_id
    POST   {prefix}/{job_id}/step                → JSON        → request_id
    POST   {prefix}/{job_id}/save                → JSON        → request_id
    POST   {prefix}/{job_id}/log-probs           → JSON        → request_id
    POST   {prefix}/{job_id}/operation           → JSON        → request_id
        (operation_type ∈ {weight-sync, reset-prefix-cache})
    GET    {prefix}/{job_id}/requests/{request_id} → completed + result

Prefix layout mirrors what the client builds:

    /api/v2/databases/{database}/schemas/{schema}/{endpoint}

Any (database, schema, endpoint) triple is accepted so a test / launcher can
pick whatever it wants.

Run standalone (default port 8080):

    python -m tests.e2e.fake_cortex_gs [--port 8080] [--host 0.0.0.0]

or from a test via the ``fake_cortex_gs_process`` fixture.
"""

from __future__ import annotations

import argparse
import base64
import logging
import random
import threading
import uuid
from typing import Any

import torch
import uvicorn
from fastapi import Body
from fastapi import FastAPI
from fastapi import Request
from fastapi.responses import JSONResponse

from arctic_platform.client import wire

log = logging.getLogger("fake_cortex_gs")


# ─── State ────────────────────────────────────────────────────────────────


class _State:
    """Ephemeral in-memory state for one running mock instance.

    Per-job: a dict of pending octet request chunks (keyed by
    (job_id, path_suffix)) and a request registry (request_id -> canned result).
    """

    def __init__(self) -> None:
        self.jobs: dict[str, dict] = {}
        # (job_id, path_suffix, chunk_group_id) -> list[bytes] as chunks arrive
        self.chunk_buffers: dict[tuple[str, str, str], list[bytes]] = {}
        # request_id -> canned response dict (result already shaped for the client)
        self.requests: dict[str, dict] = {}


STATE = _State()


# ─── Prefix routing ───────────────────────────────────────────────────────

# The client's prefix is /api/v2/databases/<db>/schemas/<sch>/<endpoint>.
# All routes below mount under that variable prefix.
_PREFIX = "/api/v2/databases/{database}/schemas/{schema}/{endpoint}"


def _make_app() -> FastAPI:
    app = FastAPI(title="fake_cortex_gs", version="0.1.0")

    # ── CreateJob ──────────────────────────────────────────────────────────
    @app.post(_PREFIX)
    async def create_job(database: str, schema: str, endpoint: str, body: dict = Body(...)) -> dict:
        job_id = f"fake-job-{uuid.uuid4().hex[:8]}"
        sub_job_configs = body.get("sub_job_configs") or []
        sub_jobs = []
        for cfg in sub_job_configs:
            jt = cfg.get("job_type", "training")
            sub_jobs.append({"job_type": f"job_type_{jt}", "sub_job_id": f"{job_id}:{jt}"})
        STATE.jobs[job_id] = {"sub_jobs": sub_jobs, "status": "job_state_running"}
        log.info("fake_cortex_gs: created job %s with sub_jobs=%s", job_id, [s["job_type"] for s in sub_jobs])
        return {"job_id": job_id}

    # ── GetJob (client polls this immediately after CreateJob) ─────────────
    #
    # Return shape follows what `CortexTransport._wait_for_job` reads:
    # top-level `status` = "job_state_running", and `_capture_sub_jobs`
    # reads `job_info.get("job", job_info)["sub_jobs"]` (accepts either).
    @app.get(_PREFIX + "/{job_id}")
    async def get_job(database: str, schema: str, endpoint: str, job_id: str) -> dict:
        job = STATE.jobs.get(job_id)
        if job is None:
            return JSONResponse({"error": "job not found"}, status_code=404)
        return {"status": job["status"], "sub_jobs": job["sub_jobs"]}

    # ── Cancel ─────────────────────────────────────────────────────────────
    @app.post(_PREFIX + "/{job_id}:cancel")
    async def cancel_job(database: str, schema: str, endpoint: str, job_id: str) -> dict:
        STATE.jobs.pop(job_id, None)
        return {}

    # ── Octet-chunked ops: forward-backward / forward-no-grad / generate ───
    for path_suffix in ("forward-backward", "forward-no-grad", "generate"):
        _register_octet_route(app, path_suffix)

    # ── JSON ops: step / save / log-probs ──────────────────────────────────
    @app.post(_PREFIX + "/{job_id}/step")
    async def step(database: str, schema: str, endpoint: str, job_id: str, body: dict = Body(...)) -> dict:
        req_id = _register_result(_fake_step_result(learning_rate=body.get("learning_rate")))
        return {"request_id": req_id}

    @app.post(_PREFIX + "/{job_id}/save")
    async def save(database: str, schema: str, endpoint: str, job_id: str, body: dict = Body(...)) -> dict:
        req_id = _register_result({"path": f"/fake/checkpoints/{uuid.uuid4().hex[:8]}"})
        return {"request_id": req_id}

    @app.post(_PREFIX + "/{job_id}/log-probs")
    async def log_probs(database: str, schema: str, endpoint: str, job_id: str, body: dict = Body(...)) -> dict:
        prompts = body.get("prompts") or []
        completions = body.get("completions") or []
        # Return a wire-encoded logprobs tensor shaped [len(prompts), <resp_len>].
        n_rows = max(1, len(prompts))
        seq = _completion_len_hint(completions) or 8
        result = _fake_log_probs_result(n_rows=n_rows, seq_len=seq)
        req_id = _register_result(result, wire_encoded=True)
        return {"request_id": req_id}

    # ── operation: weight-sync / reset-prefix-cache ────────────────────────
    @app.post(_PREFIX + "/{job_id}/operation")
    async def operation(
        database: str, schema: str, endpoint: str, job_id: str, body: dict = Body(...)
    ) -> dict:
        op_type = body.get("operation_type")
        if op_type in {"weight-sync", "reset-prefix-cache"}:
            req_id = _register_result({})
            return {"request_id": req_id}
        return JSONResponse({"error": f"unknown operation_type {op_type!r}"}, status_code=400)

    # ── Request status (client polls this until state=completed) ──────────
    @app.get(_PREFIX + "/{job_id}/requests/{request_id}")
    async def request_status(
        database: str, schema: str, endpoint: str, job_id: str, request_id: str
    ) -> dict:
        canned = STATE.requests.get(request_id)
        if canned is None:
            return JSONResponse({"error": "request not found"}, status_code=404)
        return {
            "status": "request_state_completed",
            "events": canned.get("events", []),
            "result": canned.get("result", {}),
        }

    return app


def _register_octet_route(app: FastAPI, path_suffix: str) -> None:
    """Wire up one of the DSSST1 octet-chunked routes.

    All three (fwd_bwd / fwd_no_grad / generate) share the same chunk-reassembly
    logic; response shape is the only difference.
    """

    async def handler(
        database: str, schema: str, endpoint: str, job_id: str, request: Request
    ) -> dict:
        raw = await request.body()
        desc = wire.read_byte_chunk_metadata(raw) or {"total_chunks": 1, "chunk_idx": 0}
        group_id = desc.get("chunk_group_id") or "single"
        key = (job_id, path_suffix, group_id)
        buf = STATE.chunk_buffers.setdefault(key, [])
        buf.append(raw)
        total = int(desc.get("total_chunks", 1))
        if len(buf) < total:
            # Intermediate chunk: return empty body (no request_id yet), per
            # `_post_octet_request_chunks` in the client.
            return {}

        # Final chunk. Reassemble and inspect the payload shape.
        try:
            frame = wire.decode_byte_chunks(buf) if total > 1 else buf[0]
            decoded = wire.loads(frame)
        finally:
            STATE.chunk_buffers.pop(key, None)

        result = _shape_octet_response(path_suffix, decoded)
        req_id = _register_result(result, wire_encoded=(path_suffix != "forward-backward"))
        return {"request_id": req_id}

    # FastAPI can't share a handler between routes with the same path template;
    # register under the concrete suffix.
    app.add_api_route(
        _PREFIX + f"/{{job_id}}/{path_suffix}",
        handler,
        methods=["POST"],
        name=f"octet_{path_suffix.replace('-', '_')}",
    )


# ─── Fake response shaping ────────────────────────────────────────────────


def _register_result(result: dict, *, wire_encoded: bool = False) -> str:
    """Register a canned response for a synthesized request_id.

    - `wire_encoded=False`: return `result` inline in the request-status JSON.
    - `wire_encoded=True`:  encode via DSSST1 + base64 and put it in the
      request-status `result` field, per `_decode_result_payload` in the client.
    """
    req_id = f"fake-req-{uuid.uuid4().hex[:8]}"
    if wire_encoded:
        frame = wire.dumps(result)
        STATE.requests[req_id] = {
            "result": {
                "wire_format": wire.WIRE_FORMAT_VERSION,
                "encoding": "base64",
                "payload_b64": base64.b64encode(frame).decode("ascii"),
            }
        }
    else:
        STATE.requests[req_id] = {"result": result}
    return req_id


def _shape_octet_response(path_suffix: str, decoded: Any) -> dict:
    """Produce a response for one of the octet-chunked routes based on the
    decoded request payload's shape. Correctness is not the point — we just
    need a shape the client's response shim + verl/SkyRL adapters can parse.
    """
    body = decoded if isinstance(decoded, dict) else {}
    kwargs = body.get("kwargs") or {}
    input_ids = kwargs.get("input_ids")
    if torch.is_tensor(input_ids):
        n_rows, seq_len = int(input_ids.shape[0]), int(input_ids.shape[1])
    else:
        n_rows, seq_len = 1, 8

    if path_suffix == "forward-backward":
        # verl/SkyRL read `.get("loss")` / `.get("grad_norm")`; keep it JSON.
        return _fake_step_result()
    if path_suffix == "forward-no-grad":
        return _fake_log_probs_result(n_rows=n_rows, seq_len=seq_len)
    if path_suffix == "generate":
        # `generate` body ships `{"prompts": [...]}` (no input_ids), so
        # size the fake results off the prompts list.
        prompts = body.get("prompts") or []
        return _fake_generate_result(n_prompts=max(1, len(prompts)))
    return {}


def _fake_step_result(*, learning_rate: float | None = None) -> dict:
    loss = round(random.uniform(0.1, 1.5), 4)
    return {
        "loss": loss,
        "avg_loss": loss,
        "metrics": {
            "loss": loss,
            "grad_norm": round(random.uniform(0.1, 2.0), 4),
            "ppo_kl": round(random.uniform(0.0, 0.05), 4),
            "pg_loss": loss,
            "pg_clipfrac_lower": 0.0,
            "kl_loss": 0.001,
            "kl_coef": 0.001,
            "last_lr": learning_rate or 1e-5,
        },
    }


def _fake_log_probs_result(*, n_rows: int, seq_len: int) -> dict:
    """Return a DSSST1-encodable dict with `model_outputs.logprobs` of the
    right shape. `CortexTransport._shape_train_response` aliases
    `model_outputs` -> `batch`, and verl's `_send_compute_log_prob` then
    reads `response["batch"]["log_probs"]` after renaming `logprobs` ->
    `log_probs`.
    """
    logprobs = -torch.rand(n_rows, seq_len).abs()  # sign-correct: log-probs ≤ 0
    entropy = torch.rand(n_rows, seq_len).abs()
    return {
        "model_outputs": {
            "logprobs": logprobs,
            "entropy": entropy,
        },
        "metrics": {},
    }


def _fake_generate_result(*, n_prompts: int) -> dict:
    # `_generate` in the client returns `{"results": [...]}`; each entry
    # ships prompt / response ids + text. Keep it minimal.
    results = []
    for _ in range(max(1, n_prompts)):
        n_toks = random.randint(4, 12)
        results.append(
            {
                "response_ids": [random.randint(1000, 50000) for _ in range(n_toks)],
                "response_text": "fake",
                "logprobs": [-abs(random.random()) for _ in range(n_toks)],
                "finish_reason": "stop",
            }
        )
    return {"results": results}


def _completion_len_hint(completions: list) -> int | None:
    if not completions:
        return None
    first = completions[0]
    if isinstance(first, list):
        return len(first)
    if isinstance(first, str):
        return max(1, len(first.split()))
    return None


# ─── Fixture-friendly runner ──────────────────────────────────────────────


def serve_in_background(host: str = "127.0.0.1", port: int = 8080) -> threading.Thread:
    """Start the fake GS in a daemon thread. Returns the thread so tests can
    join or drop it. Uvicorn's built-in threading integration is used so we
    don't need pytest-anyio.
    """
    app = _make_app()
    config = uvicorn.Config(app, host=host, port=port, log_level="warning", access_log=False)
    server = uvicorn.Server(config)

    def _run() -> None:
        server.run()

    thread = threading.Thread(target=_run, daemon=True, name=f"fake_cortex_gs:{port}")
    thread.start()
    # Wait for uvicorn to bind before returning.
    import time

    for _ in range(200):
        if server.started:
            return thread
        time.sleep(0.02)
    raise RuntimeError("fake_cortex_gs failed to start within 4s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    log.info("fake_cortex_gs listening on http://%s:%d", args.host, args.port)
    uvicorn.run(_make_app(), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
