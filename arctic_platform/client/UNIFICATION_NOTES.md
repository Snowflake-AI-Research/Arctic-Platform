# RL client unification notes

Goal: one `ArcticRLClient` frontend where **every backend accepts identical
per-op args/kwargs and returns identically-shaped responses**, so each transport
is a dumb forwarder of `Request(op, job_id, body)` with no per-op rewiring.

This package is the target design. The op *surface* (names, args, canonical
body, which job each op targets, response contract) is defined once in
`client.py`; a transport owns only job identity + wire mechanics.

## Design in place (this package)
- `Transport` ABC + `JobHandles` + `Request` (single op vocabulary in `client.py`).
- `OnPremTransport` base: job creation, ordering, payload building, and a uniform
  `call` that posts/dispatches the op against its target job. Concrete transports
  implement only the delivery primitives (`_start`, `_rpc`, `_destroy`,
  `_wait_running`).
- `JOB_CREATE_ORDER` + `ArcticRLClientConfig.gpus_for()` centralize GPU-gating and
  creation order so transports no longer hand-roll them.

## On-prem Ray transport
`RayTransport` makes in-process Ray actor calls — no HTTP, no serialization. The
server splits into a `state` actor (job creation) and an `ArcticRLRayServer`
wrapper (typed async ops) that snapshots workers at construction, so the wrapper
is built lazily after jobs are initialized. `_rpc` resolves `op -> method` and
forwards `(job_id, body)` unchanged, matching the server's uniform
`op(job_id, body) -> dict` surface.
