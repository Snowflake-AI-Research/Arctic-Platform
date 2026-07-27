# RL client unification notes

Goal: one `ArcticRLClient` frontend where **every backend accepts identical
per-op args/kwargs and returns identically-shaped responses**, so each transport
is a dumb forwarder of `Request(op, job_id, body)` with no per-op rewiring. The
on-prem transports (HTTP + Ray) already meet this bar; Cortex does not yet (see
"Cortex transport" below).

This package is the target design. These notes track what still diverges in the
three existing clients so the remaining convergence work is explicit. Most of the
hard blockers are **server-side** — the clients can only converge once the
servers accept the same contract.

## Clients being reconciled

1. **arctictraining** — `ArcticTraining-dss/arctic_training/arctic_rl/client.py`.
   One sync class; `backend ∈ {local, dss-platform, neutrino}`; branches
   internally on `self._is_neutrino` in nearly every method.
2. **NeutrinoClient** — `dss-client/dss_client/neutrino_client.py`. A low-level
   SnowAPI client (not an `ArcticRLClient`). Our `CortexTransport` now
   reimplements a trimmed subset of it.
3. **arctic_platform** — `arctic_platform/rl/{client,http_client,ray_client}.py`.
   `create_arctic_rl_client` picks `ArcticRLHTTPClient` or `ArcticRLRayClient`
   by `comm_protocol`; **all ops are `async def`**.

## Per-op API divergence

| op | arctictraining (sync) | arctic_platform (async) | this package |
|----|----|----|----|
| `fwd_bwd` | `(batch, processing=None, router_replay=None)` | `(batch, processing=None)` | `(batch, processing=None, router_replay=None)` |
| `fwd_no_grad` | `(batch)` | `(batch, reference_model)` | `(batch)` |
| `step` | `(learning_rate=None)` | `()` — no LR | `(learning_rate=None)` |
| `generate` | `(prompts, sampling_params=None, routing_key=None, strict=False)` | `(prompts, sampling_params=None)` | `(prompts, sampling_params=None, routing_key=None)` |
| `sync_weights` | `()` | `(cuda_ipc=False, low_memory=False)` | `()` |
| `save_checkpoint` | `(stage_info=None, path=None, checkpoint_type="resumable")` | `()` | `(stage_info=None, path=None)` |
| `reset_prefix_cache` | `(drain=True, timeout_s=60, retry_interval_s=0.1)` | `()` | `(drain=True, timeout_s=60)` |
| `log_probs` | `(prompts, completions=None, top_k=1)` | `(prompts, completions=None, top_k=1)` | `(prompts, completions=None, top_k=1)` |

## What's missing / divergent per client

### arctictraining `ArcticRLClient`
- Sync class that branches on `if self._is_neutrino` in every op — exactly the
  divergence this design removes (op logic single-sourced in `client.py`,
  backend behind a transport).
- Richest surface today: router-replay bootstrap/meta/attach, `generate_stream`
  + `get_request_status` + `cancel_request`, resume-from-checkpoint preflight,
  Neutrino experiment naming, save-checkpoint id resolution.
- Neutrino branch raises `NotImplementedError` for `fwd_no_grad`, `log_probs`,
  `save_weights` (disk reload) — same gaps as our `CortexTransport`.
- No colocation lifecycle ops; no CUDA-IPC weight sync.

### `NeutrinoClient` (dss-client)
- Not an `ArcticRLClient`: a transport-level SnowAPI client. Ops are named
  `forward_backward` / `step` / `save` / `generate` / `weight_sync` /
  `operation`, return **request ids the caller must `poll_request`**, and there
  is no unified op vocabulary or response shaping.
- No `fwd_no_grad` / `log_probs` / `save_weights`; no colocation ops.
- Has (unused by the minimal transport): chunked octet uploads, client-side
  prompt-length validation, capacity, log/event streaming, router-replay ops,
  checkpoint list/export, retry/backoff plumbing.
- Owns the client-side tokenization helpers (`build_forward_backward_kwargs`);
  the cortex fwd-bwd batch is prepared here, not in the frontend. (Our example
  reimplements a slice of this inline.)

### arctic_platform `ArcticRLClient` (rl/)
- **Async** ops (`async def fwd_bwd/step/generate/...`) vs sync cortex/dss — a
  fundamental calling-convention mismatch to reconcile.
- Split into two classes chosen by `comm_protocol`, with per-op signatures that
  drift from cortex/dss: `step()` has no `learning_rate`; `generate()` lacks
  `routing_key`/`strict`; `fwd_no_grad(batch, reference_model)` carries an extra
  routing flag; `sync_weights(cuda_ipc, low_memory)` + `colocate`;
  `save_checkpoint()` / `reset_prefix_cache()` take no args.
- **Wire codec divergence**: uses `torch.save` / `torch.load` (pickle) on the
  wire, while cortex/dss use the DSSST1 safetensors `wire` codec. Our
  `onprem_http` mirrors the pickle path (`_dumps`/`_loads`) to match today's
  server; it cannot switch to `wire` until the on-prem server does.
- Large colocation surface with no cortex/dss twin: `sleep_/wake_inference`,
  `sleep_/wake_training`, `sleep_/wake_log_prob`, `empty_training_cache`,
  `weight_norm`.
- Ray path: `ArcticRLRayServer` methods take a uniform `(job_id, body)` shape
  matching the HTTP `POST /{op}` surface, so `RayTransport._rpc` forwards
  `(job_id, body)` with no per-op binding.

## Server-side blockers (cannot be fixed in the client)
- **fwd_bwd batch contract**: cortex expects RPC-style `{args, kwargs}` tokenized
  server-side; on-prem expects pre-tokenized verl-GRPO `{batch, meta, processing}`.
- **Response schema**: cortex returns loss only; on-prem returns
  `{metrics: {grad_norm, ...}}` (scalars sometimes per-DP-rank lists).
- **One wire codec** (DSSST1 safetensors) across all servers so transports share
  serialization.
- **Sync vs async**: pick one convention for the frontend.

## Sharing already in place (this package)
- `Transport` ABC + `JobHandles` + `Request` (single op vocabulary in `client.py`).
- `OnPremTransport` base shared by `onprem_http` and `onprem_ray`.
- `JOB_CREATE_ORDER` + `ArcticRLClientConfig.gpus_for()` shared by cortex and
  on-prem job initialization.
- Cortex and on-prem no longer hand-roll GPU-gating / creation order separately.

## On-prem HTTP transport
`HttpTransport` shares the same `OnPremTransport` control plane and only supplies
the delivery primitives: it POSTs each op to `/{op}?job_id=...`. Tensor-bearing
ops (`fwd_bwd`/`fwd_no_grad`) send an octet `torch.save` payload and decode octet
responses; everything else is JSON. It also optionally launches a local server
(`launch_local_server`) and polls `/health` + `/job/{id}` to wait for readiness.
The Ray path forwards `(job_id, body)` to the same uniform server surface, so the
two transports differ only in `_start`/`_rpc`/`_destroy`.

## Cortex transport: the remaining non-trivial forwarder
`CortexTransport` is the one transport that still does per-op translation: each
op has a handler mapping the canonical `body` to a SnowAPI call (path, wire
framing, sub-job routing, submit→poll). It shares only the `Transport` ABC and
the `wire` codec with on-prem, because Cortex's async submit→poll model has no
counterpart in on-prem's synchronous request/response.

The eventual goal is to unify the Cortex and on-prem **server** APIs (op names,
argument shapes, response schemas) so `CortexTransport` can collapse to the same
dumb `_rpc(request)` as the on-prem transports — forwarding just `(job_id, body)`
with no per-method translation. Until the servers converge on the blockers above,
Cortex keeps its per-op handlers.

## Not yet in unified client (future work)
Ops present in `arctic_platform/rl/*_client.py` but not yet on `ArcticRLClient`.
Not on the immediate critical path, but they **block deleting the async
clients**, so parity here is a prerequisite for retiring `rl/*_client.py`:
- `sleep_inference` / `wake_inference`
- `sleep_training` / `wake_training`
- `sleep_log_prob` / `wake_log_prob`
- `empty_training_cache`
- `weight_norm`
- `save_weights` (disk-based weight reload): a weight-sync variant that reloads
  the sampling engine from a checkpoint on disk. Deliberately omitted for now —
  server-side reload is unimplemented on-prem (the old ray client raised
  `NotImplementedError`; the http client was a warn-on-error stub). Add back once
  a backend actually supports it.
