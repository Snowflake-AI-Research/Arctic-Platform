# Client unification notes

Goal: one client frontend where **every backend accepts identical per-op
args/kwargs and returns identically-shaped responses**, so each transport is a
dumb forwarder of `Request(op, job_id, body)` with no per-op rewiring.

This package is the target design. The op *surface* (names, args, canonical
body, which job each op targets, response contract) is defined once in
`client.py`; a transport owns only job identity + wire mechanics.

## One client, thin workload subclasses
SFT and RL share a single sync base:

- `ArcticClient` — every shared op. Ops route through `_call`, which also carries
  the (env-gated) server-profile echo.
- `ArcticSFTClient(ArcticClient)` — defaults `processing={"loss_fn": "sft"}` and
  `meta` onto the two forward bodies. Nothing else is SFT-specific.
- `SyncArcticRLClient(ArcticClient)` — adds `log_probs`, which needs a log-prob
  engine that SFT never allocates.
- `ArcticRLClient` — the async RL twin. It does not inherit from `ArcticClient`
  (sync/async method bodies cannot be shared) but reuses the same module-level
  `_*_request` builders, so the op vocabulary is still defined exactly once.

Keep new ops on `ArcticClient` unless they genuinely require a job type or a
data contract the other workload does not have.

## Design in place (this package)
- `Transport` ABC + `JobHandles` + `Request` (single op vocabulary in `client.py`).
- `OnPremTransport` base: job creation, ordering, payload building. Concrete
  transports implement only the delivery primitives (`_start`, `call`, `_destroy`,
  `_wait_running`); `call` posts/dispatches the op against its target job.
- `JOB_CREATE_ORDER` + `ArcticClientConfig.gpus_for()` centralize GPU-gating and
  creation order so transports no longer hand-roll them.

## Config nesting (canonical)
One shared shape for every backend *and* every workload — engine knobs are never
duplicated under backend-specific aliases. `ArcticRLClientConfig` remains as a
back-compat alias; the flat, on-prem-only `ArcticSFTClientConfig` is gone:

```
ArcticClientConfig
├── model_name, dtype, max_seq_len      # shared identity / length
├── training_gpus / sampling_gpus / ... # allocation only
├── training: TrainingConfig
│   ├── model: ModelBuildConfig         # ModelSpec minus path (factory)
│   │   ├── loader, attn_implementation
│   │   ├── parallelism, patches
│   ├── optimizer, train_batch_size, …
└── sampling.vllm: dict                 # vLLM engine kwargs only
└── backend_config: OnPrem | Cortex     # connection / deploy only
```

`ArcticClientConfig.model_spec()` assembles a full `arctic_platform.model.ModelSpec`
from `model_name` + `training.model` for `build_model(...)`.

### Temporary wire adapters — delete after server alignment
`ArcticClientConfig.to_onprem(job_type)` and `.to_cortex()` translate the
canonical shape into today's on-prem `/initialize` and Cortex `sub_job_configs`
wires:

- nest engine kwargs under `inference_config.vllm_config` (Neutrino)
- map `training.model` → Neutrino `model_provider` / `ep_size` / `attn_*`
- map `training.model` → on-prem `ds_worker_config` (`use_liger`, attn, …)

**These adapters should go away** once both servers accept the canonical nesting
(and call `build_model(ModelSpec)`) directly — do not grow new translation
logic here; change the server instead.

## On-prem Ray transport
`RayTransport` makes in-process Ray actor calls — no HTTP, no serialization. The
server splits into a `state` actor (job creation) and an `ArcticRLRayServer`
wrapper (typed async ops) that snapshots workers at construction, so the wrapper
is built lazily after jobs are initialized. `call` resolves `op -> method` and
forwards `(job_id, body)` unchanged, matching the server's uniform
`op(job_id, body) -> dict` surface.

## On-prem HTTP transport
`HttpTransport` shares the same `OnPremTransport` control plane and only supplies
the delivery primitives: it POSTs each op to `/{op}?job_id=...`. The canonical op
names + arg/response shapes mirror Cortex (`forward-backward`, `forward`, `step`,
`save`, `generate`, plus on-prem-only `log-probs`). Control-plane ops (weight-sync,
reset-prefix-cache) share one canonical `operation` op carrying Cortex's
`{operation_type, payload}` envelope, delivered to `/operation`. Tensor-bearing
ops (`forward-backward`/`forward`) plus `generate` send an octet **DSSST1** payload
and decode octet DSSST1 responses (generate carries no tensors, but the endpoint
speaks the binary wire to match Cortex); everything else is JSON. DSSST1 is the same
`arctic_platform.wire` codec Cortex speaks to SnowAPI, so both transports (and the
on-prem server) share one binary wire and never touch pickle/`torch.load`. It also
optionally launches a local server
(`launch_local_server`) and polls `/health` + `/job/{id}` to wait for readiness.
The Ray path forwards `(job_id, body)` to the same uniform server surface, so the
two transports differ only in `_start`/`call`/`_destroy`.

## Known divergences from SnowAPI (deferred on purpose)
The unified client aligns the **request bodies + op vocabulary** with Cortex/SnowAPI
(the `/operation` envelope, `weight-sync` source/target sub-job ids, DSSST1 tensor
bodies). The on-prem **server** intentionally does *not* reproduce the rest of
SnowAPI's shape — the transport layer is the adapter, so the server stays simple.

- **Sync vs async.** SnowAPI is submit→`{request_id}`→poll
  `GET /{job_id}/requests/{request_id}` (status enums, `result_chunk` events,
  `next_cursor`). The on-prem server returns results **inline** (octet DSSST1 for
  tensor ops, JSON otherwise). The Cortex transport owns the submit+poll loop;
  on-prem has no `request_id`/poll surface. Deferred.
- **Data-plane ops stay dedicated.** `/operation` is reserved for control-plane
  ops (`weight-sync`, `reset-prefix-cache`). We deliberately do **not** mirror
  SnowAPI's `operation_type:"forward"` routing — `forward-backward`/`forward`/
  `generate`/`log-probs` remain dedicated endpoints (this also matches how the
  neutrino client posts `forward-backward`/`generate`). Any parity is the
  transport's job.
- **Job creation.** On-prem `/initialize` takes one `JobConfig` per call;
  SnowAPI takes a single `sub_job_configs[]` (parent + sub-jobs,
  `training_config`/`inference_config`). Not unified. Deferred.
- **Addressing / id types.** On-prem uses `job_id` as a query param with `int`
  ids; SnowAPI uses path routing with string sub-job tokens
  (`"{job_id}:role:idx"`). The transport maps between them. Deferred.

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
