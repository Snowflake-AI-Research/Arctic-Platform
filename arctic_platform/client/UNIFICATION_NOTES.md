# Client unification notes

Goal: one client frontend where **every backend accepts identical per-op
args/kwargs and returns identically-shaped responses**, so each transport is a
dumb forwarder of `Request(op, job_id, body)` with no per-op rewiring.

This package is the target design. The op *surface* (names, args, canonical
body, which job each op targets, response contract) is defined once in
`requests.py`; a transport owns only job identity + wire mechanics.

## Module layout
| File | Holds |
|------|-------|
| `requests.py` | Every op -> `Request` builder. No transport, no event loop. |
| `base.py` | `make_transport` + the three shared frontends. |
| `sft.py` | `ArcticSFTClient`, `ArcticSFTClientConfig`. |
| `rl.py` | `ArcticRLClient`, `AsyncArcticRLClient`. |

Import the frontends from the package root — `from arctic_platform.client import
ArcticRLClient` — not from the module that happens to define them today.

`requests.py` is the single definition of the op vocabulary and must stay in
lockstep with `transport.OPS` (asserted both ways in `test_client_ops.py`).

## One op surface, thin workload subclasses
```
_ArcticClientCore          # transport, jobs, reconnect_config, get_server_state
├── ArcticClient           # blocking op surface
│   ├── ArcticSFTClient    # + sft loss default on the forward bodies
│   └── ArcticRLClient     # + log_probs
└── AsyncArcticClient      # awaitable op surface
    └── AsyncArcticRLClient  # + log_probs
```

Naming rule: **the unqualified name blocks; the `Async` prefix awaits.** So there
is no `SyncArcticSFTClient` announcing a distinction SFT does not have, and no
bare name whose call style you have to look up. `_ArcticClientCore` is private
because nobody instantiates it — it holds only what reads the same whether calls
block or are awaited. The two op surfaces below it are the one place sync and
async are written separately; they stay in step because both lower through the
same `requests.py` builders and route every call through `_call` / `_acall`,
which also carry the env-gated server-profile echo.

Construct clients directly (`ArcticRLClient(config)`); the call style is in the
name, so there are no `create_arctic_*_client` factories to pick it for you.

Keep new ops on the shared surfaces unless they genuinely require a job type or
a data contract the other workload does not have — `log_probs` is the only op
that clears that bar today.

## Design in place (this package)
- `Transport` ABC + `JobHandles` + `Request` (single op vocabulary in `requests.py`).
- `OnPremTransport` base: job creation, ordering, payload building. Concrete
  transports implement only the delivery primitives (`_start`, `call`, `_destroy`,
  `_wait_running`); `call` posts/dispatches the op against its target job.
- `JOB_CREATE_ORDER` + `ArcticClientConfig.gpus_for()` centralize GPU-gating and
  creation order so transports no longer hand-roll them.

## Config nesting (canonical)
One shared shape for every backend *and* every workload — engine knobs are never
duplicated under backend-specific aliases. The old flat, on-prem-only
`ArcticSFTClientConfig` and the `ArcticRLClientConfig` alias are both gone (note
the legacy `arctic_platform.rl.config.ArcticRLClientConfig` is a different
class):

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

`ArcticSFTClientConfig` (in `sft.py`) subclasses it and **adds no fields** — only
validators asserting that an SFT run actually trains: `training_gpus > 0` and a
`training.checkpoint_path`, both waived when `training_job_id` is set. These
cannot move onto the shared config because RL legitimately needs both exemptions
(sampling-only clients run with `training_gpus=0`). Prefer it for SFT so a bad
config fails before any job or GPU is claimed.

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
Ops present in `arctic_platform/rl/*_client.py` but not yet on `AsyncArcticRLClient`.
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
