# Cortex Training REST API Reference

> **Status.** This page is a repository snapshot for the client version in this
> branch. The Cortex Training SNOWAPI OpenAPI specification remains the source of
> truth for the wire schema.
>
> **Scope.** This document describes the customer-facing Cortex Training SnowAPI
> surface used by this repository's `CortexTrainingClient`. It covers REST paths,
> request framing, asynchronous polling, client-visible schemas, checkpoints,
> generation, operations, and logs.
>
> **Sources used for this snapshot.** The client and wire implementation in
> `cortex_training/client.py` and `cortex_training/wire.py`, the command-line
> interface in `cortex_training._cli.py`, their unit tests, and the adjacent Cortex Training
> control-plane protocol/server are the local evidence for this document. The
> SNOWAPI OpenAPI specification remains the source of truth when it is
> available.
>
> **Important distinction.** `CortexTrainingClient` is a low-level transport client.
> ArcticTraining may put additional keys such as `context` and `processing`
> inside a forward/backward batch, but those keys are backend contracts, not
> additional REST fields.

---

## 1. Core concepts

### 1.1 Jobs and sub-jobs

A job is the top-level lifecycle object. One job owns one or more typed
sub-jobs:

- `training`: model training, optimizer steps, saves, and runtime loads.
- `sampling`: generation and sampling-side operations.
- `log_probability`: a log-probability worker configuration.

The common RL layout is one training sub-job and one sampling sub-job in the
same job.

There is no dedicated public log-probability request method in the current
client. `fwd()` and `fwd_no_grad()` are aliases for the generic `forward`
operation; they are not separate REST endpoints.

### 1.2 Sub-job identifiers

Internal sub-job ids use:

```text
{job_id}:{sub_job_type}:{index}
```

`sub_job_type` is `training`, `sampling`, or `log_prob`, and `index` is the
zero-based occurrence of that type. Examples:

```text
b1fcb345:training:0
b1fcb345:sampling:0
b1fcb345:log_prob:0
```

The create-job `job_type` is `log_probability`, while its internal id segment
is `log_prob`.

### 1.3 Control and data planes

- Control-plane calls return their result synchronously: create, get, list,
  capacity, cancel, experiment metadata, and checkpoint metadata/export.
- Data-plane calls normally return a `request_id`: forward/backward, step,
  save, load, generate, weight sync, and generic operations that schedule work.
  Poll the request until it reaches a terminal state.
- Some generic operations are synchronous and return their result directly.

### 1.4 Model cache

The following models are currently in the model cache:

| Model | Training | Inference |
|---|:---:|:---:|
| `Qwen/Qwen3-0.6B` | ✅ | ✅ |
| `Qwen/Qwen3-1.7B` | ✅ | ✅ |
| `Qwen/Qwen3-8B` | ✅ | ✅ |
| `Qwen/Qwen3.5-4B` | ✅ | ✅ |
| `Qwen/Qwen3.6-35B-A3B` | ✅ | ✅ |
| `deepseek-ai/DeepSeek-V4-Flash-0731` |  | ✅ |
| `openai/gpt-oss-120b` |  | ✅ |
| `zai-org/GLM-5.2` | coming soon | ✅ |
| `zai-org/GLM-5.2-FP8` | coming soon | ✅ |

---

## 2. Connection, authentication, and headers

### 2.1 Base URL

Paths in this document are relative to:

```text
https://{account_host}/api/v2/databases/{database}/schemas/{schema}/{endpoint}
```

`CortexTrainingClient` defaults `endpoint` to `cortex-training`.

```python
client = CortexTrainingClient.from_pat(
    host=HOST,
    pat=PAT,
    database=DATABASE,
    schema=SCHEMA,
)
```

The SQL statements API used by execution-log download is outside this prefix:

```text
https://{account_host}/api/v2/statements
```

The adjacent local mock currently routes `cortex-post-training`, not the
client's `cortex-training` default. When using that mock, construct the client
with `endpoint="cortex-post-training"`.

### 2.2 PAT authentication

`CortexTrainingClient.from_pat(...)` sends:

```http
Authorization: Bearer <PAT>
X-Snowflake-Authorization-Token-Type: PROGRAMMATIC_ACCESS_TOKEN
```

TLS verification is enabled unless `verify_ssl=False` is passed.

### 2.3 Content types and framing

| Call | Request content type | Body |
|---|---|---|
| Control-plane calls, `step`, `save`, `load`, `operation` | `application/json` | JSON |
| `forward-backward` | `application/octet-stream` | DSSST1 safetensors frame, optionally split into DSSST1 request chunks |
| `generate` | `application/octet-stream` | DSSST1 safetensors frame, optionally split into DSSST1 request chunks |
| `generate-stream` | `application/octet-stream` | JSON encoded as UTF-8 bytes |

Raw `torch.save`/pickle is not the current binary protocol. See
[section 9](#9-dssst1-binary-wire-protocol).

### 2.4 Snowflake request id

Responses may include `x-snowflake-request-id`. Include it when diagnosing an
HTTP failure.

---

## 3. Status, polling, retries, and errors

### 3.1 Asynchronous request model

A scheduling call usually returns:

```json
{
  "request_id": "request-id",
  "job_id": "job-id"
}
```

Poll:

```text
GET /{job_id}/requests/{request_id}
```

`CortexTrainingClient.poll_request(job_id, request_id)` handles status normalization,
backoff, DSSST1 result decoding, and cursor-addressed result chunks.

### 3.2 Job states

Customer-facing job states are:

| State | Meaning |
|---|---|
| `pending` | Waiting for placement |
| `placing` | Infrastructure is starting (waiting for GPUs) |
| `initializing` | Placed; Cortex Training is warming up (loading the model, starting engines) |
| `running` | Ready for data-plane calls |
| `cancelled` | Cancelled by the caller |
| `terminated` | Platform teardown completed |
| `failed` | Terminal failure; inspect `reason` |

`initializing` is split out from `placing` so callers can distinguish
"waiting for GPUs" from "warming up"; it is customer-visible and not
collapsed by the server. `wait_for_job()` treats it like `placing` — neither
`running` nor terminal — so it keeps polling.

Legacy or internal responses can contain `done`, `unknown`, or enum names such
as `JOB_STATE_RUNNING`. `wait_for_job()` lowercases and removes the
`JOB_STATE_` prefix internally.

Current client caveat: `wait_for_job()` raises early for `failed`, `done`,
`cancelled`, and `canceled`, but not `terminated`; a terminated job therefore
waits until the polling timeout.

### 3.3 Request states

| State | Class |
|---|---|
| `pending` | In flight |
| `running` | In flight |
| `streaming` | In flight on a streaming backend |
| `done`, `completed`, `succeeded` | Success |
| `failed` | Failure |
| `cancelled`, `canceled` | Cancelled |

Raw enum names such as `REQUEST_STATE_DONE` can also appear.
`poll_request()` normalizes them internally.

### 3.4 Poll timing

Defaults:

- Initial interval: `0.5` seconds.
- Backoff multiplier: `1.25`.
- Maximum interval: `6` seconds.
- Overall timeout: `1800` seconds.

All values are configurable on `CortexTrainingClient(...)`.

### 3.5 Retries

The general request path retries connection/time-out failures and HTTP:

```text
404, 409, 429, 500, 502, 503, 504
```

The default is ten retries after the first attempt, with exponential jitter.

Create-job uses the narrower connection-establishment retry predicate so an
ambiguous response does not accidentally create a second server-assigned job.

### 3.6 Errors

Non-2xx responses are raised through `requests.Response.raise_for_status()`.
Bodies may contain:

```json
{"message": "description", "code": 409}
```

or validation details:

```json
{
  "detail": [
    {"loc": ["body", "field"], "msg": "description", "type": "value_error"}
  ]
}
```

Most data-plane calls and request polls require a `running` job. `tail-logs` is
the exception allowed during `placing`.

---

## 4. Endpoint index

Paths are relative to the prefix in [section 2.1](#21-base-url).

| REST path | HTTP | Client method | Purpose |
|---|---|---|---|
| `/` | `POST` | `create_job`, `create_job_from_body` | Create a job |
| `/` | `GET` | `list_jobs` | List jobs, optionally filtered by status |
| `/capacity` | `GET` | `get_capacity` | Account reservation and GPU usage |
| `/{job_id}` | `GET` | `get_job`, `wait_for_job` | Job and sub-job status |
| `/{job_id}:cancel` | `POST` | `cancel_job` | Cancel a job |
| `/{job_id}/experiment-run` | `GET` | `get_experiment_run` | Resolve experiment/run metadata |
| `/{job_id}/checkpoints` | `GET` | `list_checkpoints` | List checkpoints |
| `/{job_id}/checkpoints/{checkpoint_id}:export` | `POST` | `export_checkpoint` | Export checkpoint file links |
| `/{job_id}/checkpoints/{checkpoint_id}` | `DELETE` | `delete_checkpoint` | Delete a checkpoint |
| `/{job_id}/forward-backward` | `POST` | `forward_backward` | Submit forward plus backward |
| `/{job_id}/step` | `POST` | `step` | Submit an optimizer step |
| `/{job_id}/save` | `POST` | `save` | Save a checkpoint |
| `/{job_id}/load` | `POST` | `load` | Load a checkpoint into a running training job |
| `/{job_id}/generate` | `POST` | `generate` | Batch generation |
| `/{job_id}/generate-stream` | `POST` | `generate_stream` | Start polling-based streaming generation |
| `/{job_id}/operation` | `POST` | operation helpers | Generic routed operation |
| `/{job_id}/requests/{request_id}` | `GET` | `get_request_status`, `poll_request` | Poll async work |

The operation endpoint supports the operation types in
[section 7](#7-generic-operation-envelope).

One auxiliary call is outside the Cortex Training prefix:

| REST path | HTTP | Client use | Purpose |
|---|---|---|---|
| `/api/v2/statements` | `POST` | `fetch_execution_logs` | Resolve scoped experiment-stage credentials |

---

## 5. Control-plane endpoints

### 5.1 Create job - `POST /`

Typed Python call:

```python
job_id = client.create_job(
    sub_jobs=[training_sub_job, sampling_sub_job],
    experiment_name=None,
)
```

REST body:

```json
{
  "sub_job_configs": [
    {
      "job_type": "sampling",
      "model_name": "Qwen/Qwen3-1.7B",
      "inference_config": {
        "max_seq_len": 2048,
        "n_gpus": 1
      }
    }
  ],
  "experiment_name": "optional-experiment"
}
```

`sub_job_configs` must be a non-empty list. The typed path validates each
`SubJobConfig`; `create_job_from_body()` only checks the outer body and non-empty
list before forwarding it.

Response:

```json
{"job_id": "job-id"}
```

#### Internal debug body

`create_job_from_body()` can forward a top-level `debug` block only when
`CORTEX_TRAINING_ENABLE_DEBUG_OPTIONS` is truthy (`1`, `true`, `yes`, or `on`).
This is an internal capability and is separately gated by the server.

### 5.2 Get job - `GET /{job_id}`

The SnowAPI shape consumed by this repository is flat at each sub-job:

```json
{
  "job_id": "job-id",
  "status": "running",
  "reason": "",
  "created_at": "2026-07-20T18:00:00Z",
  "updated_at": "2026-07-20T18:01:00Z",
  "image_tag": "release-tag",
  "submitted_by": "SOME_USER",
  "owner_role": "SOME_ROLE",
  "sub_jobs": [
    {
      "sub_job_id": "job-id:training:0",
      "job_type": "training",
      "status": "running",
      "model_name": "Qwen/Qwen3-1.7B",
      "training_config": {
        "optimizer": {"name": "AdamW", "lr": 0.0001},
        "max_seq_len": 2048,
        "train_batch_size": 8,
        "n_gpus": 8
      }
    }
  ]
}
```

`image_tag` can be empty before placement. `wait_for_job()` repeatedly calls
this endpoint until `running`.

`submitted_by` (submitting user's name) and `owner_role` (owning role's name)
are best-effort and may be absent, e.g. if the user/role has since been
dropped or the requesting role no longer has access to it. Both fields also
appear on each job entry returned by `list_jobs()`
([section 5.3](#53-list-jobs---get-)).

### 5.3 List jobs - `GET /`

Optional query:

```text
?status=running
```

REST response:

```json
{"jobs": [{"job_id": "job-id", "status": "running", "sub_jobs": []}]}
```

`CortexTrainingClient.list_jobs()` returns only the `jobs` list, not the outer object.
The client forwards the status string without validating an enum.

### 5.4 Capacity - `GET /capacity`

This account-scoped endpoint takes no account id from the caller. The server
resolves the account from the authenticated session.

```json
{
  "has_reservation": true,
  "max_total_gpus": 64,
  "reserved_gpus": 64,
  "in_use_gpus": 8,
  "available_gpus": 56
}
```

- `has_reservation`: whether the account has reserved GPU capacity.
- `max_total_gpus`: the account's GPU ceiling.
- `reserved_gpus` **(deprecated)**: use `max_total_gpus` with
  `has_reservation` instead.
- `in_use_gpus`: GPUs used by the account's `pending`, `placing`,
  `initializing`, and `running` jobs.
- `available_gpus`: remaining capacity, floored at zero and potentially
  capped by currently schedulable capacity.

Proto3 JSON may omit false or zero fields. `get_capacity()` does not yet
surface `max_total_gpus`; it fills in defaults for the other four keys.

### 5.5 Cancel job - `POST /{job_id}:cancel`

No body. Pending, placing, and running jobs enter cancellation. Repeating a
cancel while a job is cancelling or already cancelled is an idempotent success.
Terminated or failed jobs return a precondition/conflict-style error.

`cancel_job()` returns `None`.

### 5.6 Experiment run - `GET /{job_id}/experiment-run`

```json
{
  "experiment_name": "DB.SCHEMA.EXPERIMENT",
  "experiment_run_name": "RUN_NAME"
}
```

`fetch_execution_logs()` uses these values to locate the run's stage.

### 5.7 List checkpoints - `GET /{job_id}/checkpoints`

```json
{
  "checkpoints": [
    {
      "checkpoint_id": "global_step12",
      "global_steps": 12,
      "avg_loss": 0.83,
      "created_at": "2026-07-20T18:05:00Z",
      "checkpoint_type": "resumable"
    }
  ]
}
```

`list_checkpoints()` returns only the `checkpoints` list. The CLI exposes this
method with a required job id and restores the server-shaped JSON envelope:

```bash
cortex-training checkpoints JOB_ID
```

The command prints `{"checkpoints": [...]}`. As with other commands, compact
output is selected with the global option:

```bash
cortex-training --compact checkpoints JOB_ID
```

### 5.8 Export checkpoint

```text
POST /{job_id}/checkpoints/{checkpoint_id}:export
```

No body.

```json
{
  "checkpoint_id": "global_step12",
  "files": [
    {
      "filename": "model.safetensors",
      "url": "https://presigned.example/...",
      "size_bytes": 1073741824
    }
  ],
  "expires_at": "2026-07-20T19:05:00Z"
}
```

The URLs are short-lived.

### 5.9 Delete checkpoint

```text
DELETE /{job_id}/checkpoints/{checkpoint_id}
```

No body, and none returned: `200`/`204` on success, so `delete_checkpoint()`
returns `None`. Unlike `:export`, this is a plain `DELETE` verb on the checkpoint
resource. A missing checkpoint is not treated as success — the server's status
(e.g. `404`) propagates as a raised error.

---

## 6. Data-plane endpoints

### 6.1 Forward/backward - `POST /{job_id}/forward-backward`

The body is a DSSST1 frame. The standard object is:

```python
{
    "args": (),
    "kwargs": {
        "input_ids": input_ids,
        "labels": labels,
    },
}
```

ArcticTraining can add backend-owned `context` and `processing` keys.

Build a basic frame with:

```python
payload = serialize_forward_backward_args(
    args=(),
    kwargs={"input_ids": input_ids, "labels": labels},
)
request_id = client.forward_backward(job_id, payload)
result = client.poll_request(job_id, request_id)
```

Or serialize an extended batch directly:

```python
from cortex_training import wire

payload = wire.dumps(
    batch,
    metadata={
        "response_options": {
            "format": "dssst1",
            "delivery": "chunked",
        }
    },
)
```

`forward_backward()` splits a frame larger than 60 MiB into DSSST1 request
chunks. Each chunk is posted to the same path; only the last response may carry
the `request_id`.

Typical polled result:

```json
{
  "job_id": "job-id",
  "avg_loss": 1.0237,
  "metrics": {},
  "post_process_outputs": {}
}
```

The exact metrics are backend/loss dependent. The current ArcticTraining
forward/backward response assembler intentionally returns
`post_process_outputs` as an empty object; callers must not assume that
`compute_logprobs` appears there.

### 6.2 Optimizer step - `POST /{job_id}/step`

```json
{"learning_rate": 0.00002}
```

The field is optional. `step(job_id)` sends `{}`.

Immediate response:

```json
{"request_id": "request-id", "job_id": "job-id"}
```

Typical result:

```json
{"global_steps": 12, "last_lr": 0.00002}
```

`last_lr` may be a scalar or list depending on the backend.

When per-step peak-memory reporting is enabled (see
[section 8.5](#85-memory-diagnostics-settings)), the result also carries a
`peak_memory` object of byte counts:

```json
{
  "global_steps": 12,
  "last_lr": 0.00002,
  "peak_memory": {"gpu_peak": 123456789, "cpu_peak": 987654321}
}
```

`gpu_peak` is the job-wide maximum GPU high-water mark across ranks for the
current `/forward-backward` plus `/step` cycle; `cpu_peak` is rank 0's current
process RSS. The key is absent when the setting is disabled.

### 6.3 Save checkpoint - `POST /{job_id}/save`

```json
{
  "checkpoint_id": "optional-tag",
  "checkpoint_type": "weights-only"
}
```

Both fields are optional in the Python client request. When `checkpoint_type`
is supplied, the client lowercases and validates it as:

- `resumable`: training weights plus optimizer/training state.
- `weights-only`: Hugging Face-style model assets suitable for sampling
  initialization.

The backend default is `resumable`.

Compatibility caveat: `checkpoint_id` is not represented in the adjacent
control-plane `SaveRequest`, so it is not forwarded in that control plane's
save payload. Treat the `checkpoint_id` returned by the polled result as
authoritative rather than relying on a caller-selected value.

Immediate response:

```json
{"request_id": "request-id", "job_id": "job-id"}
```

Typical result fields include `checkpoint_id`, `checkpoint_path`, and
`checkpoint_tag`; consumers should use the fields actually present.

### 6.4 Runtime load - `POST /{job_id}/load`

This endpoint was missing from the previous document.

```json
{
  "checkpoint_id": "global_step12",
  "source_job_id": "optional-source-job",
  "target_sub_job_id": "optional-job-id:training:0"
}
```

`checkpoint_id` is required. `source_job_id` loads from another job's checkpoint
store. `target_sub_job_id` routes the load to a specific training sub-job of the
session. The route is asynchronous:

```python
request_id = client.load(
    job_id,
    checkpoint_id="global_step12",
    source_job_id=source_job_id,
    target_sub_job_id=f"{job_id}:training:0",  # optional
)
result = client.poll_request(job_id, request_id)
```

When `target_sub_job_id` is omitted, the control plane routes the load to the
session's training sub-job (the historical behavior). When supplied, the sub-job
must belong to this session — otherwise the request fails with `400` — and must
be a training sub-job; loading into a non-training (e.g. sampling) zone is
rejected with `501`.

#### Discovering sub-job IDs

Sub-job IDs follow the format `{job_id}:{job_type}:{index}`, for example:
`b1fcb345:training:0`. To discover available training sub-jobs:

```python
job = client.get_job(job_id)
training_sub_jobs = [
    sj for sj in job["sub_jobs"] if sj["job_type"] == "training"
]
for sj in training_sub_jobs:
    print(f"ID: {sj['sub_job_id']}, DP size: {sj['training_config']['n_gpus']}")
```

#### When to use target_sub_job_id

Most sessions have a single training sub-job, so omit `target_sub_job_id` to use
the default. Use it when:

- The session has multiple training sub-jobs
- You need to load different checkpoints into different sub-jobs
- You want explicit routing control

#### DP size compatibility

If the target sub-job has a different `n_gpus` than the checkpoint's source
sub-job (changing DP size), the target sub-job **must** have been created with
`load_optimizer_states: false` in its `SubJobConfig`. The optimizer states are
DP-sharded and cannot be resized. This is a **creation-time** setting; it cannot
be changed at runtime. The server will reject incompatible loads.

### 6.5 Create-time checkpoint initialization

Create-time initialization is not `resume_from_checkpoint` in the typed
`cortex-training` API. Use:

```json
{
  "source_checkpoint_info": {
    "checkpoint_id": "checkpoint-id",
    "source_job_id": "source-job-id"
  }
}
```

This object is a field on `SubJobConfig`. The server stamps scoped `stage_info`
credentials; clients should not provide credentials themselves.

Sampling initialization requires a `weights-only` checkpoint. The source job no
longer needs to be running after the checkpoint has been saved.

For training initialization from a `resumable` checkpoint, set
`training_config.load_optimizer_states=false` (see
[section 8.2](#82-trainingconfig)) to restore weights only with a fresh
optimizer. This is required when the new job changes data-parallel size, since
the DP-sharded optimizer cannot be resized.

### 6.6 Generate - `POST /{job_id}/generate`

Python call:

```python
request_id = client.generate(
    job_id,
    prompts=["Hello", [1, 2, 3]],
    sampling_params={"max_tokens": 64, "temperature": 0.7},
    routing_key="conversation-1",
    strict=False,
)
```

Logical request object inside the DSSST1 frame:

```json
{
  "prompts": ["Hello", [1, 2, 3]],
  "sampling_params": {"max_tokens": 64, "temperature": 0.7},
  "routing_key": "conversation-1",
  "strict": false
}
```

Rules:

- `prompts` is a list of string prompts and/or token-id lists. A single
  tokenized prompt is `[[1, 2, 3]]`, not `[1, 2, 3]`.
- `sampling_params` may be one object or a list of objects/null values aligned
  with `prompts`.
- `routing_key` may be one string or an aligned list of strings/null values.
- `strict` controls strict routing-key affinity.

For pre-tokenized prompts, the client fetches and caches the sampling sub-job's
`inference_config.max_seq_len`. It rejects a prompt when
`len(prompt) >= max_seq_len`, preserving room for at least one output token.
String prompts are left for the server tokenizer to validate.

Like forward/backward, generate uses DSSST1 response options and splits frames
larger than 60 MiB into request chunks.

Typical polled result:

```json
{
  "job_id": "job-id",
  "results": [
    {
      "text": " generated text",
      "token_ids": [101, 102],
      "finish_reason": "stop"
    }
  ]
}
```

The backend can add fields such as log probabilities or action masks. For a
request submitted by the same `CortexTrainingClient` instance, `poll_request()`
converts tensor values under `results` back to Python lists.

### 6.7 Streaming generate - `POST /{job_id}/generate-stream`

`generate_stream()` accepts the same logical fields as `generate()`, but its
body is UTF-8 JSON under `application/octet-stream`, not DSSST1.

The encoded body must not exceed 60 MiB. This path is not request-chunked by the
client.

Immediate response:

```json
{
  "request_id": "request-id",
  "job_id": "job-id",
  "count": 2
}
```

Poll the unified request endpoint with `max_events`:

```python
status = client.get_request_status(
    job_id,
    request_id,
    max_events=64,
)
```

Events are opaque backend objects. Common event shapes are:

```json
{"type": "result", "index": 0, "result": {"text": "text", "token_ids": [1, 2]}}
```

```json
{"type": "error", "index": 1, "error": "description"}
```

```json
{"type": "done", "completed": 1, "failed": 1}
```

Streaming event delivery advances a server-side delivery cursor. There is no
client-supplied retry cursor for these stream events, so losing a successful
poll response can lose the events consumed by that response.

### 6.8 Poll request - `GET /{job_id}/requests/{request_id}`

Optional query parameters:

| Parameter | Meaning |
|---|---|
| `max_events` | Event/result-chunk count; server default is 16 and current cap is 512 |
| `cursor` | Retry cursor for cursor-addressed DSSST1 result chunks |

Example:

```json
{
  "request_id": "request-id",
  "status": "done",
  "created_at": "2026-07-20T18:00:00Z",
  "updated_at": "2026-07-20T18:00:01Z",
  "result": {},
  "error": "",
  "events": [],
  "next_cursor": ""
}
```

There are two distinct event-delivery modes:

1. Streaming-generation events are destructively drained by `max_events`.
2. Large DSSST1 results use `result_chunk` events and `next_cursor`.
   `poll_request()` echoes `next_cursor` as `cursor`, validates each chunk hash,
   reassembles the frame, and decodes the result.

`poll_request()` returns `{}` if a successful terminal status has no result. It
raises `RuntimeError` for failed/cancelled status and `TimeoutError` at the
configured deadline.

---

## 7. Generic operation envelope

```text
POST /{job_id}/operation
Content-Type: application/json
```

```json
{
  "operation_type": "weight-sync",
  "sub_job_id": "job-id:training:0",
  "sub_job_type": "training",
  "payload": {}
}
```

- `operation_type` is required.
- `sub_job_id` and `sub_job_type` are optional routing hints. If both are
  supplied, they must resolve to the same target.
- `payload` is operation-specific.
- Byte payloads passed to `CortexTrainingClient.forward()` are converted to
  `{"payload_b64": "...", "content_type": "application/octet-stream"}` inside
  the JSON envelope.
- Operation responses are opaque. If a response contains a `request_id`, poll
  it; otherwise consume it inline.

The adjacent current control plane accepts:

| Operation type | Client method | Execution |
|---|---|---|
| `forward` | `forward`, `fwd`, `fwd_no_grad` | Async on the current backend |
| `weight-sync` | `weight_sync` | Async |
| `bootstrap-router-replay` | `bootstrap_router_replay` | Inline on the current backend; poll if a deployment returns a request id |
| `router-replay-discard` | `router_replay_discard` | Inline |
| `reset-prefix-cache` | `reset_prefix_cache` | Inline |
| `cancel-request` | `cancel_request` | Inline acknowledgement |
| `tail-logs` | `tail_logs`, `stream_logs` | Inline cursor page |

Only `tail-logs` is allowed while the job is `placing`; all other operations
require `running`.

### 7.1 Forward

```json
{
  "operation_type": "forward",
  "sub_job_id": "job-id:training:0",
  "sub_job_type": "training",
  "payload": {
    "payload_b64": "base64-data",
    "content_type": "application/octet-stream"
  }
}
```

`forward()`, `fwd()`, and `fwd_no_grad()` all send exactly
`operation_type="forward"`. The aliases do not add no-gradient semantics.

Byte payloads over 60 MiB are rejected; this operation path is not request
chunked. The current ArcticTraining Cortex Training adapter still treats no-grad
forward as unavailable, so this low-level route should not be presented as a
portable log-probability API.

There is also a current wire mismatch: `_operation()` wraps byte payloads in a
base64 JSON object, while the adjacent operation proxy forwards that JSON to a
backend `/forward` route that expects raw DSSST1 bytes. Request construction is
unit-tested, but byte-based `forward()` is not presently end-to-end compatible.

### 7.2 Weight sync

```json
{
  "operation_type": "weight-sync",
  "sub_job_id": "job-id:training:0",
  "sub_job_type": "training",
  "payload": {
    "source_sub_job_id": "job-id:training:0",
    "target_sub_job_ids": ["job-id:sampling:0"],
    "weight_format": "vllm"
  }
}
```

`weight_sync()` defaults operation routing to the source training sub-job and
returns the `request_id` string.

Optional payload fields:

- `weight_format`: `"vllm"` (default on server) | `"hf"` for full-model
  weights. For adapter-only synchronization, see
  [section 7.3](#73-lora-adapter-sync).

### 7.3 LoRA adapter sync

LoRA synchronization uses the same `weight-sync` operation, but it requires
LoRA to be enabled on both sub-jobs when they are created. It is not sufficient
to change only `weight_format` at sync time:

- The training sub-job's `training_config.peft_config` applies PEFT and makes
  the LoRA adapter parameters trainable.
- The sampling sub-job's `inference_config.peft_config` enables the vLLM LoRA
  manager at engine startup.
- The sampling configuration must match the training adapter's `r`,
  `lora_alpha`, and `target_modules`. Reusing the same configuration object for
  both sub-jobs is the simplest way to keep them aligned.

The sync request transfers only the trained adapter tensors:

```json
{
  "operation_type": "weight-sync",
  "sub_job_id": "job-id:training:0",
  "sub_job_type": "training",
  "payload": {
    "source_sub_job_id": "job-id:training:0",
    "target_sub_job_ids": ["job-id:sampling:0"],
    "weight_format": "lora"
  }
}
```

Subsequent syncs replace the resident adapter under the same stable adapter
identity.

CLI equivalent:

```bash
cortex-training --job-id JOB_ID weight-sync \
  --weight-format lora
```

See [section 13.5](#135-enable-lora-training-and-adapter-sync) for the complete
create, train, sync, and generate workflow.

### 7.4 Bootstrap router replay

```json
{
  "operation_type": "bootstrap-router-replay",
  "sub_job_id": "job-id:training:0",
  "sub_job_type": "training",
  "payload": {
    "source_sub_job_id": "job-id:sampling:0",
    "target_sub_job_id": "job-id:training:0",
    "max_cache_bytes": 4096
  }
}
```

The sampling sub-job is the routing source and the training sub-job is the
replay target/operation receiver. `max_cache_bytes` is optional.
For a mixed training/sampling job, pass `sub_job_id=target_sub_job_id` (or the
matching `sub_job_type`) because the low-level client does not infer the
operation receiver.

### 7.5 Router replay discard

This API was missing from the previous document.

```json
{
  "operation_type": "router-replay-discard",
  "sub_job_id": "job-id:sampling:0",
  "sub_job_type": "sampling",
  "payload": {
    "sample_ids": ["sample-1", "sample-2"]
  }
}
```

`router_replay_discard()` also accepts an `extra` payload for forward-compatible
fields. Explicit `extra["sample_ids"]` takes precedence over the method's
`sample_ids` argument.

### 7.6 Reset prefix cache

```json
{
  "operation_type": "reset-prefix-cache",
  "sub_job_id": "job-id:sampling:0",
  "sub_job_type": "sampling",
  "payload": {
    "drain": true,
    "timeout_s": 60.0,
    "retry_interval_s": 0.1
  }
}
```

The shown payload values are the client defaults. `extra` can supply new
backend fields and override `drain`.

### 7.7 Cancel request

```json
{
  "operation_type": "cancel-request",
  "payload": {
    "request_id": "request-id"
  }
}
```

With a self-describing request id, the server can recover the owning sub-job.
For legacy ids, omitting a sub-job hint causes server-side fan-out across the
job's sub-jobs.

### 7.8 Tail logs

```json
{
  "operation_type": "tail-logs",
  "sub_job_id": "job-id:training:0",
  "payload": {
    "cursor": "cursor-0",
    "max_lines": 50
  }
}
```

Response:

```json
{
  "entries": [],
  "next_cursor": "cursor-1",
  "eof": true
}
```

`stream_logs()` repeatedly calls this operation. With `follow=False`, it stops
at an empty EOF page; with `follow=True`, it keeps polling.

### 7.9 Unsupported `zmd-events` client helper

`CortexTrainingClient.tail_events()` and `stream_events()` currently send:

```json
{"operation_type": "zmd-events"}
```

However, `zmd-events` is absent from the adjacent control plane's canonical
supported-operation set and operation registry. The methods are unit-tested
only for request construction. Treat them as unavailable until the server adds
the operation or the client removes/changes the helpers.

---

## 8. Create-job schemas

### 8.1 `SubJobConfig`

Exactly one type-specific config is set.

| Field | Type | Required | Notes |
|---|---|---|---|
| `job_type` | `training`, `sampling`, `log_probability` | yes | Typed `JobType` enum |
| `model_name` | string | yes | Must be non-empty |
| `training_config` | object | for training | Produced from `TrainingConfig` |
| `inference_config` | object | for sampling/log probability | Produced from `InferenceConfig` |
| `global_batch_size` | integer | no | Top-level passthrough field |
| `dtype` | string | no | Example: `bfloat16` |
| `seed` | integer | no | |
| `model_post_init` | list of strings | no | Server maps to post-init hooks |
| `source_checkpoint_info` | object | no | Create-time checkpoint initialization |

Typed factories:

```python
SubJobConfig.training_job(...)
SubJobConfig.sampling_job(...)
```

The typed client has no `resume_from_checkpoint` field. Use
`source_checkpoint_info`.

### 8.2 `TrainingConfig`

The typed client requires:

| Field | Type | Client validation |
|---|---|---|
| `optimizer` | object | Non-empty |
| `max_seq_len` | integer | Greater than zero |
| `train_batch_size` | integer | Greater than zero |
| `n_gpus` | integer | Greater than zero |

Optional typed fields:

- `gradient_clipping`
- `multiplex_job_id`
- `load_optimizer_states`

`load_optimizer_states` controls checkpoint resume behavior for this training
sub-job. When `false`, resuming a `resumable` checkpoint (via
`source_checkpoint_info` or the runtime `/load` endpoint) restores model weights
only and starts the optimizer fresh. Set it `false` to change data-parallel size
while keeping expert parallelism, because the DP-sharded optimizer cannot be
resized; a `true`/default resume against a different DP/world size fails fast
with an actionable error. `None` (omitted) uses the server default (`true`).

`extra_training` is merged as open passthrough data, without overriding typed
keys. Examples include `model_provider`, `ep_size`, `ds_config`,
`activation_checkpointing`, `prime_rl`, `router_replay`, `peft_config`, the
memory-diagnostics settings in [section 8.5](#85-memory-diagnostics-settings),
and these newer long-context/memory knobs:

- `sp_size`: Ulysses sequence-parallel degree, sharding each sample's sequence
  across `sp_size` ranks. `1` (default) disables it; the server requires
  `world_size % sp_size == 0`.
- `cuda_allocator_conf`: per-job `PYTORCH_CUDA_ALLOC_CONF` string applied before
  PyTorch is imported, e.g. `"expandable_segments:True"` (or `"backend:native"`
  to disable). Omitted uses the server default.
- `ac_config`: activation-checkpointing config, including CPU activation offload
  via `offload_config.enabled=true`. Offload requires `mode="full"`.

For LoRA training, set `extra_training["peft_config"]` to a PEFT
`LoraConfig`-compatible object. At minimum, specify `peft_type="Lora"`; `r` and
`lora_alpha` default to `8` on the server, although explicit values are
recommended when the same configuration is also used by sampling.

The current client also rejects an effective PrimeRL config that combines an
enabled/default `fused_cross_entropy` with `fp32_lm_head=True` or an integer
`fused_lm_head_token_chunk_size`.

### 8.3 `InferenceConfig`

The typed client requires:

| Field | Type | Client validation |
|---|---|---|
| `max_seq_len` | integer | Greater than zero |
| `n_gpus` | integer | Greater than zero |

`multiplex_job_id` is optional. `extra_sampling` is an open passthrough object;
common values include `gpu_memory_utilization` and a nested `vllm_config`.

For LoRA sampling, set `extra_sampling["peft_config"]` when creating the
sub-job. This enables the vLLM LoRA manager before the model starts. The
adapter's `r`, `lora_alpha`, and `target_modules` must match the training
configuration used for adapter synchronization.

For either config type, the server requires `multiplex_job_id` to be a complete
`{job_id}:{sub_job_type}:{index}` id outside the job being created.

### 8.4 `source_checkpoint_info`

```json
{
  "checkpoint_id": "checkpoint-id",
  "source_job_id": "source-job-id"
}
```

`checkpoint_id` is required by the server-side source-checkpoint model.
`source_job_id` is optional in the client/proto shape, but cross-job
initialization must identify the job that owns the saved checkpoint. The typed
client forwards this object without validating either field.

### 8.5 Memory-diagnostics settings

Training workers expose low-volume, customer-facing memory observability through
`training_config` keys. The typed client has no dedicated fields for these, so
pass them through `extra_training`. Each is an optional per-job override; when
omitted or `null`, the server system default applies.

| Key | Type | Effect |
|---|---|---|
| `step_peak_memory_log` | bool | Adds a `peak_memory` object (`gpu_peak`, `cpu_peak` byte counts) to each `/step` result. See [section 6.2](#62-optimizer-step---post-job_idstep). |
| `training_memory_telemetry` | bool | Emits structured allocator events (`fwd_bwd_end`, `step_start`, `step_end`) to the training event stream. |

```python
training = SubJobConfig.training_job(
    model_name="Qwen/Qwen3-1.7B",
    optimizer={"name": "AdamW", "lr": 1e-4},
    max_seq_len=2048,
    train_batch_size=8,
    n_gpus=8,
    extra_training={"step_peak_memory_log": True},
)
```

`step_peak_memory_log` costs one `all_reduce(MAX)` per step; both settings are
off by default. Failure snapshots on a failed `/forward-backward` or `/step` are
independent of these flags. Per-rank operator diagnostics configured by the
launcher are out of scope for this client-facing surface.

---

## 9. DSSST1 binary wire protocol

### 9.1 Why DSSST1

DSSST1 is a safetensors-based frame implemented by `cortex_training.wire`. It
serializes nested dict/list/tuple structures containing tensors and JSON-safe
values without pickle execution.

`wire.loads()` explicitly rejects legacy pickle and `torch.save` signatures.

### 9.2 Frame contents

A frame is one safetensors blob:

```text
u64 header length | safetensors JSON header | tensor bytes
```

Safetensors metadata contains:

- `dss`: the nested structure skeleton and wire version `DSSST1`.
- `op`: optional operation metadata such as response options, router replay,
  and request/result chunk descriptors.

Encode/decode:

```python
from cortex_training import wire

frame = wire.dumps(value, metadata=metadata)
value = wire.loads(frame)
metadata = wire.read_metadata(frame)
```

### 9.3 Request chunking

`forward_backward()` and `generate()` call:

```python
wire.encode_byte_chunks(
    frame,
    kind="request",
    operation="fwd-bwd-or-generate",
    max_bytes=60 * 1024 * 1024,
)
```

`forward_backward()` always sends a chunk envelope, including when the request
fits in one chunk, so every logical operation has a caller-generated identity.
`generate()` still sends the original frame unchanged when it fits. Each
DSSST1 request chunk contains:

- A `uint8` payload tensor.
- `chunk_idx` and `total_chunks`.
- `chunk_group_id`.
- Original frame size and SHA-256.
- Operation name.

For forward-backward, `chunk_group_id` is also the idempotency identity and
`frame_sha256` binds that identity to one exact frame:

- The server retains accepted chunks with the training job rather than the
  HTTP connection.
- An unknown group must start at chunk `0`. A later first chunk returns `409`
  with code `chunk_group_restart_required`; the client may replay once from
  chunk `0` using the same encoded chunks and group id.
- Repeating the same group id and frame hash returns or waits for the original
  execution result. It does not run forward-backward again.
- Reusing a group id with a different frame hash, chunk count, or chunk payload
  returns `409`.

The final chunk schedules work and returns the pollable `request_id`. Retrying a
final chunk after losing its response can return a new `request_id`, but both
request records resolve through the same at-most-once forward-backward
execution.

### 9.4 Encoded results

A non-chunked DSSST1 result can appear inside poll JSON as:

```json
{
  "content_type": "application/octet-stream",
  "encoding": "base64",
  "wire_format": "DSSST1",
  "payload_b64": "base64-frame"
}
```

`poll_request()` base64-decodes and passes the frame to `wire.loads()`.

### 9.5 Result chunks

Large results can arrive through poll events:

```json
{
  "type": "result_chunk",
  "payload_b64": "base64-chunk-frame",
  "payload_sha256": "sha256"
}
```

A poll can also include `next_cursor`. `poll_request()` drains all pages,
validates chunk SHA-256 values, uses `wire.decode_result_chunks()`, and returns
the reconstructed object.

---

## 10. Forward/backward batch contract

### 10.1 Transport-level object

The transport accepts a DSSST1 object. The minimal conventional shape is:

```python
batch = {
    "args": (),
    "kwargs": {
        "input_ids": input_ids,
        "labels": labels,
    },
}
```

Common model kwargs are `input_ids`, `attention_mask`, `position_ids`, and
`labels`, generally shaped `[batch, sequence]`.

### 10.2 Readable payload helper

`build_forward_backward_payload(spec)` is a CLI/readability helper. It builds
only `args` and `kwargs`, then calls `serialize_forward_backward_args()`.

Direct tensor-shaped JSON:

```json
{
  "payload": {
    "kwargs": {
      "input_ids": {"data": [[1, 2, 3]], "dtype": "long"},
      "labels": {"data": [[2, 3, -100]], "dtype": "long"}
    }
  }
}
```

It can also tokenize `texts` with a configured Transformers tokenizer.

When building labels, supported strategies are:

- `next_token` / `shifted_input_ids`
- `input_ids` / `self`
- `none`
- explicit tensor data

The default helper strategy is next-token labels. It rolls input ids left,
sets the last target to `ignore_index` (default `-100`), and can mask padding.

### 10.3 ArcticTraining extensions

ArcticTraining serializes a larger object directly with `wire.dumps()`:

```python
batch = {
    "args": (),
    "kwargs": {...},
    "context": {...},
    "processing": {
        "loss_fn": "grpo",
        "config": {...},
        "post": [...],
    },
}
```

`context` and `processing` are open backend contracts. Their accepted keys,
registered loss functions, and post-processors come from the deployed training
backend, not the SnowAPI schema.

Do not use `build_forward_backward_payload()` for this extended shape: that
helper currently ignores `context` and `processing`.

### 10.4 Shifted log-probability convention

ArcticTraining's RL adapter documents `_shifted` log-probability tensors as:

```text
tensor[:, i] is the log probability of input_ids[:, i + 1]
```

This is an ArcticTraining processing convention, not a REST-level field
requirement.

---

## 11. Result summary

These are conventional backend results, not closed SnowAPI schemas:

| Operation | Common result |
|---|---|
| `forward-backward` | `job_id`, `avg_loss`, `metrics`, `post_process_outputs` |
| `step` | `global_steps`, `last_lr`, optional `peak_memory` |
| `save` | `checkpoint_id`, `checkpoint_path`, `checkpoint_tag` |
| `load` | `checkpoint_id` and backend load metadata |
| `generate` | `job_id`, `results[]` |
| `weight-sync` | Completion/transfer metadata |
| `forward` | Backend-specific forward-only result |

Generic-operation responses and fields inside `metrics` or generation results
are intentionally open.

---

## 12. Logs and events

### 12.1 Live logs

Use `tail_logs()` for one cursor page or `stream_logs()` for an iterator. This
uses the `tail-logs` operation described in [section 7.8](#78-tail-logs).

The server reads the selected sub-job's zone-manager/head-pod stdout. It can
serve empty, non-EOF pages during placement while the pod is still appearing.

### 12.2 Full execution-log download

`fetch_execution_logs(job_id)`:

1. Calls `GET /{job_id}/experiment-run`.
2. Calls `POST /api/v2/statements` with
   `SYSTEM$GET_VSTAGE_WRITE_CREDS(...)`.
3. Uses the returned scoped S3 credentials to list the experiment stage.
4. Downloads every object below a `/_logs/{sub_job_id}/` subtree.

Return:

```python
[
    {
        "sub_job_id": "job-id:training:0",
        "filename": "execution.jsonl",
        "s3_uri": "s3://bucket/key",
        "content": "...",
    }
]
```

Only S3 stage credentials are implemented by this client.

### 12.3 ZMD events

The client contains `tail_events()` and `stream_events()`, but their
`zmd-events` operation is not accepted by the current adjacent server. See
[section 7.9](#79-unsupported-zmd-events-client-helper).

---

## 13. End-to-end examples

### 13.1 Create training and sampling sub-jobs

```python
from cortex_training.client import CortexTrainingClient, SubJobConfig

client = CortexTrainingClient.from_pat(
    host=HOST,
    pat=PAT,
    database=DATABASE,
    schema=SCHEMA,
)

training = SubJobConfig.training_job(
    model_name="Qwen/Qwen3-1.7B",
    optimizer={"name": "AdamW", "lr": 1e-4},
    max_seq_len=2048,
    train_batch_size=8,
    n_gpus=8,
    dtype="bfloat16",
)
sampling = SubJobConfig.sampling_job(
    model_name="Qwen/Qwen3-1.7B",
    max_seq_len=2048,
    n_gpus=1,
    dtype="bfloat16",
)

job_id = client.create_job(sub_jobs=[training, sampling])
client.wait_for_job(job_id)

training_id = f"{job_id}:training:0"
sampling_id = f"{job_id}:sampling:0"
```

### 13.2 Generate, train, step, and sync

```python
request_id = client.generate(
    job_id,
    prompts=["Write a short proof."],
    sampling_params={"max_tokens": 256, "temperature": 0.7},
)
rollouts = client.poll_request(job_id, request_id)["results"]

request_id = client.forward_backward(job_id, fwd_bwd_dssst1_frame)
train_result = client.poll_request(job_id, request_id)

request_id = client.step(job_id, learning_rate=1e-4)
step_result = client.poll_request(job_id, request_id)

request_id = client.weight_sync(
    job_id,
    source_sub_job_id=training_id,
    target_sub_job_ids=[sampling_id],
)
client.poll_request(job_id, request_id)
```

### 13.3 Save and runtime load

```python
request_id = client.save(job_id, checkpoint_type="resumable")
checkpoint = client.poll_request(job_id, request_id)

request_id = client.load(
    job_id,
    checkpoint_id=checkpoint["checkpoint_id"],
)
client.poll_request(job_id, request_id)
```

### 13.4 Start sampling from saved weights

```python
request_id = client.save(training_job_id, checkpoint_type="weights-only")
checkpoint = client.poll_request(training_job_id, request_id)

sampling = SubJobConfig.sampling_job(
    model_name="Qwen/Qwen3-1.7B",
    max_seq_len=2048,
    n_gpus=1,
    source_checkpoint_info={
        "checkpoint_id": checkpoint["checkpoint_id"],
        "source_job_id": training_job_id,
    },
)
sampling_job_id = client.create_job(sub_jobs=[sampling])
```

### 13.5 Enable LoRA training and adapter sync

Configure the same adapter on both sub-jobs. The training configuration creates
the trainable PEFT parameters; the sampling configuration enables LoRA support
in vLLM before the first adapter sync.

```python
from cortex_training.client import CortexTrainingClient, SubJobConfig

client = CortexTrainingClient.from_pat(
    host=HOST,
    pat=PAT,
    database=DATABASE,
    schema=SCHEMA,
)

lora_config = {
    "peft_type": "Lora",
    "r": 8,
    "lora_alpha": 8,
    "lora_dropout": 0.0,
    "bias": "none",
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
}

training = SubJobConfig.training_job(
    model_name="Qwen/Qwen3-1.7B",
    optimizer={"name": "AdamW", "lr": 1e-4},
    max_seq_len=2048,
    train_batch_size=8,
    n_gpus=8,
    dtype="bfloat16",
    extra_training={"peft_config": lora_config},
)
sampling = SubJobConfig.sampling_job(
    model_name="Qwen/Qwen3-1.7B",
    max_seq_len=2048,
    n_gpus=1,
    dtype="bfloat16",
    extra_sampling={"peft_config": lora_config},
)

job_id = client.create_job(sub_jobs=[training, sampling])
client.wait_for_job(job_id)

training_id = f"{job_id}:training:0"
sampling_id = f"{job_id}:sampling:0"

# Submit a DSSST1 training batch, then update the adapter parameters.
request_id = client.forward_backward(job_id, fwd_bwd_dssst1_frame)
client.poll_request(job_id, request_id)

request_id = client.step(job_id, learning_rate=1e-4)
client.poll_request(job_id, request_id)

# Broadcast only LoRA tensors. Later syncs update the same resident adapter.
request_id = client.weight_sync(
    job_id,
    source_sub_job_id=training_id,
    target_sub_job_ids=[sampling_id],
    weight_format="lora",
)
client.poll_request(job_id, request_id)

# Generation now uses the synchronized resident adapter.
request_id = client.generate(
    job_id,
    prompts=["Write a short proof."],
    sampling_params={"max_tokens": 256, "temperature": 0.7},
)
result = client.poll_request(job_id, request_id)
```

---

## 14. Current repository compatibility notes

These are implementation gaps, not supported API behavior:

1. `CortexTrainingEngine.backward()` still uses `torch.save`, while the
   current server wire protocol is DSSST1 and rejects pickle/torch-save frames.
2. The README still says the forward/backward CLI serializes with `torch.save`;
   the CLI helper actually uses DSSST1.
3. `tail_events()` and `stream_events()` send unsupported `zmd-events`.
4. `wait_for_job()` does not treat `terminated` as terminal.
5. `save(checkpoint_id=...)` sends a field that is absent from the adjacent
   control-plane `SaveRequest`; callers must use the id in the save result.
6. Generic `forward()` wraps binary input in a base64 JSON payload, but the
   current adjacent backend expects raw DSSST1 bytes.
7. Generate prompt validation selects the first sub-job with an
   `inference_config` rather than explicitly selecting `job_type="sampling"`;
   a preceding log-probability sub-job can supply the wrong `max_seq_len`.
8. The local mock's documented/routed endpoint is `cortex-post-training`, while
   `CortexTrainingClient` defaults to `cortex-training`.
9. The in-memory mock and its README cover only a subset of the current client
   surface and should not be used as the complete API inventory.
