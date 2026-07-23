# DSS Server API Specification

> **Scope.** This document describes the customer-facing Neutrino SnowAPI
> surface used by this repository's `NeutrinoClient`. It covers REST paths,
> request framing, asynchronous polling, client-visible schemas, checkpoints,
> generation, operations, and logs.
>
> **Sources used for this snapshot.** The client and wire implementation in
> `dss_client/neutrino_client.py` and `dss_client/wire.py`, the command-line
> interface in `dss_neutrino_cli.py`, their unit tests, and the adjacent Neutrino
> control-plane protocol/server are the local evidence for this document. The
> SNOWAPI OpenAPI specification remains the source of truth when it is
> available.
>
> **Important distinction.** `NeutrinoClient` is a low-level transport client.
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

---

## 2. Connection, authentication, and headers

### 2.1 Base URL

Paths in this document are relative to:

```text
https://{account_host}/api/v2/databases/{database}/schemas/{schema}/{endpoint}
```

`NeutrinoClient` defaults `endpoint` to `cortex-training`.

```python
client = NeutrinoClient.from_pat(
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

`NeutrinoClient.from_pat(...)` sends:

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

`NeutrinoClient.poll_request(job_id, request_id)` handles status normalization,
backoff, DSSST1 result decoding, and cursor-addressed result chunks.

### 3.2 Job states

Customer-facing job states are:

| State | Meaning |
|---|---|
| `pending` | Waiting for placement |
| `placing` | Infrastructure is starting |
| `running` | Ready for data-plane calls |
| `cancelled` | Cancelled by the caller |
| `terminated` | Platform teardown completed |
| `failed` | Terminal failure; inspect `reason` |

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

All values are configurable on `NeutrinoClient(...)`.

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

One auxiliary call is outside the Neutrino prefix:

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
    job_id=None,
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
  "job_id": "optional-client-id",
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
`DSS_NEUTRINO_ENABLE_DEBUG_OPTIONS` is truthy (`1`, `true`, `yes`, or `on`).
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

### 5.3 List jobs - `GET /`

Optional query:

```text
?status=running
```

REST response:

```json
{"jobs": [{"job_id": "job-id", "status": "running", "sub_jobs": []}]}
```

`NeutrinoClient.list_jobs()` returns only the `jobs` list, not the outer object.
The client forwards the status string without validating an enum.

### 5.4 Capacity - `GET /capacity`

This account-scoped endpoint takes no account id from the caller. The server
resolves the account from the authenticated session.

```json
{
  "has_reservation": true,
  "reserved_gpus": 64,
  "in_use_gpus": 8,
  "available_gpus": 56
}
```

- `has_reservation`: whether the account has reserved GPU capacity.
- `reserved_gpus`: configured reservation.
- `in_use_gpus`: GPUs used by `running` and `placing` jobs.
- `available_gpus`: remaining reservation, floored at zero and potentially
  capped by currently schedulable capacity.

Proto3 JSON may omit false or zero fields. `get_capacity()` always returns all
four keys and fills omitted values with `False` or `0`.

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
dss-neutrino checkpoints JOB_ID
```

The command prints `{"checkpoints": [...]}`. As with other commands, compact
output is selected with the global option:

```bash
dss-neutrino --compact checkpoints JOB_ID
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
from dss_client import wire

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
  "source_job_id": "optional-source-job"
}
```

`checkpoint_id` is required. `source_job_id` loads from another job's checkpoint
store. The route is asynchronous:

```python
request_id = client.load(
    job_id,
    checkpoint_id="global_step12",
    source_job_id=source_job_id,
)
result = client.poll_request(job_id, request_id)
```

The current control plane routes runtime load to a training sub-job.

### 6.5 Create-time checkpoint initialization

Create-time initialization is not `resume_from_checkpoint` in the typed
`dss-client` API. Use:

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
request submitted by the same `NeutrinoClient` instance, `poll_request()`
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
- Byte payloads passed to `NeutrinoClient.forward()` are converted to
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
chunked. The current ArcticTraining Neutrino adapter still treats no-grad
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
    "target_sub_job_ids": ["job-id:sampling:0"]
  }
}
```

`weight_sync()` defaults operation routing to the source training sub-job and
returns the `request_id` string.

### 7.3 Bootstrap router replay

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

### 7.4 Router replay discard

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

### 7.5 Reset prefix cache

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

### 7.6 Cancel request

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

### 7.7 Tail logs

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

### 7.8 Unsupported `zmd-events` client helper

`NeutrinoClient.tail_events()` and `stream_events()` currently send:

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

`extra_training` is merged as open passthrough data, without overriding typed
keys. Examples include `model_provider`, `ep_size`, `ds_config`,
`activation_checkpointing`, `prime_rl`, and `router_replay`.

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

---

## 9. DSSST1 binary wire protocol

### 9.1 Why DSSST1

DSSST1 is a safetensors-based frame implemented by `dss_client.wire`. It
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
from dss_client import wire

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

If the original frame fits, it is sent unchanged. Otherwise each DSSST1 chunk
contains:

- A `uint8` payload tensor.
- `chunk_idx` and `total_chunks`.
- `chunk_group_id`.
- Original frame size and SHA-256.
- Operation name.

The server caches intermediate chunks and schedules work after the final chunk.

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
| `step` | `global_steps`, `last_lr` |
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
uses the `tail-logs` operation described in [section 7.7](#77-tail-logs).

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
[section 7.8](#78-unsupported-zmd-events-client-helper).

---

## 13. End-to-end examples

### 13.1 Create training and sampling sub-jobs

```python
from dss_client.neutrino_client import NeutrinoClient, SubJobConfig

client = NeutrinoClient.from_pat(
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

---

## 14. Current repository compatibility notes

These are implementation gaps, not supported API behavior:

1. `NeutrinoTrainingEngine.backward()` still uses `torch.save`, while the
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
   `NeutrinoClient` defaults to `cortex-training`.
9. The in-memory mock and its README cover only a subset of the current client
   surface and should not be used as the complete API inventory.
