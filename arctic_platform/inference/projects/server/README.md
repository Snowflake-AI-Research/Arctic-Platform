## Arctic Inference Server

Multi-model inference server built on Ray and vLLM. Serves multiple models simultaneously with automatic GPU sharing, dynamic rebalancing, and per-pool concurrency control.

### Architecture

```
Client ──> FastAPI ──> Driver ──> ReplicaPool (model-a) ──> Scheduler ──> Worker 0
                         │                                            └──> Worker 1
                         └──────> ReplicaPool (model-b) ──> Scheduler ──> Worker 0
```

**Components:**

- **Driver** — Main server class. Tracks replica pools, manages GPU allocation with even sharing, and routes requests by `model_id`.
- **ReplicaPool** — Manages workers for a single model. Owns a Scheduler, handles scaling, health monitoring, weight sync, and request submission.
- **Scheduler** — Routes requests to the least-loaded worker with utilization-based concurrency adjustment.
- **Worker** — Light wrapper around a vLLM `AsyncLLM` engine.
- **Pipeline** — High-level dataset processing supporting single or multi-model workflows.

### Quick Start

```console
$ pip install -e ".[server]"
```

### Usage

#### HTTP API

Start the server, then use curl to load models and send requests:

```console
$ arctic_inference_server --port 8000
$ bash projects/server/api_example.sh
```

#### Python API (Driver)

Prepared incremental generation and explicit cancellation are documented in
[STREAMING.md](STREAMING.md). GPU validation is required before deployment.

```console
$ python projects/server/driver_example.py
$ python projects/server/driver_example_scaling.py  # dynamic GPU rebalancing demo
```

#### Python API (Pipeline)

```console
$ python projects/server/pipeline_example.py
```

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/init` | Load a model. Body: `{config: ModelConfig, model_id?: str}`. Returns `model_id`. |
| `POST` | `/generate` | Generate from prompts or token IDs. Requires `model_id`. |
| `POST` | `/sleep` | Free GPU memory for a model (drains in-flight requests first). Body: `{model_id, level?}`. |
| `POST` | `/wake_up` | Restore GPU memory and resume serving. Body: `{model_id, tags?}`. |
| `GET` | `/weights_info` | Get weight metadata for a model. Requires `model_id` query param. |
| `POST` | `/sync_weights` | Receive weights via NCCL. Requires `model_id`. |
| `POST` | `/close_weight_sync` | Close NCCL engines. Requires `model_id` query param. |
| `GET` | `/status` | GPU allocation and per-model replica status. |
| `POST` | `/shutdown` | Shut down one or all models. Pass `model_id` query param to shut down a single model. |

### Strict Router Replay

With `enable_return_routed_experts=True`, `/generate` can retain router captures
across turns. Set `extra_args.dss_router_replay_strict=true` on every turn,
provide `dss_sample_id`, and use a fresh `rr1:` replay ID for every backend
attempt. For example, the first turn's sampling parameters are:

```json
{
  "max_tokens": 32,
  "dss_sample_id": "trajectory-1",
  "dss_router_replay_id": "rr1:attempt-1",
  "extra_args": {"dss_router_replay_strict": true}
}
```

For a continuation, copy `router_replay.replay_id` from the accepted response
into `extra_args.dss_router_replay_prefix_id`, and copy `router_replay.replica`
into `extra_args.dss_router_replay_prefix_replica`. Use a new replay ID for the
continuation. The server routes it to that capture's owner, waits for capacity
there, and holds the prefix through generation and capture insertion. The
client must check `router_replay.strict == true` to confirm server support;
`prefix_len` reports how many earlier capture rows were reused.

Retained captures count toward `router_replay_max_cache_bytes` and resist
pressure eviction. Release unneeded captures with the existing
`discard_router_replay` API only after acknowledging the continuation; keeping
the earlier capture until then permits retries after a lost response. Retention
expires at `ARCTIC_ROUTER_REPLAY_TX_TTL_S` (24 hours by default; zero disables
expiry), but expiry cannot evict a prefix already held by an active request.

A missing/expired prefix or unavailable owner returns HTTP 410 instead of
falling back to freshly computed prefix routes. A full cache returns HTTP 503;
release unneeded captures or increase capacity before retrying with a new
attempt ID. Replica/service loss requires restarting or dropping the affected
trajectory; captures are not replicated. The caller must extend the same token
prefix as the earlier request. Strict replay applies to `/generate`; prepared
streaming does not accept router-replay parameters. Requests without the strict
flag retain best-effort behavior.

### Multi-Model GPU Sharing

When multiple models are loaded, GPUs are split evenly. Loading a new model automatically scales down existing pools:

```
Init model-a (4 GPUs available) → model-a gets 4 replicas
Init model-b                     → model-a scales to 2, model-b gets 2
Shutdown model-a                 → model-b scales up to 4 replicas
```

### Model Registry

Pre-tuned configs via `ModelConfig.from_registry()`:

| GPU | Model |
|-----|-------|
| H200 | Qwen3-30B-A3B, Qwen3-30B-A3B-Instruct-2507, Qwen3-235B-A22B-Instruct-2507, Qwen3-Coder-480B-A35B-Instruct |
| B200 | Qwen3-30B-A3B, Qwen3-235B-A22B-Instruct-2507 |
