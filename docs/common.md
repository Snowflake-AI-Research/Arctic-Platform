# Shared server infrastructure (`arctic_platform.common`)

Protocol-agnostic GPU backend used by **RL** today and intended for forthcoming
**SFT**: DeepSpeed workers, HTTP/Ray servers, Ray cluster helpers, loss
registries, and batch utils. RL-specific protocol docs: [`rl.md`](rl.md).
SFT client/API docs will land with the SFT PR.

```
arctic_platform/common/
├── deepspeed_worker.py   # Ray actor: DeepSpeed train / log-prob engines
├── http_server.py        # FastAPI HTTP server (uvicorn)
├── ray_server.py         # In-process Ray server (same op surface)
├── ray_cluster.py        # Ray bootstrap (attach or spawn)
├── server.py             # Minimal ArcticRLServerState base
├── registry.py           # LOSS_FNS / POST_PROCESSORS
└── utils/
    ├── batch.py          # shard, merge, metric aggregation
    ├── server_models.py  # JobConfig, request models
    ├── ray_pg.py         # colocate placement groups
    ├── cuda_ipc.py       # CUDA-IPC weight sync
    ├── debug.py          # determinism, timers, memory
    └── record_replay.py  # optional record/replay harness
```

Prefer `arctic_platform.common.*` imports. Back-compat shims still exist under
`arctic_platform.rl.{http_server,ray_server,deepspeed_worker}`.

## Launch the HTTP server

```bash
python -m arctic_platform.common.http_server \
  --host 0.0.0.0 \
  --port 7000 \
  --training-gpus 4 \
  --sampling-gpus 2 \
  --log-prob-gpus 2 \
  --log-prob-engine vllm \
  --colocate
```

| Flag | Default | Meaning |
|------|---------|---------|
| `--host` | `localhost` | Bind address |
| `--port` / `-p` | `7000` | HTTP port |
| `--training-gpus` | `0` | DeepSpeed training ranks |
| `--sampling-gpus` | `0` | vLLM sampling replicas (ArcticInference `ReplicaPool`) |
| `--log-prob-gpus` | `0` | Log-prob engine GPUs |
| `--log-prob-engine` | `vllm` | `vllm` or `deepspeed` for the log-prob job |
| `--colocate` | off | Fractional Ray GPU sharing across engines |
| `--verbose` | off | Uvicorn access logs |
| `--no-ray-auto-attach` | attach on | Always start a fresh Ray cluster |

At least one of `--training-gpus`, `--sampling-gpus`, `--log-prob-gpus` must
be > 0.

**Training-only** (no vLLM / ArcticInference required):

```bash
python -m arctic_platform.common.http_server \
  --host 0.0.0.0 --port 8765 \
  --training-gpus 2 --sampling-gpus 0 --log-prob-gpus 0
```

The server lazy-imports inference deps only when sampling/log-prob GPUs are
requested.

**Ray transport:** same op surface via `arctic_platform.common.ray_server`,
usually started by the Ray client rather than as a standalone process.

## Job types

Created with `POST /initialize` (`JobConfig.job_type`):

| `job_type` | Engine | When |
|------------|--------|------|
| `training` | DeepSpeed + optimizer | `training_gpus > 0` |
| `sampling` | vLLM + ArcticInference | `sampling_gpus > 0` |
| `log_prob` | DeepSpeed forward-only **or** vLLM | `log_prob_gpus > 0` |

Log-prob backend: DeepSpeed when a DS config is provided for that job; vLLM
when only `vllm_config` is set. Engine choice is also controlled by
`--log-prob-engine` / client `log_prob_engine`.

Client create order is `sampling` → `log_prob` → `training` so the training
NCCL rendezvous is last.

`checkpoint_path` is **required** for new training jobs (asserted at init and
at save).

## HTTP endpoints

| Endpoint | Job(s) | Purpose |
|----------|--------|---------|
| `GET /health` | — | Liveness |
| `POST /initialize` | — | Create a job |
| `POST /destroy?job_id=` | any | Tear down |
| `GET /job/{job_id}` | — | Status |
| `GET /status` | — | GPU counts + job map |
| `POST /fwd-bwd` | training | Forward + backward (octet stream) |
| `POST /fwd-no-grad` | training or log_prob | Forward only |
| `POST /step` | training | Optimizer step |
| `POST /save-checkpoint` | training | Checkpoint (`path` body overrides job dir) |
| `POST /load-checkpoint` | training | Restore engine state for resume; returns `global_step` |
| `POST /empty-training-cache` | training | Clear caches |
| `POST /generate` | sampling | Rollouts |
| `POST /log-probs` | log_prob | Reference / old log-probs |
| `POST /sync-weights` | training + sampling | Trainer → sampler sync |
| `POST /weight-norm` | training + sampling | Debug: norm after sync |
| `POST /reset-prefix-cache` | sampling | Prefix cache reset |
| `POST /sleep-inference` / `wake-inference` | sampling | VRAM time-sharing |
| `POST /sleep-training` / `wake-training` | training | Offload / reload |
| `POST /sleep-log-prob` / `wake-log-prob` | log_prob | Sleep / wake |

Ray server methods mirror these ops (same names, in-process).

## DeepSpeed worker

`DeepSpeedWorker` is a single-GPU Ray actor
(`arctic_platform.common.deepspeed_worker`).

**Engine modes**

- **`training`** — full DeepSpeed engine with optimizer (`ds_config` +
  `training_config` + `ds_worker_config`).
- **`log_prob`** — forward-only (`log_prob_config`); no optimizer state.

**Pipeline dispatch** (per `fwd_bwd` call, from `processing.loss_fn`):

```text
loss_fn → run_pipeline (arctic_platform.rl GRPO, …)
```

A forthcoming SFT PR will add an SFT loss path (`sft` / `sft_ce`) on the same
worker; that API is not present in this tree yet.

If `ds_worker_config["zorro_train_enable"]` is true at init, the worker patches
the HF model for ZoRRo Train (see [`rl.md`](rl.md#zorro-train)).

## Config blobs

Forwarded in the `/initialize` payload (`JobConfig`):

| Key | Used by | Purpose |
|-----|---------|---------|
| `model_name` | all | HF model id |
| `job_type` | init | `training` / `sampling` / `log_prob` |
| `ds_config` | training, log_prob (DS) | DeepSpeed JSON (micro-batch, ZeRO, bf16, …) |
| `training_config` | training | Optimizer, LR schedule, `training_horizon`, `max_length`, GAS |
| `log_prob_config` | log_prob (DS) | Forward-only DS settings |
| `ds_worker_config` | training / log_prob (DS) | `attn_implementation`, `zorro_train_enable`, checkpointing, … |
| `vllm_config` | sampling / log_prob (vLLM) | vLLM engine config |
| `arctic_inference_config` | sampling / log_prob (vLLM) | ZoRRo Inference, speculative decoding |
| `checkpoint_path` | training | Checkpoint directory |
| `full_determinism`, `seed` | training | Reproducibility |

`training_horizon` is the LR scheduler's total optimizer-step count
(DeepSpeed `total_num_steps`).

`build_model_config()` merges `vllm_config` with ArcticInference signals from
`arctic_inference_config` (e.g. `zorro_inference.enable` → `use_fca=True`).

## Metric aggregation

Workers emit paired metrics `{name}.sum` / `{name}.tokens`:

1. Per-rank microbatches → `combine_metric_microbatches`
2. Across DP ranks on the server → `combine_metric_shards`

Resulting client metric:

```text
metrics["loss"] = Σ(loss.sum) / Σ(loss.tokens)   # global token-mean
```

Empty token count → `0.0`. Same convention for SFT and GRPO losses.

## Colocation

With `--colocate` / `colocate=True`:

- Per-node `STRICT_PACK` placement groups (`utils/ray_pg.py`)
- Fractional Ray GPU accounting across training / sampling / log-prob
- vLLM sleep mode enabled; weight sync via NCCL or CUDA-IPC. The CUDA-IPC vs
  CPU-file strategy (`cuda_ipc` / `low_memory`) is baked onto the training
  `JobConfig` at `/initialize` and reused by every `/weight-sync`; a
  `WeightSyncRequest` may still set either field to override one call. `colocate`
  is server-launch state (`--colocate`), never a per-call weight-sync argument.

## Environment variables

| Variable | Purpose |
|----------|---------|
| `ARL_WEIGHT_SYNC_PORT` | NCCL weight-sync base port (default `29600`) |
| `MASTER_PORT` | DeepSpeed rendezvous (default `29500`; log-prob DS often `29501`) |
| `ARL_RAY_TEMP_DIR` | Ray temp dir prefix |
| `ARL_RAY_MIN_WORKER_PORT` / `ARL_RAY_MAX_WORKER_PORT` | Ray worker port range |
| `RAY_PORT` / `RAY_DASHBOARD_PORT` | Ray head |
| `ARL_LOG_DP_SHARD_TOKENS` | Debug per-DP token stats |
| `ARCTIC_INFERENCE_ENABLED` | Set when `arctic_inference_config` is present |
| `NCCL_TOPO_FILE` | Stale inherited `/proc/self/fd/*` values are dropped in the worker (avoids OFI topology deadlock) |

## Gotchas

- Multi-node: bind a routable `--host` (not `localhost`) so off-node workers
  can reach the server.
- Concurrent jobs on one host: use distinct `MASTER_PORT` /
  `ARL_WEIGHT_SYNC_PORT`.
- Training-only servers must keep `--sampling-gpus 0` unless inference deps
  are installed.
- `checkpoint_path` is mandatory for new training jobs.
- Default HTTP port for the standalone server is **7000**; the unified
  `arctic_platform.client` config defaults to **8000**; SFT demos often use
  caller-chosen ports (e.g. 8765). Pick one and stay consistent.

## Registry

`arctic_platform.common.registry` holds shared `LOSS_FNS` and
`POST_PROCESSORS`. SFT and RL processors register via decorators;
`resolve_fn()` also accepts dotted import paths.
