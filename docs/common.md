# Shared server infrastructure (`arctic_platform.common`)

Protocol-agnostic GPU backend used by **RL** today and intended for forthcoming
**SFT**: DeepSpeed workers, the Ray server, Ray cluster helpers, loss
registries, and batch utils. RL-specific protocol docs: [`rl.md`](rl.md).
SFT client/API docs will land with the SFT PR.

```
arctic_platform/common/
├── deepspeed_worker.py   # Ray actor: DeepSpeed train / log-prob engines
├── ray_server.py         # In-process Ray server
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
`arctic_platform.rl.{ray_server,deepspeed_worker}`.

## Ray server

`arctic_platform.common.ray_server` runs in-process: the client creates an
`ArcticRLRayServerState` actor (job creation) and an `ArcticRLRayServer`
wrapper (typed async ops). At least one
of `training_gpus`, `sampling_gpus`, `log_prob_gpus` must be > 0. The server
lazy-imports inference deps only when sampling/log-prob GPUs are requested.

## Job types

Created with `initialize` (`JobConfig.job_type`):

| `job_type` | Engine | When |
|------------|--------|------|
| `training` | DeepSpeed + optimizer | `training_gpus > 0` |
| `sampling` | vLLM + ArcticInference | `sampling_gpus > 0` |
| `log_prob` | DeepSpeed forward-only **or** vLLM | `log_prob_gpus > 0` |

Log-prob backend: DeepSpeed when a DS config is provided for that job; vLLM
when only `vllm_config` is set. Engine choice is also controlled by the client
`log_prob_engine`.

Client create order is `sampling` → `log_prob` → `training` so the training
NCCL rendezvous is last.

`checkpoint_path` is **required** for new training jobs (asserted at init and
at save).

## Server ops (`ArcticRLRayServer`)

| Method | Job(s) | Purpose |
|--------|--------|---------|
| `health` | — | Liveness |
| `destroy` | any | Tear down |
| `get_job_status` / `status` | — | Job status / GPU counts + job map |
| `forward_backward` | training | Forward + backward |
| `forward` | training or log_prob | Forward only |
| `step` | training | Optimizer step |
| `save` | training | Checkpoint (`path` body overrides job dir) |
| `load_checkpoint` | training | Restore engine state for resume; returns `global_step` |
| `empty_training_cache` | training | Clear caches |
| `generate` | sampling | Rollouts |
| `log_probs` | log_prob | Reference / old log-probs |
| `weight_sync` | training + sampling | Trainer → sampler sync |
| `weight_norm` | training + sampling | Debug: norm after sync |
| `reset_prefix_cache` | sampling | Prefix cache reset |
| `sleep_inference` / `wake_inference` | sampling | VRAM time-sharing |
| `sleep_training` / `wake_training` | training | Offload / reload |
| `sleep_log_prob` / `wake_log_prob` | log_prob | Sleep / wake |

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

Forwarded in the `initialize` payload (`JobConfig`):

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

With `colocate=True`:

- Per-node `STRICT_PACK` placement groups (`utils/ray_pg.py`)
- Fractional Ray GPU accounting across training / sampling / log-prob
- vLLM sleep mode enabled; weight sync via NCCL or CUDA-IPC. The CUDA-IPC vs CPU-file strategy (`cuda_ipc` / `low_memory`) is on the training `JobConfig` at `initialize` and reused by every `weight_sync`; a `WeightSyncRequest` may set either field to override one call. `colocate` is server-creation state.

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

- Concurrent jobs on one host: use distinct `MASTER_PORT` /
  `ARL_WEIGHT_SYNC_PORT`.
- Training-only servers must keep `sampling_gpus=0` unless inference deps
  are installed.
- `checkpoint_path` is mandatory for new training jobs.

## Registry

`arctic_platform.common.registry` holds shared `LOSS_FNS` and
`POST_PROCESSORS`. SFT and RL processors register via decorators;
`resolve_fn()` also accepts dotted import paths.
