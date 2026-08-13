# Arctic Platform RL

Reinforcement-learning backend: a thin client drives three GPU engines on a
remote (or colocated) Arctic server. The RL framework keeps the training loop,
rewards, and advantage estimation; Arctic owns the heavy compute.

```
┌─────────────────────────────────────────────────────────────┐
│  RL framework (verl / SkyRL / custom loop)                  │
│  rollouts → rewards → advantages → ArcticRL client          │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP or Ray
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Arctic Platform server  (see common.md)                    │
│  • Training   — DeepSpeed + optimizer (GRPO / custom loss)  │
│  • Sampling   — vLLM + ArcticInference                      │
│  • Log-prob   — DeepSpeed forward-only or vLLM              │
└─────────────────────────────────────────────────────────────┘
```

Shared server details (CLI, endpoints, metrics, colocation):
[`common.md`](common.md). Training-only SFT (no sampling) is planned in a
forthcoming SFT PR.

## vs planned SFT

| | SFT (planned) | RL |
|---|---|---|
| Jobs | training only | training + sampling (+ optional log_prob) |
| Client | forthcoming `arctic_platform.sft` | `arctic_platform.rl` factory (async HTTP/Ray) |
| Loss path | planned `sft` / `sft_ce` | `run_pipeline` (e.g. GRPO) |
| Extra ops | — | `generate`, `log_probs`, `sync_weights`, sleep/wake |

## Entry points

**Primary (production — tests, verl adapter, recipes):**

```python
from arctic_platform.rl import ArcticRLClientConfig, create_arctic_rl_client
```

- Config: `arctic_platform.rl.config.ArcticRLClientConfig`
- Factory: `arctic_platform.rl.client.create_arctic_rl_client` → async
  `ArcticRLHTTPClient` or `ArcticRLRayClient`

**Unified sync client (migration target; subset of ops):**

```python
from arctic_platform.client import ArcticRLClient, ArcticRLClientConfig, create_arctic_rl_client
```

Prefer `arctic_platform.rl` for full RL (sleep/wake, `weight_norm`, …). See
`arctic_platform/client/UNIFICATION_NOTES.md` for what the unified client still
lacks.

## Quick start

```python
import asyncio
from arctic_platform.rl import ArcticRLClientConfig, create_arctic_rl_client

config = ArcticRLClientConfig(
    backend="local",
    model_name="Qwen/Qwen3-4B",
    comm_protocol="ray",          # or "http"
    training_gpus=8,
    sampling_gpus=8,
    log_prob_gpus=0,              # 0 = disabled
    colocate=True,
    checkpoint_path="/data-fast/my-rl-run/ckpt",  # required when training_gpus > 0
)
client = create_arctic_rl_client(config)

results = asyncio.run(client.generate(
    prompts=["Hello"],
    sampling_params={"max_tokens": 64, "temperature": 0.7},
))
client.shutdown()
```

With `backend="local"` + `comm_protocol="http"`, the client can launch a local
server subprocess (`python -m arctic_platform.common.http_server` via the
`rl.http_server` shim).

Standalone server:

```bash
python -m arctic_platform.common.http_server \
  --host 0.0.0.0 --port 7000 \
  --training-gpus 4 --sampling-gpus 4 --colocate
```

Reconnect after a Ray handoff:

```python
rc = client.reconnect_config()  # fills training/sampling/log_prob job ids
client2 = create_arctic_rl_client(rc)
```

## Config (`ArcticRLClientConfig`)

| Field | Default | Notes |
|-------|---------|-------|
| `backend` | `"local"` | `"local"` or `"dss-platform"` |
| `comm_protocol` | `"http"` | `"http"` or `"ray"` |
| `model_name` | **required** | HF id |
| `training_gpus` / `sampling_gpus` / `log_prob_gpus` | `0` | Job created iff > 0 |
| `log_prob_engine` | `"vllm"` | `"deepspeed"` or `"vllm"` |
| `colocate` | `False` | Fractional GPU sharing |
| `host` / `port` | derived | HTTP often routable IP + **7000** |
| `ds_config` | `{}` | Training DeepSpeed config |
| `training_config` | `None` | Optimizer / scheduler / `training_horizon` |
| `log_prob_ds_config` | `None` | Log-prob DeepSpeed engine |
| `ds_worker_config` | `None` | Worker knobs, including **`zorro_train_enable`** |
| `vllm_config` | `None` | Sampling / vLLM log-prob |
| `arctic_inference_config` | `None` | **`zorro_inference`**, speculative decoding |
| `checkpoint_path` | `None` | Required for new training jobs |
| `full_determinism` / `seed` | `False` / `42` | Reproducibility |
| `startup_timeout` / `job_ready_timeout` | `300` / `600` | Seconds |
| `training_job_id` / `sampling_job_id` / `log_prob_job_id` | `None` | Reconnect mode |

## Client methods

Async methods on the HTTP/Ray clients from `create_arctic_rl_client`:

| Method | Job | Notes |
|--------|-----|-------|
| `fwd_bwd(batch, processing=None)` | training | GRPO via `processing["loss_fn"]` |
| `fwd_no_grad(batch, reference_model=False)` | training or log_prob | Old / ref log-probs |
| `step()` | training | Optimizer step |
| `save_checkpoint()` | training | |
| `generate(prompts, sampling_params, …)` | sampling | Rollouts |
| `log_probs(prompts, completions, top_k=1)` | log_prob | |
| `sync_weights(cuda_ipc=False, low_memory=False)` | cross-job | NCCL or CUDA-IPC |
| `weight_norm()` | cross-job | Debug after sync |
| `reset_prefix_cache()` | sampling | |
| `sleep_inference` / `wake_inference` | sampling | Colocation VRAM |
| `sleep_training` / `wake_training` | training | Offload / reload |
| `sleep_log_prob` / `wake_log_prob` | log_prob | |
| `empty_training_cache()` | training | |
| `reconnect_config()` | — | Serializable reconnect |
| `shutdown()` | all | Destroy jobs + stop local server |

`save_weights(path)` is a stub on-prem (not implemented for disk reload yet).

## GRPO wire batch (sketch)

Unlike SFT's flat labels batch, RL `fwd_bwd` typically carries:

```python
{
    "batch": {...},
    "meta": {
        "rollout_n": int,
        "max_prompt_len": int,
        "max_response_len": int,
        "zorro_train_enable": bool,   # optional per-call override
        ...
    },
    "processing": {
        "loss_fn": "grpo",            # or dotted path
        "post": ["compute_logprobs", ...],
        "config": {"eps_clip": 0.2, ...},
    },
}
```

Log-prob tensors often use a `_shifted` suffix convention (see
`arctic_platform.rl.http_client`). Batch shapes still differ slightly across
backends — treat unification as WIP.

Metrics use the shared `{name}.sum` / `{name}.tokens` pairing; see
[`common.md`](common.md#metric-aggregation).

## ZoRRo Train

**What:** Prompt deduplication during RL forward/backward. Shared prompts are
packed once; per-response logprobs/gradients are reconstructed. Mathematically
equivalent to the naive path for supported models, with large wins on
long/shared prompts.

**Code:** `arctic_platform/rl/zorro_train/` (design + supported models:
[zorro_train/README.md](../arctic_platform/rl/zorro_train/README.md)).

**Enable (direct client / server)** — flat key on the worker config:

```python
ArcticRLClientConfig(
    ...,
    ds_worker_config={
        "zorro_train_enable": True,
        "response_len": 512,
        "max_token_len": 8192,
        "rollout_n": 8,
        "temperature": 1.0,
        "use_unpad": True,
        "logits_optimization": "none",  # "none" | "memory" | "compute"
    },
)
```

**Enable (verl Hydra)** — nested yaml (not the same string as the flat key):

```yaml
remote_backend:
  train:
    zorro_train:
      enable: True
      max_rollouts: ${actor_rollout_ref.rollout.n}
```

Shell override: `remote_backend.train.zorro_train.enable=True`.

> README shorthand `zorro_train.enable` refers to the **verl yaml** path. The
> server/client flat key is `ds_worker_config.zorro_train_enable`.

## ZoRRo Inference (Forest Cascade Attention)

**What:** During decode, groups requests that share a KV-cache prefix and runs
grouped + per-suffix attention so each shared prefix block is read once per
*group* instead of once per *request*. Equivalent to standard attention;
larger wins with longer / more-shared prefixes.

**Code:** Implemented in ArcticInference / vLLM attention — not in this
package. Activated when `arctic_inference_config` contains:

```python
arctic_inference_config={
    "zorro_inference": {"enable": True},
}
```

That maps to `ModelConfig.use_fca = True` in
`arctic_platform.common.utils.server_models`.

**verl yaml:**

```yaml
remote_backend:
  rollout:
    zorro_inference:
      enable: True
```

Requires a matching vLLM + ArcticInference build. Design/tuning:
[Forest Cascade Attention](https://github.com/snowflakedb/ArcticInference/tree/main/arctic_inference/vllm/attention).

## Framework integrations

| Framework | In-repo path | Recipes |
|-----------|--------------|---------|
| **verl** | `arctic_platform/integrations/verl/` (`ArcticRLClientWrapper`, `arctic.yaml`) | [`recipes/rl/verl/`](../recipes/rl/verl/) |
| **SkyRL** | Driven from SkyRL's `integrations/arctic_rl/` | [`recipes/rl/skyrl/`](../recipes/rl/skyrl/) |

verl bootstrap sketch:

```bash
export VERL_USE_EXTERNAL_MODULES=arctic_platform.integrations.verl.register
# hydra.searchpath += integrations/verl/config
trainer.remote_backend=arctic
```

verl integration lives in-tree; upstream merge may still be pending (see
project README).

## Examples and tests

Prefer the README snippet, `tests/rl/rl_harness.py`, and `tests/rl/test_e2e.py`
as reference. Some files under `arctic_platform/rl/examples/` are stale
(imports / sync-vs-async mismatches) — do not treat them as canonical until
cleaned up.

## Status / WIP

- Dual client stacks (`rl` async full vs `client` sync partial).
- Batch/response schema unification across backends.
- On-prem `save_weights` disk path unimplemented.
- ZoRRo Train model coverage is limited — see the supported-models list in
  `zorro_train/README.md`.
